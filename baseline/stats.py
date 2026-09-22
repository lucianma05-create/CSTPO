"""跨方法统计：按 case（原始对话/用户）聚类的 cluster bootstrap 成对比较。

口径（04§6）：统计单位为原始 case，分支与重复不是独立样本；配对比较按
原始 case 聚类做 95% bootstrap 区间，报告效应量与方向，不只报 p 值；区间
宽时记 inconclusive。小样本（p4g/cb 各 5 case）只作筛查，不写结论。

用法：python -m baseline.stats --methods standard,proactive,procot,ane,mi_prompt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS_ROOT = Path(__file__).resolve().parent / "runs"
RNG = np.random.default_rng(20260920)


def load_records(method: str, tag: str) -> dict[tuple[str, str], dict]:
    """{(task, dialogue_id): record}，只取 error 为空的记录。"""
    out = {}
    for jf in (RUNS_ROOT / tag / method).rglob("*.json"):
        if jf.name in ("manifest.json", "summary.json"):
            continue
        rec = json.loads(jf.read_text())
        if rec["error"] is not None:
            continue
        out[(rec["task"], rec["dialogue_id"])] = rec
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, b_boot: int = 2000,
                     seed: int = 20260920):
    """同 case 配对的均值差 cluster bootstrap（cluster = case 本身）。

    返回 {n, mean_a, mean_b, diff, ci95_low, ci95_high, p_two_sided,
    inconclusive}；区间跨 0 且 |diff| 小 → inconclusive=True。
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return None
    diff_obs = a - b
    mean_diff = float(diff_obs.mean())
    boots = []
    for _ in range(b_boot):
        idx = rng.integers(0, n, n)
        boots.append(diff_obs[idx].mean())
    boots = np.asarray(boots)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    # 双侧 p（严格不等避免 tie 双计；全 tie → p=1；clamp 到 [0,1]）
    p_hi, p_lo = float(np.mean(boots > 0)), float(np.mean(boots < 0))
    if p_hi == 0 and p_lo == 0:
        p = 1.0
    else:
        p = min(1.0, 2 * min(p_hi, p_lo))
    inconclusive = lo < 0 < hi
    return {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": mean_diff, "ci95": [lo, hi], "p_two_sided": p,
            "inconclusive": inconclusive}


def compare(method_a: str, method_b: str, tag: str) -> dict:
    """两方法在共同 case 上的分任务成对比较（diff = a - b，即 a 相对 b）。"""
    recs_a = load_records(method_a, tag)
    recs_b = load_records(method_b, tag)
    common = sorted(set(recs_a) & set(recs_b))
    out = {"method_a": method_a, "method_b": method_b,
           "n_common": len(common), "tasks": {}}
    for task in sorted({k[0] for k in common}):
        keys = [k for k in common if k[0] == task]
        a = np.array([recs_a[k]["reward"] for k in keys], dtype=float)
        b = np.array([recs_b[k]["reward"] for k in keys], dtype=float)
        out["tasks"][task] = paired_bootstrap(a, b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="基线成对比较（cluster bootstrap）")
    ap.add_argument("--methods", required=True,
                    help="逗号分隔的方法名列表（默认顺序即基准序）")
    ap.add_argument("--tag", default="calib30",
                    help="评估集标签（runs/<tag>/<method>/）")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = RUNS_ROOT / args.tag / "comparison.json"
    methods = args.methods.split(",")

    summary = {}
    for m in methods:
        recs = load_records(m, args.tag)
        summary[m] = {"n": len(recs),
                      "by_task": {t: {"n": sum(1 for k in recs if k[0] == t),
                                      "mean_reward": float(np.mean(
                                          [r["reward"] for k, r in recs.items()
                                           if k[0] == t]))}
                                  for t in sorted({k[0] for k in recs})}}

    # 锚定第一个方法做成对比较（diff = 方法 - 基准）
    pairs = {}
    for m in methods[1:]:
        pairs[f"{m}_vs_{methods[0]}"] = compare(m, methods[0], args.tag)
    out = {"summary": summary, "pairs": pairs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
