"""四维 judge 批跑 + 逐维度 win/lose 统计 + 人工盲评表单生成。

数据：data/diversity_probe/ 已有对话（统一情景 = 同任务×同种子×同 agent
风格，四档模拟器分别与同一 agent 对话）。四档 =
cogsim / std_persona / std_persona_resist / std_bdi（P2 梯度对照）。

win/lose 定义：同一情景格内，逐维度比 Cog-Sim 与对照档的 0-2 分——
高 = win，低 = lose，同 = tie。跨情景聚合，按 (任务, 风格, 维度, 对手)
分层报告，seed 聚类 bootstrap 95% CI（(win−lose)/n）。

产物：
- baseline/runs/cogsim_eval/judge4_scores.jsonl  逐对话四维分
- baseline/runs/cogsim_eval/win_lose.json         分层 win/lose 统计
- baseline/runs/cogsim_eval/blind_form.md         人工盲评表单（编号打乱）
- baseline/runs/cogsim_eval/blind_map.json        编号→来源映射（评后揭盲用）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient  # noqa: E402

from baseline.cogsim_eval.judge4 import judge_dialogue  # noqa: E402
from baseline.runner import load_api_key  # noqa: E402

DATA_DIR = ROOT / "data" / "diversity_probe"
OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "cogsim_eval"
SIMS = ["cogsim", "std_persona", "std_persona_resist", "std_bdi"]
DIMS = ["drift", "rigidity", "contradiction", "consistency"]
TASKS = ["esconv", "p4g", "craigslistbargain"]


def load_dialogues() -> list[dict]:
    """四档模拟器 × 同情景格。格 = (task, style, seed)；同格内四档各一条。"""
    by_cell = defaultdict(dict)
    for sim in SIMS:
        for jf in (DATA_DIR / sim).rglob("*.json"):
            if jf.name in ("summary.json",):
                continue
            d = json.loads(jf.read_text())
            cell = (d["task"], d["style"], d["seed"]["seed_id"])
            by_cell[cell][sim] = d
    out = []
    for cell, sims in sorted(by_cell.items()):
        if len(sims) == len(SIMS):  # 只保留四档齐全的完整格
            out.append({"cell": cell, "sims": {k: v["turns"] for k, v in sims.items()}})
    return out


def score_all(cells: list[dict], workers: int = 50) -> dict:
    """逐对话四维打分（跳已有）。key = f"{sim}|{task}|{style}|{seed}"。"""
    scores = {}
    scores_file = OUT_DIR / "judge4_scores.jsonl"
    if scores_file.exists():
        for line in scores_file.read_text().splitlines():
            s = json.loads(line)
            scores[f"{s['sim']}|{s['task']}|{s['style']}|{s['seed_id']}"] = s
    todo = []
    for c in cells:
        task, style, seed = c["cell"]
        for sim in SIMS:
            key = f"{sim}|{task}|{style}|{seed}"
            if key not in scores:
                todo.append((key, task, c["sims"][sim], seed))
    print(f"[judge4] 已有 {len(scores)}，待评 {len(todo)}", flush=True)

    def work(item):
        key, task, turns, seed = item
        llm = LLMClient()
        s = judge_dialogue(llm, task, turns, seed)
        s["sim"] = key.split("|")[0]
        s["style"] = key.split("|")[2]
        return key, s

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, item) for item in todo]
        done = 0
        for fut in as_completed(futs):
            key, s = fut.result()
            scores[key] = s
            done += 1
            if done % 50 == 0:
                print(f"[judge4] {done}/{len(todo)}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(scores_file, "w") as f:
        for key in sorted(scores):
            f.write(json.dumps(scores[key], ensure_ascii=False) + "\n")
    return scores


def win_lose(scores: dict, cells: list[dict]) -> dict:
    """同格内 Cog-Sim vs 对照档逐维度 win/lose；按 (任务,风格,维度,对手)
    分层 + 全量池化，seed 聚类 bootstrap 95% CI。"""
    rng = np.random.default_rng(20260920)
    per = defaultdict(lambda: defaultdict(list))  # (task,style,dim,opp) -> [per-cell diff]
    for c in cells:
        task, style, seed = c["cell"]
        base = scores.get(f"cogsim|{task}|{style}|{seed}")
        if base is None:
            continue
        for opp in SIMS[1:]:
            s = scores.get(f"{opp}|{task}|{style}|{seed}")
            if s is None:
                continue
            for dim in DIMS:
                # 四维都是"毛病分"（越低越好）：对照分 − cogsim 分 > 0
                # 即 cogsim 毛病更少 = win
                d = s[dim] - base[dim]
                per[(task, style, dim, opp)][seed].append(
                    d if d > 0 else (0 if d == 0 else -1))
    out = {}
    for (task, style, dim, opp), by_seed in per.items():
        diffs = [d for seed_d in by_seed.values() for d in seed_d]
        n = len(diffs)
        win = sum(1 for d in diffs if d > 0)
        lose = sum(1 for d in diffs if d < 0)
        tie = n - win - lose
        # seed 聚类 bootstrap：以格为抽样单元
        cells_diffs = list(by_seed.values())
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(cells_diffs), len(cells_diffs))
            sample = [d for i in idx for d in cells_diffs[i]]
            if sample:
                w = sum(1 for d in sample if d > 0)
                l = sum(1 for d in sample if d < 0)
                boots.append((w - l) / len(sample))
        boots = np.asarray(boots)
        out[f"{task}|{style}|{dim}|{opp}"] = {
            "n": n, "win": win, "lose": lose, "tie": tie,
            "win_rate": round(win / n, 3), "lose_rate": round(lose / n, 3),
            "wl_diff": round((win - lose) / n, 3),
            "ci95_wl": [round(float(np.percentile(boots, 2.5)), 3),
                        round(float(np.percentile(boots, 97.5)), 3)],
        }
    return out


def blind_form(cells: list[dict], n_quads: int | None = None) -> None:
    """人工盲评表单：每格四档对话随机编号打乱，附四维评分卡与说明。"""
    rng = random.Random(20260921)
    quads = cells if n_quads is None else rng.sample(cells, n_quads)
    items, mapping = [], {}
    for c in quads:
        for sim in SIMS:
            items.append((sim, c["cell"], c["sims"][sim]))
    rng.shuffle(items)
    lines = ["# Cog-Sim 拟真度人工盲评表单", "",
             "请逐条阅读完整对话，只评**用户侧行为**，按四个维度打分（各 0/1/2），",
             "并选出该情景内四条中**最像真人**的一条（只填序号，可并列）。", "",
             "评分卡：", "- 无依据改变：agent 没给新证据，用户态度就变了（0=无 1=轻微 2=明显）",
             "- 有依据却僵化：agent 给了相关新证据/有效帮助，用户无合理反应（0=无 1=部分 2=明显）",
             "- 历史/画像矛盾：回复与用户之前的话或自身处境矛盾（0=无 1=轻微 2=明显）",
             "- 表达一致性：语言与内容/情绪状态一致（0=一致 1=略有出入 2=明显不一致）", ""]
    for i, (sim, (task, style, seed), turns) in enumerate(items, 1):
        mapping[i] = {"sim": sim, "task": task, "style": style, "seed_id": seed}
        lines.append(f"## 对话 {i}（任务 {task}）")
        for t in turns:
            lines.append(f"- [{t['role']}] {t['text']}")
        lines.append("")
        lines.append("无依据改变 ___  有依据却僵化 ___  历史/画像矛盾 ___  表达一致性 ___")
        lines.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "blind_form.md").write_text("\n".join(lines))
    (OUT_DIR / "blind_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="四维 judge 批跑 + win/lose + 盲评表单")
    ap.add_argument("--workers", type=int, default=50)
    ap.add_argument("--blind-quads", type=int, default=None,
                    help="盲评表单抽样格数（默认全部）")
    args = ap.parse_args()
    load_api_key()
    cells = load_dialogues()
    print(f"[run] 完整格 {len(cells)}（每格四档对话）")
    scores = score_all(cells, args.workers)
    stats = win_lose(scores, cells)
    (OUT_DIR / "win_lose.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print("\n=== win/lose 汇总（cogsim − 对照，按任务池化）===")
    agg = defaultdict(lambda: {"w": 0, "l": 0, "t": 0})
    for k, v in stats.items():
        task, style, dim, opp = k.split("|")
        if style == "pooled":
            continue
        agg[(dim, opp)]["w"] += v["win"]
        agg[(dim, opp)]["l"] += v["lose"]
        agg[(dim, opp)]["t"] += v["tie"]
    for (dim, opp), v in sorted(agg.items()):
        n = v["w"] + v["l"] + v["t"]
        print(f"{dim:<14} vs {opp:<20} W={v['w']:>3} L={v['l']:>3} T={v['t']:>3} "
              f"WL={(v['w']-v['l'])/n:+.3f} (n={n})")
    blind_form(cells, args.blind_quads)
    print(f"\n产物已写 {OUT_DIR}")


if __name__ == "__main__":
    main()
