"""基线统一评估 runner。

流程：30 个冻结留出 case（judge_calibration/heldout，p4g 5 / cb 5 /
esconv 20）→ 按 seed_id 从 seeds_draft 解析完整种子（只读）→ Policy 生成
话语 → TaskEnv(Cog-Sim v1.0.1-fix) 走回合（12 回合上限，主方法评估口径）→
judge rev4（n=3 聚合）→ 逐 case 落 JSON + manifest + summary。

产物只写 baseline/runs/<method>/；外部目录只读。
每个 case 一个独立 LLMClient（线程安全 + 计数器归零 → 组件级 token 账本
干净，不受并发干扰）。跳过已存在的记录实现断点续跑（--force 重跑）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # CSTPO 仓库根
BASELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Cog-Sim"))

from cstpo.core.judge import judge_with_aggregation  # noqa: E402
from cstpo.core.stale_judger import check_stale  # noqa: E402
from cstpo.core.task_env import TaskEnv, prefix_turns  # noqa: E402
from cstpo.core.terminal_judge import raw_sl  # noqa: E402
from simulator.llm import LLMClient  # noqa: E402

from baseline.manifest import build_manifest  # noqa: E402
from baseline.policies import REGISTRY, make_policy  # noqa: E402

DEFAULTS = {
    "case_dir": REPO_ROOT / "data" / "judge_calibration" / "heldout",
    "seeds_dir": REPO_ROOT / "data" / "seeds_draft",
    "out_root": BASELINE_ROOT / "runs",
    "tasks": "esconv,craigslistbargain,p4g",
    # 用户允许至多 100 worker 并行（LLMClient 线程安全）；注意 Cog-Sim 每轮
    # 多次内部调用会把真实并发放大 5-6 倍，遇限流可 --workers 下调。
    "workers": 100,
    "max_turns": 12,
    "judge_n": 3,
}


def load_api_key() -> None:
    """只读解析 CSTPO/.env 的 DEEPSEEK_API_KEY（LLMClient 要求环境变量）。"""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
                return
    raise RuntimeError("缺少 DEEPSEEK_API_KEY（环境变量或 CSTPO/.env）")


def load_cases(case_dir: Path, seeds_dir: Path, tasks: list[str]) -> list[dict]:
    """judge_calibration 留出 case（带人工标注参考集）→
    [{task, dialogue_id, seed, seed_path}]。

    case 内置 seed 仅 {seed_id, situation, emotion} 摘要；完整种子按
    seed_id 从 seeds_dir/<task>/<seed_id>.json 解析（缺失即报错，不静默）。
    """
    cases = []
    for task in tasks:
        tdir = case_dir / task
        if not tdir.is_dir():
            raise FileNotFoundError(f"留出 case 目录不存在：{tdir}")
        for cf in sorted(tdir.glob("*.json")):
            case = json.loads(cf.read_text())
            seed_id = case["seed"]["seed_id"]
            seed_path = seeds_dir / task / f"{seed_id}.json"
            if not seed_path.exists():
                raise FileNotFoundError(
                    f"{case['dialogue_id']} 的完整种子缺失：{seed_path}")
            seed = json.loads(seed_path.read_text())
            cases.append({"task": task, "dialogue_id": case["dialogue_id"],
                          "seed": seed, "seed_path": seed_path})
    return cases


def load_cases_from_set(eval_set_path: Path, seeds_dir: Path,
                        tasks: list[str]) -> list[dict]:
    """冻结评估集（baseline/eval_set.json：每任务 30 个 test 种子）→
    同 shape 的 case 列表；dialogue_id 即 seed_id。

    test 种子本身是完整种子（data/seeds_test/ 即文件所在，含 task/
    context/BDI 全字段），直接按 eval_set 里冻结的路径加载，并校验
    sha256 与冻结时一致（防评估集被改写）。
    """
    from baseline.manifest import file_sha

    spec = json.loads(eval_set_path.read_text())
    cases = []
    for task in tasks:
        for item in spec["tasks"].get(task, []):
            seed_id = item["seed_id"]
            seed_path = REPO_ROOT / item["file"]
            if not seed_path.exists():
                raise FileNotFoundError(f"{seed_id} 的测试种子缺失：{seed_path}")
            if file_sha(seed_path) != item["sha256"]:
                raise RuntimeError(f"{seed_id} 的 sha256 与冻结评估集不符，"
                                   f"请重新 freeze_eval_set")
            cases.append({"task": task, "dialogue_id": seed_id,
                          "seed": json.loads(seed_path.read_text()),
                          "seed_path": seed_path})
    return cases


def _delta(before, after) -> dict:
    """LLMClient 计数器增量 → {calls, prompt_tokens, completion_tokens}。"""
    return {"calls": after[0] - before[0],
            "prompt_tokens": after[1] - before[1],
            "completion_tokens": after[2] - before[2]}


def _counters(llm) -> tuple:
    return (llm.calls, llm.prompt_tokens, llm.completion_tokens)


def run_case(case: dict, cfg) -> dict:
    """跑一个 case：独立客户端 + TaskEnv + Policy，返回记录 dict。

    策略骨干与环境的客户端分离：--backend vllm 时策略走本地 Qwen3-14B
    vLLM，环境（Cog-Sim 用户）与 judge 仍走 deepseek（同条件口径）。
    """
    task, dialogue_id, seed = case["task"], case["dialogue_id"], case["seed"]
    env_llm = LLMClient()
    if cfg.backend == "vllm":
        from baseline.vllm_client import VLLMClient

        policy_llm = VLLMClient(base_url=cfg.vllm_url, model=cfg.vllm_model)
    else:
        policy_llm = env_llm
    policy = make_policy(cfg.method, policy_llm)
    env = TaskEnv(llm=env_llm, max_turns=cfg.max_turns, debug=False)

    history = prefix_turns(seed)
    cp = env.reset(seed)["checkpoint"]
    sim_cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    actor_cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    stale_cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    termination, err = None, None
    try:
        n_assistant = 0
        for _ in range(cfg.max_turns):
            b = _counters(policy_llm)
            utter = policy.turn(seed, history)
            for k, v in _delta(b, _counters(policy_llm)).items():
                actor_cost[k] += v
            history.append({"role": "assistant", "text": utter})
            r = env.step(cp, utter)
            cp = r["checkpoint"]
            history.append({"role": "user", "text": r["user_reply"]})
            for k, v in r["costs"].items():
                sim_cost[k] += v
            n_assistant += 1
            if r["terminated"]:
                termination = r["termination_reason"]
                break
            # 自由结束协议：第 stale_start 轮起每 2 轮判一次冗余，
            # 判冗余则追加用户口吻结束语并终止（judger_stale_end）。
            if (cfg.stale_start > 0 and n_assistant >= cfg.stale_start
                    and (n_assistant - cfg.stale_start) % 2 == 0):
                b = _counters(env_llm)
                j = check_stale(env_llm, history, task)
                for k, v in _delta(b, _counters(env_llm)).items():
                    stale_cost[k] += v
                if j["stale"] and j["final_line"]:
                    history.append({"role": "user", "text": j["final_line"]})
                    termination = "judger_stale_end"
                    break
        if termination is None:
            termination = "time_limit"
    except Exception:
        err = traceback.format_exc()

    judge_cost = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    agg = None
    if err is None:
        b = _counters(env_llm)
        try:
            if task == "esconv":
                agg = judge_with_aggregation(
                    env_llm, task, history,
                    situation=seed["persona"]["situation_en"],
                    emotion=seed["initial_emotion"]["category"], n=cfg.judge_n)
            else:
                agg = judge_with_aggregation(env_llm, task, history, n=cfg.judge_n)
        except Exception:
            err = traceback.format_exc()
        for k, v in _delta(b, _counters(env_llm)).items():
            judge_cost[k] += v

    # 回报口径与主方法完全一致（verl_cstpo_reward.py 的 dict 版映射）：
    # esconv=(E+A)/8；p4g=commitment 且未撤回→1；cb=clip(raw_sl,0,1)。
    reward, raw_sl_value = None, None
    if agg is not None:
        if task == "esconv":
            reward = (agg["E"] + agg["A"]) / 8.0
        elif task == "p4g":
            reward = 1.0 if (agg["commitment"] and not agg["withdrawn"]) else 0.0
        else:
            price = agg.get("final_price")
            try:
                price = float(price) if price is not None else None
            except (TypeError, ValueError):
                price = None
            raw_sl_value = raw_sl(price, seed["user_task_view"]["target"],
                                  seed["actor_task_view"]["target"],
                                  agg.get("deal"))
            reward = min(max(raw_sl_value, 0.0), 1.0)

    pre_n = len(prefix_turns(seed))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": cfg.method,
        "task": task,
        "dialogue_id": dialogue_id,
        "seed_id": seed["seed_id"],
        "termination": termination,
        "n_turns": len([t for t in history if t["role"] == "assistant"]) - pre_n,
        "dialogue": history,
        "verdict": agg,
        "reward": reward,
        "raw_sl": raw_sl_value,
        "costs": {"simulator": sim_cost, "actor": actor_cost,
                  "judge": judge_cost, "stale_judger": stale_cost},
        "error": err,
    }


def summarize(records: list[dict]) -> dict:
    """按任务汇总：均值回报、终止分布、token 成本、故障。"""
    by_task: dict[str, list[dict]] = {}
    for rec in records:
        by_task.setdefault(rec["task"], []).append(rec)
    out, failures = {}, 0
    for task, recs in sorted(by_task.items()):
        ok = [r for r in recs if r["error"] is None]
        failures += len(recs) - len(ok)
        terms = Counter(r["termination"] for r in ok)
        toks = {}
        comps = sorted({c for r in ok for c in r["costs"]})
        zero = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for comp in comps:
            toks[comp] = {
                "calls": sum(r["costs"].get(comp, zero)["calls"] for r in ok),
                "prompt_tokens": sum(r["costs"].get(comp, zero)["prompt_tokens"] for r in ok),
                "completion_tokens": sum(r["costs"].get(comp, zero)["completion_tokens"] for r in ok)}
        out[task] = {
            "n": len(ok),
            "mean_reward": (sum(r["reward"] for r in ok) / len(ok)) if ok else None,
            "termination": dict(terms),
            "tokens": toks,
        }
    out["_failures"] = failures
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="基线统一评估 runner")
    ap.add_argument("--method", required=True, help="策略注册名（如 standard）")
    ap.add_argument("--tasks", default=DEFAULTS["tasks"],
                    help="逗号分隔，默认 esconv,craigslistbargain,p4g")
    ap.add_argument("--case-dir", type=Path, default=DEFAULTS["case_dir"])
    ap.add_argument("--eval-set", type=Path, default=None,
                    help="冻结评估集 JSON（baseline/eval_set.json）")
    ap.add_argument("--seeds-dir", type=Path, default=DEFAULTS["seeds_dir"])
    ap.add_argument("--out-root", type=Path, default=DEFAULTS["out_root"])
    ap.add_argument("--tag", default="calib30",
                    help="评估集标签，产物写到 runs/<tag>/<method>/")
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--max-turns", type=int, default=DEFAULTS["max_turns"])
    ap.add_argument("--stale-start", type=int, default=0,
                    help="自由结束协议：第 N 轮起每 2 轮判一次冗余"
                         "（0 = 关闭，固定轮数上限）")
    ap.add_argument("--judge-n", type=int, default=DEFAULTS["judge_n"])
    ap.add_argument("--backend", choices=["api", "vllm"], default="api",
                    help="策略骨干：api=deepseek-flash（默认），vllm=本地服务")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--vllm-model", default="/publicdata/model/Qwen3-14B")
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑前 N 个 case（smoke）")
    ap.add_argument("--force", action="store_true", help="重跑已有记录")
    args = ap.parse_args()

    load_api_key()
    tasks = args.tasks.split(",")
    if args.eval_set is not None:
        cases = load_cases_from_set(args.eval_set, args.seeds_dir, tasks)
    else:
        cases = load_cases(args.case_dir, args.seeds_dir, tasks)
    if args.limit:
        cases = cases[: args.limit]
    # 底座从策略注册表读取（提示族 = deepseek-flash）；vllm 后端覆盖
    args.backbone = REGISTRY[args.method].backbone
    if args.backend == "vllm":
        args.backbone = f"{args.vllm_model} (vllm)"
        # DialogXpert 先验客户端据此切到本地骨干
        os.environ["CSTPO_VLLM_URL"] = args.vllm_url
        os.environ["CSTPO_VLLM_MODEL"] = args.vllm_model

    out_dir = args.out_root / args.tag / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases
            if args.force or not (out_dir / c["task"] / f"{c['dialogue_id']}.json").exists()]
    if todo:
        manifest = build_manifest(args, cases)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(f"[runner] {len(cases)} 个 case 均已跑过（--force 可重跑），"
              "仅重算 summary")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_case, c, args): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            rec = fut.result()
            (out_dir / c["task"]).mkdir(parents=True, exist_ok=True)
            (out_dir / c["task"] / f"{c['dialogue_id']}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2))
            done += 1
            print(f"[{done}/{len(todo)}] {c['dialogue_id']} "
                  f"termination={rec['termination']} reward={rec['reward']}")

    records = []
    for jf in sorted(out_dir.rglob("*.json")):
        if jf.name in ("manifest.json", "summary.json"):
            continue
        records.append(json.loads(jf.read_text()))
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "method": args.method, **summarize(records)}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
