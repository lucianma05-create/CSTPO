"""CSTPO 终局 reward（verl 0.8 naive reward manager 的 compute_score 契约）。

签名（naive.py run_single 约定）：
  compute_score(data_source, solution_str, ground_truth, extra_info) -> dict
  必须含 "score"；其余键进 reward_extra_info。

extra_info 由 CstpoAgentLoop 的 extra_fields 传入：
  dialogue: 完整对话 turns（含前缀轮，不含策略标签）
  seed_json: 种子原始 JSON
  termination: 终止原因

回报口径（04§2，与 verl_env_adapter.compute_reward 一致）：
  esconv: (E + A) / 8
  p4g: commitment 且未 withdraw → 1，否则 0
  cb: clip(raw_SL, 0, 1)
judge 走 judge_with_aggregation（3 次独立评分：数值平均、布尔多数投票）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "Cog-Sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cstpo.core.judge import judge_with_aggregation
from simulator.llm import LLMClient

# rollout 轨迹落盘（诊断分布漂移用）：logs/rollout_traj/{task}.jsonl
# 并行实验时用 CSTPO_TRAJ_DIR 分目录，避免交错归因
TRAJ_DIR = Path(os.environ.get("CSTPO_TRAJ_DIR", str(ROOT / "logs" / "rollout_traj")))
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


def _dump_trajectory(task: str, rec: dict) -> None:
    """每轨迹一行 append（ray 多进程低并发下单行写原子性足够）。"""
    try:
        with open(TRAJ_DIR / f"{task}.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass  # 落盘失败不影响训练


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    """终局 judge 聚合 → 任务回报。"""
    extra_info = extra_info or {}
    dialogue = extra_info.get("dialogue")
    seed = json.loads(extra_info["seed_json"])
    task = seed["task"]["task_id"]

    if not dialogue:
        return {"score": 0.0, "error": "empty_dialogue"}

    situation = seed.get("persona", {}).get("situation_en")
    emotion = seed.get("initial_emotion", {}).get("category")
    agg = judge_with_aggregation(LLMClient(), task, dialogue, situation, emotion)

    rec = {
        "seed_id": seed.get("seed_id"),
        "dialogue": dialogue,
        "labels": extra_info.get("labels"),
        "termination": extra_info.get("termination"),
        "global_step": extra_info.get("global_step", -1),
        "is_validate": extra_info.get("is_validate"),
        "E": agg.get("E"), "A": agg.get("A"),
    }
    if task == "esconv":
        rec["G"] = (agg["E"] + agg["A"]) / 8.0
    elif task == "p4g":
        rec["commitment"] = agg.get("commitment")
        rec["withdrawn"] = agg.get("withdrawn")
    else:
        rec["deal"] = agg.get("deal")
        rec["raw_sl"] = agg.get("raw_sl")
    _dump_trajectory(task, rec)

    if task == "esconv":
        return {"score": (agg["E"] + agg["A"]) / 8.0,
                "E": agg["E"], "A": agg["A"],
                "termination": extra_info.get("termination")}
    if task == "p4g":
        r = 1.0 if (agg["commitment"] and not agg["withdrawn"]) else 0.0
        return {"score": r, "commitment": agg["commitment"],
                "withdrawn": agg["withdrawn"],
                "termination": extra_info.get("termination")}
    # cb：raw_SL clip（程序化，不进 judge 可见信息）
    from cstpo.core.terminal_judge import raw_sl
    p_buyer = seed["actor_task_view"]["target"]
    p_seller = seed["user_task_view"]["target"]
    price = agg.get("final_price")
    try:
        price = float(price) if price is not None else None
    except (TypeError, ValueError):
        price = None
    sl = raw_sl(price, p_seller, p_buyer, agg.get("deal"))
    return {"score": min(max(sl, 0.0), 1.0), "deal": agg["deal"],
            "raw_sl": sl, "termination": extra_info.get("termination")}
