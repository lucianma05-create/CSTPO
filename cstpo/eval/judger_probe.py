"""冗余 judger 验证：8 条对话并行（4 种子 × cooperative/pushy）。

对照无机制基线（esconv coop-02 30 轮空聊、p4g/cb pushy 30 轮僵局）：
预期 judger 在 ≥15 轮把冗余对话裁掉并带自然告别语；<15 轮自然结束的
对话不受影响。

用法：cd CSTPO && python -m cstpo.eval.judger_probe
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient

from cstpo.core.agent import agent_turn
from cstpo.core.cost_ledger import RecordingLLM
from cstpo.core.stale_judger import check_stale, should_check
from cstpo.core.task_env import TaskEnv, prefix_turns

PLANS = [("esconv", 0, "cooperative"), ("esconv", 0, "pushy"),
         ("esconv", 1, "cooperative"), ("esconv", 1, "pushy"),
         ("p4g", 0, "cooperative"), ("p4g", 0, "pushy"),
         ("craigslistbargain", 0, "cooperative"), ("craigslistbargain", 0, "pushy")]


def run(p):
    task, seed_i, style = p
    seeds = [x for x in sorted((ROOT / "data" / "seeds_draft" / task).glob("*.json"))
             if x.name != "manifest.json"]
    seed = json.loads(seeds[seed_i].read_text())
    pre = prefix_turns(seed)
    llm, jllm = RecordingLLM(), LLMClient()
    env = TaskEnv(llm=llm, max_turns=30)
    cp = env.reset(seed)["checkpoint"]
    turns = prefix_turns(seed)
    r = None
    n_pre_ass = len([t for t in pre if t["role"] == "assistant"])
    reason, judged_at = None, None
    while not (r and r["terminated"]):
        n_ass = len([t for t in turns if t["role"] == "assistant"]) - n_pre_ass
        if should_check(n_ass):
            j = check_stale(jllm, turns, task)
            if j["stale"] and j["final_line"]:
                turns.append({"role": "user", "text": j["final_line"]})
                reason, judged_at = "judger_stale_end", n_ass
                break
        utter = agent_turn(llm, task, style, turns, seed)
        r = env.step(cp, utter)
        cp = r["checkpoint"]
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": r["user_reply"]})
    return task, style, reason or r["termination_reason"], turns[len(pre):], judged_at


def main():
    with ThreadPoolExecutor(max_workers=len(PLANS)) as ex:
        results = list(ex.map(run, PLANS))
    for task, style, term, new, ja in results:
        n = len(new) // 2
        tail_u = new[-1]["text"][:85] if new and new[-1]["role"] == "user" else ""
        print(f"== {task:18s} [{style:11s}] {n:2d} 轮 {term:16s} judger@r{ja}"
              f" | 末句: {tail_u!r}")
        if ja is not None:
            print(f"   [judger 结束语] {new[-1]['text']!r}")


if __name__ == "__main__":
    main()
