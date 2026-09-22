"""P0 连通测试：真实模型（deepseek-flash）单包往返（03§7/04§7）。

每任务取 1 个冻结种子：1 主干 + 3 尾部（各自独立恢复父快照后走 1 轮），
验证：真实调用不崩溃、父快照哈希一致、分支隔离、终止原因与逐轮成本记录。
仅测链路，不构成科学结果（04§7）。

用法：cd CSTPO && python -m cstpo.p0.p0_real
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient

from cstpo.core.task_env import TaskEnv

PROBES = {
    "esconv": [
        "You don't have to solve everything at once - start with one small step.",
        "Maybe try focusing on one thing you can control, like sending your advisor a short message.",
        "Have you considered talking to a counselor about this?",
        "That sounds really tough, I'm sorry you're going through this.",
    ],
    "p4g": [
        "A donation of 10 dollars would help a child in need.",
        "Even a small donation can make a real difference for these children.",
        "Would you consider donating to support children's education?",
        "Many people find that donating gives them a sense of purpose.",
    ],
    "craigslistbargain": [
        "I can offer $300 for the phone.",
        "Would you take $300 for it?",
        "What's the lowest price you could do?",
        "I'm on a tight budget, can you go any lower?",
    ],
}

SEED_PICK = {"esconv": "esconv_01", "p4g": "p4g_01",
             "craigslistbargain": "cb_01"}


def snap_hash(cp: dict) -> str:
    return hashlib.sha256(json.dumps(cp["sim_snapshot"],
                                     sort_keys=True,
                                     default=str).encode()).hexdigest()[:12]


def main():
    llm = LLMClient()
    env = TaskEnv(llm=llm, max_turns=8)
    checks = []
    for task, sid in SEED_PICK.items():
        seed = json.loads((ROOT / "data" / "seeds_draft" / task /
                           f"{sid}.json").read_text())
        out = env.reset(seed)
        parent = out["checkpoint"]
        parent_hash = snap_hash(parent)
        print(f"\n=== {task} ({sid}) | 前缀 {len(seed['context']['prefix'])} 条 | "
              f"B/D/I {len(seed['initial_bdi']['beliefs'])}/"
              f"{len(seed['initial_bdi']['desires'])}/"
              f"{len(seed['initial_bdi']['intentions'])} ===")
        for i, probe in enumerate(PROBES[task]):
            tag = "主干" if i == 0 else f"尾部{i}"
            try:
                r = env.step(parent, probe)
                ok = (not r["terminated"] or r["termination_reason"] in
                      ("user_ended", "time_limit"))
                same_parent = snap_hash(parent) == parent_hash
                reply = r["user_reply"][:50]
                cost = r["costs"]
                checks.append((ok and same_parent and bool(reply),
                               f"{task}/{tag}"))
                print(f"  {tag}: mode={r['turn_record']['mode']:9s} "
                      f"回复=\"{reply}\" | calls={cost['calls']} "
                      f"tokens={cost['prompt_tokens'] + cost['completion_tokens']}"
                      f" | 终止={r['termination_reason']}")
                for n in r["turn_record"]["update_notes"]:
                    if "超限" in n or "截断" in n:
                        print(f"       [updater] {n[:70]}")
            except Exception as exc:
                checks.append((False, f"{task}/{tag}"))
                print(f"  {tag}: 异常 {type(exc).__name__}: {exc}")
    passed = sum(1 for ok, _ in checks if ok)
    print(f"\nP0 真实模型单包: {passed}/{len(checks)} PASS")
    print(f"总用量: {llm.usage_report()}")
    sys.exit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
