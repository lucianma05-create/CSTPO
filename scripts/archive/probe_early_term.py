"""排查 val 中 1 轮终止的种子：固定话语 vs 前缀强度。

用温和固定话语跑 2 轮 simulator：若仍提前终止 → 前缀本身导致
（种子问题，评估时剔除）；若不终止 → 模型首轮话语导致（训练问题）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from cstpo.core.task_env import TaskEnv  # noqa: E402

seeds = sorted((ROOT / "data" / "seeds_draft" / "esconv").glob("*.json"))[:8]
for fp in seeds:
    seed = json.loads(fp.read_text())
    env = TaskEnv(max_turns=12, debug=False)
    cp = env.reset(seed)["checkpoint"]
    r = None
    for i in range(2):
        r = env.step(cp, "I hear you. Tell me more about how you feel.")
        cp = r["checkpoint"]
        if r["terminated"]:
            print(f"{seed['seed_id']}: TERMINATED at turn {i+1} ({r['termination_reason']})")
            break
    else:
        print(f"{seed['seed_id']}: 2 turns OK, last_reply={r['user_reply'][:60]!r}")
