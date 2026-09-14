"""P0 成本账本验收：真实一轮三任务，逐调用归因 + TurnRecord 完整性。

验证：① 所有调用被归因（无 unattributed）；② 归因调用数 = 总调用数；
③ 逐调用 token 差分求和 = 总 tokens；④ TurnRecord 关键字段非空。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from cstpo.cost_ledger import (CostLedger, RecordingLLM, attribute_calls,
                               make_turn_record)
from cstpo.task_env import TaskEnv

PROBES = {"esconv": "You don't have to solve everything at once - start with one small step.",
          "p4g": "A donation of 10 dollars would help a child in need.",
          "craigslistbargain": "What's the lowest price you could do?"}


def main():
    llm = RecordingLLM()
    env = TaskEnv(llm=llm, max_turns=8)
    ledger = CostLedger()
    failed = 0

    def check(ok, name, detail=""):
        nonlocal failed
        if not ok:
            failed += 1
        print(("PASS" if ok else "FAIL"), name, "" if ok else detail)

    for task, sid in (("esconv", "esconv_01"), ("p4g", "p4g_01"),
                      ("craigslistbargain", "cb_01")):
        seed = json.loads((ROOT / "data" / "seeds_draft" / task /
                           f"{sid}.json").read_text())
        out = env.reset(seed)
        start = len(llm.call_log)
        r = env.step(out["checkpoint"], PROBES[task])
        call_slice = llm.call_log[start:]
        mode = r["turn_record"]["mode"]
        comp = attribute_calls(call_slice, mode)

        print(f"\n=== {task}（mode={mode}，{len(call_slice)} 调用）===")
        print(f"  归因: {comp}")
        check("unattributed" not in comp, "全部调用被归因",
              str(comp.get("unattributed")))
        check(sum(comp.values()) == len(call_slice), "归因调用数 = 实际调用数",
              f"{sum(comp.values())} vs {len(call_slice)}")
        # 逐调用 token 差分求和（prev = 切片前的累计值）
        diffs = {"prompt": 0, "completion": 0}
        if start > 0:
            before = llm.call_log[start - 1]
            prev = (before["cum_prompt"] or 0, before["cum_completion"] or 0)
        else:
            prev = (0, 0)
        for rec in call_slice:
            cur = (rec["cum_prompt"] or 0, rec["cum_completion"] or 0)
            diffs["prompt"] += cur[0] - prev[0]
            diffs["completion"] += cur[1] - prev[1]
            prev = cur
        check(diffs["prompt"] == r["costs"]["prompt_tokens"]
              and diffs["completion"] == r["costs"]["completion_tokens"],
              "逐调用差分 = 单轮总 tokens",
              f"{diffs} vs {r['costs']}")

        rec = make_turn_record(r, seed, f"{sid}_main", "root", True,
                               out["x"], comp)
        ledger.add(rec)
        required = [rec.seed_id, rec.group_id, rec.o_hash, rec.x_hash,
                    rec.total_cost]
        check(all(required), "TurnRecord 关键字段完整")
        check(rec.end_type in ("continue", "user_ended", "time_limit"),
              "结束类型合法", rec.end_type)

    print(f"\n=== 账本汇总 ===")
    print(json.dumps(ledger.summary(), ensure_ascii=False, indent=1))
    print(f"\n成本账本验收: {6 * 3 - failed}/{6 * 3} PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
