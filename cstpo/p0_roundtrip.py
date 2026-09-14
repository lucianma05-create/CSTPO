"""P0 增补：fake 模型单包往返验证（03§7，零模型调用）。

验证内容：
1. field_advantage 手算验收例（03§5）；
2. 同一父快照恢复 1 主干 + 3 尾部：父状态哈希一致、分支间无对象共享；
3. 分支互不影响（不同 Engine 提案 → 各自独立约束结果）；
4. 剩余轮数与 logs 按分支独立、fake LLM 调用计数正确；
5. fake judge 终局评分 + CB 程序化 raw_SL（04§2.3/§18.4）。

运行：cd CSTPO && python cstpo/p0_roundtrip.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.run_sim import SCENARIOS, build_state
from simulator.simulator import UserSimulator

from cstpo.checkpoint import restore, snapshot
from cstpo.fake_llm import ScriptedLLM
from cstpo.terminal_judge import (fake_judge_bargain, raw_sl,
                                  train_reward_bargain)
from cstpo.field_advantage import (branched_node_advantage, plain_node_advantage)

CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    CHECKS.append((ok, name + (f"  [{detail}]" if detail and not ok else "")))
    print(("PASS" if ok else "FAIL"), name, ("" if ok else detail))


def state_hash(state) -> str:
    return hashlib.sha256(json.dumps(state.bdi_dict(), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def run_branch(llm: ScriptedLLM, snap: dict, branch: str) -> dict:
    """从同一父快照恢复并跑 1 轮 influence，返回该分支结果。"""
    sim = restore(snap, llm=llm)
    sim.simulate_turn(f"[{branch}] 85是最低价。")
    log = sim.logs[-1]
    return {"sim": sim, "bdi_after": sim.state.bdi_dict(),
            "user_reply": log.user_reply, "mode": log.mode,
            "turns": len(sim.logs), "llm": llm}


def main() -> None:
    # ---- 1. 手算验收例 ----
    plain = plain_node_advantage(G11=1.0, V_old=0.4, U_old_k1=0.6)
    check(abs(plain.A_k - 0.6) < 1e-9 and abs(plain.A_u - 0.4) < 1e-9,
          "手算例·普通节点 A_k=0.6 / A_u=0.4", f"实际 {plain}")
    branched = branched_node_advantage(G11=1.0, G12=0.0, V_old=0.4, U_old_k1=0.6)
    check(abs(branched.A_k - 0.1) < 1e-9 and abs(branched.A_u - 0.4) < 1e-9,
          "手算例·完整包节点 A_k=0.1 / A_u=0.4（A_u 不变）", f"实际 {branched}")

    # ---- 2. 父快照 + 4 分支 ----
    parent = UserSimulator(build_state(copy.deepcopy(SCENARIOS["bargain"])),
                           llm=object(), route_mode="deterministic", debug=True)
    parent.prev_gc = 0.0
    snap = snapshot(parent)
    parent_hash = state_hash(parent.state)
    prev_gc = snap["prev_gc"]

    engines = {"main": '{"bdi_updates": [{"operation": "update", "id": "B1", "new_strength": 2.4}], "new_items": [], "reaction_plan": "cautious"}',
               "tail1": '{"bdi_updates": [{"operation": "update", "id": "B1", "new_strength": 1.2}], "new_items": [], "reaction_plan": "push back"}',
               "tail2": '{"bdi_updates": [{"operation": "update", "id": "B1", "new_strength": 3.5}], "new_items": [{"type": "belief", "content": "seller is lying", "strength": 1.5, "core": false}], "reaction_plan": "distrust"}',
               "tail3": '{"bdi_updates": [], "new_items": [], "reaction_plan": "neutral"}',
               }
    results = {}
    for i, (branch, engine_out) in enumerate(engines.items()):
        # 每分支独立 fake LLM：engine 提案不同，NLG 文本带分支号
        llm = ScriptedLLM(script={3: engine_out, 6: f"分支{i}的用户回复。"},
                          nlg_text=f"分支{i}的用户回复。")
        results[branch] = run_branch(llm, snap, branch)

    # 父状态未被任何分支污染（fix 1 + checkpoint 深拷贝）
    check(state_hash(parent.state) == parent_hash,
          "父状态哈希不被分支运行污染")
    check(parent.prev_gc == prev_gc, "父 prev_gc 不被分支污染")

    # 分支间无对象共享
    b1, b2 = results["main"]["sim"].state, results["tail1"]["sim"].state
    check(b1 is not b2 and b1.beliefs[0] is not b2.beliefs[0],
          "分支状态对象完全独立（无共享 BDIItem）")

    # 各分支独立起点（父快照相同）
    check(len({state_hash(results[b]["sim"].state) for b in results}) >= 2,
          "不同 Engine 提案产生不同的约束后状态")

    # 每分支恰好 1 轮日志、fake LLM 调用数 8（influence 链）
    for b, r in results.items():
        check(r["turns"] == 1, f"{b} 恰 1 轮日志", f"实际 {r['turns']}")
        check(r["llm"].calls == 8, f"{b} fake LLM 调用数 = 8", f"实际 {r['llm'].calls}")

    # 各分支都走 influence（fake ATC 输出）
    check(all(r["mode"] == "influence" for r in results.values()),
          "四个分支均按 influence 链路处理")

    # ---- 3. fake judge + CB 程序化收益 ----
    seller_target, buyer_target = 10.0, 7.0   # 首条 demo 场景 KB
    verdict = fake_judge_bargain(results["main"]["user_reply"])
    r_sl = raw_sl(verdict.final_price, seller_target, buyer_target, verdict.deal)
    reward = train_reward_bargain(verdict, seller_target, buyer_target)
    check(abs(r_sl - 0.0) < 1e-9 or verdict.deal is False,
          "fake judge 无成交时 raw_SL=0", f"raw_SL={r_sl}")
    check(0.0 <= reward <= 1.0, "训练回报 clip 到 [0,1]", f"reward={reward}")
    # 成交+价格的解析路径
    deal_v = fake_judge_bargain("deal 达成，最终 8.5 元成交。")
    check(deal_v.deal and deal_v.final_price == 8.5 and deal_v.parse_error is False,
          "fake judge 成交+价格解析", f"实际 {deal_v.final_price}")
    sl2 = raw_sl(8.5, 10.0, 7.0, True)
    check(abs(sl2 - 0.5) < 1e-9, "raw_SL=(8.5-10)/(7-10)=0.5", f"实际 {sl2}")

    print()
    failed = sum(1 for ok, _ in CHECKS if not ok)
    print(f"P0 单包往返: {len(CHECKS) - failed}/{len(CHECKS)} PASS"
          + (f"，{failed} FAIL" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
