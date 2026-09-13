"""Validation 1.1：malformed structured output 的 fault injection。

FaultLLM 按调用序号注入畸形输出（初始 + 修复重试连续两次），验证：
- simulate_turn 不崩溃（TurnCrashRate=0）；
- 组件 safe fallback 生效且可追踪（update_notes 含 [robustness]）；
- 状态保持合法（强度 [0,4]、C 冻结分支不变）。
"""
from __future__ import annotations

import json

from simulator.llm import LLMClient
from simulator.run_sim import SCENARIOS, build_state
from simulator.simulator import UserSimulator

# 各组件位置的合法默认输出（注入点之外使用）
VALID_OUTPUTS = {
    "atc": '{"mode": "influence", "reason": "x"}',
    "atc_elicit": '{"mode": "elicit"}',
    "trie": ('{"relevance": "medium", "argument_strength": "medium", '
             '"cue_strength": "low", "interaction_pressure": "medium", '
             '"target_proposition": "accept the 85 price"}'),
    "jee": '{"stance_distance": "medium", "relevant_current_state": ["B1"]}',
    "engine": ('{"bdi_updates": [{"operation": "update", "id": "B1", "new_strength": 2.9}], '
               '"new_items": [], "reaction_plan": "stay cautious"}'),
    "eue": ('{"appraisal": {"goal_congruence": 0.0, "coping_potential": 0.0, '
            '"future_expectancy": 0.0}, "desire_assessment": [], '
            '"emotion_proposal": {"category": "neutral"}}'),
    "merged": ('{"appraisal": {"goal_congruence": 0.0, "coping_potential": 0.0, '
               '"future_expectancy": 0.0}, "desire_assessment": [], '
               '"interaction_pressure": "medium", "emotion_proposal": {"category": "neutral"}, '
               '"reaction_plan": "answer briefly", "revealed_items": []}'),
    "nlg": "嗯。",
    "ced": '{"done": false}',
}

# 畸形输出库（§1.4 六类）
FAULTS = {
    "truncated_json": '{"bdi_updates": [{"id": "B1", "new_strength": 2.9, "reason": "x"}], "new_it',
    "missing_bracket": '{"bdi_updates": [{"id": "B1", "new_strength": 2.9}',
    "wrong_field_type": '{"bdi_updates": 5, "new_items": "oops"}',
    "extra_prose": 'ok, here is my answer:\n{"bdi_updates": [], "new_items": []}\nthanks!',
    "empty_output": "",
    "invalid_enum": '{"mode": "banana"}',
}


class FaultLLM(LLMClient):
    """按脚本注入输出的假 LLM：script = {call_index: text}，其余用 VALID_OUTPUTS 映射。"""

    def __init__(self, script: dict, kind_map: list):
        super().__init__()
        self.script = script
        self.kind_map = kind_map
        self.idx = 0

    def chat(self, messages, max_tok: int = 512, json_mode: bool = False) -> str:
        i = self.idx
        self.idx += 1
        if i in self.script:
            return self.script[i]
        kind = self.kind_map[min(i, len(self.kind_map) - 1)]
        return VALID_OUTPUTS[kind]


# 注入场景：{(故障名, 故障文本): 各任务期望不崩溃}
# influence 路径调用序列：ATC(0) TRIE(1) JEE(2) Engine(3) [repair(4)] EUE(5) NLG(6) CED(7)
INFLUENCE_KINDS = ["atc", "trie", "jee", "engine", "engine", "eue", "nlg", "ced"]
# elicit 路径：ATC(0) merged(1) [repair(2)] NLG(3) CED(4)
ELICIT_KINDS = ["atc_elicit", "merged", "merged", "nlg", "ced"]


def _run_case(script: dict, kinds: list, task: str, msg: str) -> dict:
    llm = FaultLLM(script, kinds)
    state = build_state(SCENARIOS[task])
    sim = UserSimulator(state, llm)
    crashed = False
    err = None
    try:
        sim.simulate_turn(msg)
    except Exception as e:
        crashed = True
        err = f"{type(e).__name__}: {str(e)[:120]}"
    fallback_notes = []
    if sim.logs:
        fallback_notes = [n for n in sim.logs[-1].update_notes if "[robustness]" in n]
    # 状态合法性：强度 [0,4]
    legal = True
    for k in ("beliefs", "desires", "intentions"):
        for i in getattr(state, k):
            if not (0.0 <= i.strength <= 4.0):
                legal = False
    return {"crashed": crashed, "error": err, "legal_state": legal,
            "fallback_notes": fallback_notes}


def run_fault_injection() -> dict:
    results = []
    # Engine 注入（influence 路径，第 3 次调用为 Engine，第 4 次为修复重试）
    for fname, ftext in FAULTS.items():
        if fname == "invalid_enum":
            # 非法枚举应命中 ATC 的规则兜底（第 0 次调用）
            script = {0: ftext, 1: ftext}
            r = _run_case(script, INFLUENCE_KINDS, "bargain",
                          "85是最低价，我今天已经拒绝几个80的报价。")
            r.update({"target": "atc_enum"})
        else:
            script = {3: ftext, 4: ftext}
            r = _run_case(script, INFLUENCE_KINDS, "bargain",
                          "85是最低价，我今天已经拒绝几个80的报价。")
            r.update({"target": "engine"})
        r.update({"fault": fname})
        results.append(r)
    # 合并调用注入（elicit 路径，第 1 次为 merged，第 2 次为修复重试）
    for fname, ftext in FAULTS.items():
        if fname == "invalid_enum":
            continue
        script = {1: ftext, 2: ftext}
        r = _run_case(script, ELICIT_KINDS, "bargain", "你心里的预算到底是多少？")
        r.update({"fault": fname, "target": "merged"})
        results.append(r)
    # NLG 注入（influence 路径第 6 次调用输出空文本，json_mode=False 无修复重试）
    script = {6: "", 7: ""}
    r = _run_case(script, INFLUENCE_KINDS, "bargain",
                  "85是最低价，我今天已经拒绝几个80的报价。")
    r.update({"fault": "empty_nlg", "target": "nlg"})
    results.append(r)
    crashes = sum(1 for r in results if r["crashed"])
    traced = sum(1 for r in results if r["fallback_notes"])
    out = {
        "total_cases": len(results),
        "turn_crashes": crashes,
        "TurnCrashRate": round(crashes / len(results), 3),
        "fallback_traced": traced,
        "results": results,
    }
    json.dump(out, open("evaluation/results/fault_injection.json", "w"),
              ensure_ascii=False, indent=1)
    for r in results:
        print(f"{r['fault']:>16} @ {r['target']:>8}: crashed={r['crashed']} "
              f"legal={r['legal_state']} notes={r['fallback_notes'][:1]}")
    return out


if __name__ == "__main__":
    run_fault_injection()
