"""P0 用确定性假 LLM：按调用序号返回脚本化输出，零网络调用。

influence 轮调用序列（与 fault_injection 的 kind 映射一致）：
ATC(0) TRIE(1) JEE(2) Engine(3) [repair(4)] EUE(5) NLG(6) CED(7)。
NLG 为纯文本（json_mode=False），其余为 JSON。

用途：fake 模型单包往返验证（03§7 P0 增补"先 fake 模型，再真实模型单包"）。
"""
from __future__ import annotations


class ScriptedLLM:
    """按脚本输出。script: {call_index: text}；miss 时用 kind 默认输出。"""

    def __init__(self, script: dict | None = None,
                 kinds: list[str] | None = None,
                 nlg_text: str = "嗯，我再想想。"):
        self.script = dict(script or {})
        self.kinds = kinds or (["atc", "trie", "jee", "engine", "engine",
                                "eue", "nlg", "ced"])
        self.nlg_text = nlg_text
        self.idx = 0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.parse_errors = 0
        self.repair_attempts = 0
        self.repair_successes = 0

    def _next(self) -> str:
        i = self.idx
        self.idx += 1
        self.calls += 1
        if i in self.script:
            return self.script[i]
        kind = self.kinds[min(i, len(self.kinds) - 1)]
        if kind == "nlg":
            return self.nlg_text
        return DEFAULT_OUTPUTS[kind]

    def chat(self, messages, max_tok: int = 512, json_mode: bool = False) -> str:
        return self._next()

    def chat_json(self, messages, max_tok: int = 512) -> dict:
        # 沿用真实客户端的契约：本地修复 + 1 次修复重试
        from simulator.llm import StructuredCallError, _extract_json, _local_repair
        text = self.chat(messages, max_tok=max_tok, json_mode=True)
        try:
            out = _extract_json(text)
            if not isinstance(out, dict):
                raise ValueError(f"JSON 输出不是对象: {type(out)}")
            return out
        except (ValueError, KeyError):
            pass
        self.parse_errors += 1
        repaired = _local_repair(text)
        if repaired is not None:
            try:
                out = _extract_json(repaired)
                if isinstance(out, dict):
                    self.repair_successes += 1
                    return out
            except (ValueError, KeyError):
                pass
        self.repair_attempts += 1
        text2 = self.chat(messages, max_tok=max_tok, json_mode=True)
        try:
            out = _extract_json(text2)
            if not isinstance(out, dict):
                raise ValueError(f"JSON 输出不是对象: {type(out)}")
            self.repair_successes += 1
            return out
        except (ValueError, KeyError):
            pass
        raise StructuredCallError([text, text2])

    def usage_report(self) -> str:
        return (f"fake 调用 {self.calls} 次 | prompt {self.prompt_tokens} | "
                f"completion {self.completion_tokens}")


# 各组件位置的合法默认输出（engine 的提案可被调用方覆盖以区分分支）
DEFAULT_OUTPUTS = {
    "atc": '{"mode": "influence", "reason": "fake"}',
    "trie": ('{"relevance": "high", "argument_strength": "medium", '
             '"cue_strength": "low", "interaction_pressure": "medium", '
             '"target_proposition": "accept the 85 price"}'),
    "jee": '{"stance_distance": "high", "relevant_current_state": ["B1"]}',
    "engine": ('{"bdi_updates": [{"operation": "update", "id": "B1", '
               '"new_strength": 2.4}], "new_items": [], '
               '"reaction_plan": "stay cautious"}'),
    "eue": ('{"appraisal": {"goal_congruence": -0.2, "coping_potential": 0.3, '
            '"future_expectancy": 0.1}, "desire_assessment": [], '
            '"emotion_proposal": {"category": "neutral"}}'),
    "nlg": "嗯，我再想想。",
    "ced": '{"done": false}',
}
