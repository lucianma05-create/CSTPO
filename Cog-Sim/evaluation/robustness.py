"""Validation 辅助：robustness——LLM 失败兜底与 JSON 容错的单元级检查（不调用 API）。"""
from __future__ import annotations

from simulator.llm import _extract_json
from simulator.routing.mode_classifier import _fallback as mode_fallback
from simulator.routing.route_features import _fallback as feats_fallback
from simulator.generation.conversation_end import classify_user_done, FAREWELL_PATTERNS


class _FailingLLM:
    """任何调用都抛异常的假 LLM。"""
    def chat(self, *a, **k):
        raise RuntimeError("simulated API failure")
    def chat_json(self, *a, **k):
        raise RuntimeError("simulated API failure")


def run() -> dict:
    checks = []
    # 1) mode fallback：带问号 -> elicit；无问号 -> influence
    checks.append(("mode fallback 问号", mode_fallback("你的预算是多少？") == "elicit"))
    checks.append(("mode fallback 主张", mode_fallback("85已经很便宜了") == "influence"))
    # 2) features fallback 默认值
    f = feats_fallback()
    checks.append(("features fallback", f["relevance"] == 0.5 and f["cue_strength"] == 0.2
                   and f["target_proposition"] is None))
    # 3) CED 关键词兜底（中文 + 英文）
    class _S:
        history = []
    ok_cn = classify_user_done(_FailingLLM(), _S(), "行，那今天先聊到这，再见。")[0]
    ok_en = classify_user_done(_FailingLLM(), _S(), "ok bye, talk later")[0]
    ok_no = classify_user_done(_FailingLLM(), _S(), "那再便宜点我就买了")[0]
    checks.append(("CED fallback 中文再见", ok_cn is True))
    checks.append(("CED fallback 英文再见", ok_en is True))
    checks.append(("CED fallback 非结束", ok_no is False))
    # 4) JSON 容错：markdown fence / 前后缀 / 非法
    checks.append(("json fence", _extract_json('```json\n{"a": 1}\n```') == {"a": 1}))
    checks.append(("json 前后缀", _extract_json('ok, here: {"a": 2} end') == {"a": 2}))
    try:
        _extract_json("no json here")
        checks.append(("json 非法抛错", False))
    except ValueError:
        checks.append(("json 非法抛错", True))
    # 5) 关键词表覆盖基本场景
    checks.append(("farewell 词表非空", len(FAREWELL_PATTERNS) > 0))
    passed = sum(1 for _, ok in checks if ok)
    return {"checks": [{"name": n, "ok": ok} for n, ok in checks],
            "passed": passed, "total": len(checks)}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), ensure_ascii=False, indent=1))
