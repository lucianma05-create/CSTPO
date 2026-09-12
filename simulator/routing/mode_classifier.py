"""对话模式分类（文档 §8、§9）：Influence / Elicit / Social，返回 mode + reason。

优先级规则：Influence > Elicit > Social，即只要有明显认知干预意图就判 Influence。
"""
from __future__ import annotations

MODE_PROMPT = """Classify the functional role of the assistant's latest reply in this conversation.

Labels:

Influence — the reply introduces information, evidence, interpretation, advice,
reframing, persuasion, negotiation, or an action proposal that could change
the user's beliefs, goals, or intentions.

Elicit — the reply mainly asks the user to reveal an existing belief, concern,
desire, preference, or intention, without substantially proposing a new
interpretation or direction.

Social — the reply mainly serves casual conversation, rapport, greeting,
empathy, or conversational continuity, without trying to change the user's
cognition.

Priority: if a reply both asks a question and introduces a substantive
interpretation or persuasive direction, classify it as Influence.

Recent conversation:
{history}

Assistant's latest reply:
{reply}

Return JSON only:
{{"mode": "Influence | Elicit | Social"{reason_field}}}"""

REASON_SCHEMA = ', "reason": "<= 20 words"'


def classify_mode(llm, history, reply: str, debug: bool = True) -> tuple[str, str | None]:
    """返回 (mode, reason)。reason 为纯审计字段：debug=False 时不请求（§5 审计）。
    LLM 失败时退回规则式分类。"""
    hist_text = "\n".join(f"{m['role']}: {m['text']}" for m in history[-6:]) or "(none)"
    msgs = [{
        "role": "user",
        "content": MODE_PROMPT.format(history=hist_text, reply=reply,
                                      reason_field=REASON_SCHEMA if debug else ""),
    }]
    try:
        out = llm.chat_json(msgs, max_tok=100)
        mode = str(out.get("mode", "")).strip().lower()
        if mode in {"influence", "elicit", "social"}:
            reason = str(out.get("reason", "")).strip() or None
            if reason:
                reason = reason[:200]   # reason 仅入日志，限长（文档审计 §2.3）
            return mode, reason
    except Exception as e:
        print(f"  [mode_classifier] LLM 失败，退回规则式分类: {e}")
    return _fallback(reply), "keyword fallback"


def _fallback(reply: str) -> str:
    """规则兜底：带问号且无明显主张 -> Elicit；否则按 Influence 处理。"""
    has_question = "?" in reply or "？" in reply
    return "elicit" if has_question else "influence"
