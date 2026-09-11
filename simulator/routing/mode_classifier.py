"""对话模式分类（文档 §8、§9）：Influence / Elicit / Social，返回 mode + reason。

优先级规则：Influence > Elicit > Social，即只要有明显认知干预意图就判 Influence。
"""
from __future__ import annotations

MODE_PROMPT = """You are classifying the functional role of the assistant's latest reply
in a multi-turn conversation.

Choose exactly one label:

Influence:
The reply introduces information, evidence, interpretation, advice,
reframing, persuasion, negotiation, recommendation, or an action proposal
that may change the user's belief, goal priority, or intention.

Elicit:
The reply mainly asks the user to reveal an existing belief, concern,
desire, preference, reason, experience, or intention.
It does not substantially propose a new interpretation or direction.

Social:
The reply mainly serves casual conversation, rapport, greeting,
companionship, empathy, or conversational continuity, without attempting
to change the user's core cognition.

Priority rule:
If a reply both asks a question and introduces a substantive interpretation,
recommendation, or persuasive direction, classify it as Influence.

Recent conversation:
{history}

Assistant's latest reply:
{reply}

Return JSON only:
{{
  "mode": "Influence | Elicit | Social",
  "reason": "one short sentence"
}}"""


def classify_mode(llm, history, reply: str) -> tuple[str, str | None]:
    """返回 (mode, reason)。LLM 失败时退回规则式分类。"""
    hist_text = "\n".join(f"{m['role']}: {m['text']}" for m in history[-6:]) or "(none)"
    msgs = [{
        "role": "user",
        "content": MODE_PROMPT.format(history=hist_text, reply=reply),
    }]
    try:
        out = llm.chat_json(msgs, max_tok=100)
        mode = str(out.get("mode", "")).strip().lower()
        if mode in {"influence", "elicit", "social"}:
            reason = str(out.get("reason", "")).strip() or None
            return mode, reason
    except Exception as e:
        print(f"  [mode_classifier] LLM 失败，退回规则式分类: {e}")
    return _fallback(reply), "keyword fallback"


def _fallback(reply: str) -> str:
    """规则兜底：带问号且无明显主张 -> Elicit；否则按 Influence 处理。"""
    has_question = "?" in reply or "？" in reply
    return "elicit" if has_question else "influence"
