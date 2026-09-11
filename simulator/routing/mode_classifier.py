"""对话模式分类（文档 §6、§7）：Influence / Elicit / Social。

优先级规则：Influence > Elicit > Social，即只要有明显认知干预意图就判 Influence。
"""
from __future__ import annotations

MODE_PROMPT = """Classify the function of the assistant's latest reply.

Influence:
The reply introduces information, interpretation, advice, persuasion,
reframing, negotiation, recommendation, or a request intended to
change the user's belief, preference, goal priority, or intention.

Elicit:
The reply primarily asks the user to reveal an existing belief,
concern, desire, reason, preference, experience, or intention,
without proposing a new interpretation or direction.

Social:
The reply mainly serves casual conversation, rapport, greeting,
companionship, or conversational continuity.

Priority rule: Influence > Elicit > Social.
If a reply combines a question with a proposal or suggestion,
classify it as Influence.

Recent conversation:
{history}

Assistant's latest reply:
{reply}

Return exactly one JSON object: {{"mode": "Influence"}}, {{"mode": "Elicit"}}, or {{"mode": "Social"}}."""


def classify_mode(llm, history, reply: str) -> str:
    hist_text = "\n".join(f"{m['role']}: {m['text']}" for m in history[-6:]) or "(none)"
    msgs = [{
        "role": "user",
        "content": MODE_PROMPT.format(history=hist_text, reply=reply),
    }]
    try:
        out = llm.chat_json(msgs, max_tok=100)
        mode = str(out.get("mode", "")).strip().lower()
        if mode in {"influence", "elicit", "social"}:
            return mode
    except Exception as e:
        print(f"  [mode_classifier] LLM 失败，退回规则式分类: {e}")
    return _fallback(reply)


def _fallback(reply: str) -> str:
    """规则兜底：带问号且无明显主张 -> Elicit；否则按 Influence 处理。"""
    has_question = "?" in reply or "？" in reply
    return "elicit" if has_question else "influence"
