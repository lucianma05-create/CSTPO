"""对话结束信号（任务中立）：用户是否想结束本次对话。

注意 user_done != task_done（见文档 §25 的 Simulator/TaskEvaluator 分离）：
- 用户说"那就 85 成交"是任务完成，不代表对话结束；
- 用户说"我不想聊了"是对话结束，不代表任务成功。
本模块只判断前者口径的"对话闭合行为"（farewell / 离开 / 拒绝继续交谈）。
"""
from __future__ import annotations

DONE_PROMPT = """You are analyzing the user's latest message in a multi-turn conversation.

Question: does the user intend to END the current conversation now?

"Ending the conversation" includes: saying goodbye or farewell, leaving,
refusing to continue talking, or otherwise closing the exchange.

It does NOT mean: agreeing to a proposal, making a decision, accepting a deal,
or answering a question. Task outcomes are irrelevant here.

Recent conversation:
{history}

User's latest message:
{utterance}

Return exactly one JSON object: {{"done": true|false, "reason": "brief"}}"""

FAREWELL_PATTERNS = [
    "再见", "拜拜", "回头再", "下次再聊", "下次聊", "回聊", "先这样",
    "不聊了", "不想说了", "不想聊了", "就这样吧", "挂了吧", "先挂了",
    "我要下了", "我先下了", "先忙了", "改天再", "到此为止", "就这样",
    "goodbye", "bye", "farewell", "talk later", "i have to go",
]


def classify_user_done(llm, state, utterance: str) -> tuple[bool, str | None]:
    """返回 (user_done, reason)。LLM 失败时退回规则式关键词匹配。"""
    if not utterance.strip():
        return False, "empty utterance"
    hist_text = "\n".join(f"{m['role']}: {m['text']}" for m in state.history[-8:])
    try:
        out = llm.chat_json(
            [{"role": "user", "content": DONE_PROMPT.format(history=hist_text, utterance=utterance)}],
            max_tok=100,
        )
        return bool(out.get("done", False)), str(out.get("reason", "")) or None
    except Exception as e:
        print(f"  [user_done] LLM 失败，退回规则匹配: {e}")
    low = utterance.lower()
    return any(p in low for p in FAREWELL_PATTERNS), "keyword fallback"
