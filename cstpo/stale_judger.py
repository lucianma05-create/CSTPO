"""对话冗余判定（第三方，实验层；不改模拟器源码）。

15 轮后每 2 轮观察一次：若对话已冗余（用户重复同一抵抗 / 空转寒暄、
无新进展），判定真实用户会想自然结束，并生成一句用户口吻的结束语
（第三方打断）。与 CED 的差异：CED 只判用户最新话语是否表达结束意图；
本判定在对话级判断「是否该结束了」，且只在长对话（≥15 轮）启动。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient

STALE_CHECK_START = 15  # 第 15 轮起
STALE_CHECK_EVERY = 2   # 每 2 轮观察一次

SYSTEM = """You are an outside observer of a conversation between an agent
(assistant) and a user. Judge whether the conversation has become redundant:
the user keeps repeating the same resistance or position, or the exchanges are
empty small talk with no new progress, such that a real user would naturally
want to end the conversation.

Do NOT end merely because the conversation is long - a long conversation with
genuine new content should continue. Do NOT end because a negotiation is
ongoing with real concession movement.

Return exactly one JSON object: {"stale": true, "final_line": "..."} where
final_line (only when stale=true) is the one-sentence closing utterance the
USER would naturally say to end the conversation, in the user's own voice
(e.g., "I think I've said all I can for now. Thanks, bye.")."""


def should_check(n_assistant_turns: int) -> bool:
    """第 15 轮起每 2 轮观察一次。"""
    return (n_assistant_turns >= STALE_CHECK_START
            and (n_assistant_turns - STALE_CHECK_START) % STALE_CHECK_EVERY == 0)


def check_stale(llm: LLMClient, turns: list[dict], task: str) -> dict:
    tail = "\n".join(f"{t['role']}: {t['text']}" for t in turns[-12:])
    out = llm.chat_json(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Task type: {task}\n\nRecent conversation:\n{tail}"}],
        max_tok=200)
    return {"stale": bool(out.get("stale", False)),
            "final_line": (out.get("final_line") or "").strip()}
