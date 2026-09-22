"""LLM 先验：候选话语采样（temp 采样 4 条）+ 每轮用户情绪推断
（vendor env.py 的 Emotion 机制），拼成 (状态+候选) 文本喂 BERT。
"""
from __future__ import annotations

import re

from baseline.dialogxpert.llm_sampling import SamplingLLM

TOP_K = 4
CAND_TEMP = 0.7

_EMOTION_SYSTEM = (
    "You are an observer of a conversation between an assistant and a user. "
    "Infer the user's current emotional state from the conversation. Reply "
    "with one short emotion word or phrase only.")

_EMOTIONS = ["joyful", "hopeful", "calm", "neutral", "worried", "sad",
             "frustrated", "angry", "anxious", "relieved"]


class DialogXpertPrior:
    def __init__(self):
        self.sampler = SamplingLLM(temperature=CAND_TEMP)
        self.emotion_llm = SamplingLLM(temperature=0.0)

    def _history_msgs(self, system: str, history: list[dict]) -> list[dict]:
        msgs = [{"role": "system", "content": system}]
        for h in history:
            msgs.append({"role": "assistant" if h["role"] == "assistant"
                         else "user", "content": h["text"]})
        return msgs

    def propose(self, seed: dict, history: list[dict], k: int = TOP_K) -> list[str]:
        """采样 k 条候选话语（temperature 采样保证多样性）。"""
        from cstpo.core.agent import SYSTEM_BUILDERS
        from baseline.policies.policy import strip_role_prefix

        task = seed["task"]["task_id"]
        msgs = self._history_msgs(SYSTEM_BUILDERS[task](seed), history)
        out = []
        for _ in range(k):
            text = self.sampler.chat(msgs, max_tok=150)
            text = strip_role_prefix(text) or "(keep talking)"
            if text not in out:
                out.append(text)
        return out

    def infer_emotion(self, history: list[dict]) -> str:
        """推断用户当前情绪（vendor 的每轮 Emotion 先验）。"""
        msgs = [{"role": "system", "content": _EMOTION_SYSTEM}]
        for h in history[-8:]:  # 只看最近 4 回合
            msgs.append({"role": "assistant" if h["role"] == "assistant"
                         else "user", "content": h["text"]})
        raw = self.emotion_llm.chat(msgs, max_tok=20)
        low = raw.lower()
        for e in _EMOTIONS:
            if e in low:
                return e
        # 无匹配：直接截短（≤2 词），保留 LLM 输出本身作情绪标签
        return " ".join(raw.split()[:2]) or "neutral"

    @staticmethod
    def state_text(history: list[dict], candidate: str, emotion: str) -> str:
        """(状态+候选) 文本：vendor 的 full conversation + action 同构。"""
        conv = "\n".join(f"{t['role']}: {t['text']}" for t in history[-16:])
        return f"User emotion: {emotion}\nConversation:\n{conv}\nCandidate action:\n{candidate}"
