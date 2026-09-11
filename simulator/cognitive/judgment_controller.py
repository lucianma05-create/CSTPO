"""Judgment 判断（文档 §10）：语义估计 + 用户阈值。

LLM 提取 agent 真正想改变的 proposition 并估计与当前 BDI 的距离，
程序侧用 tau_A / tau_R 阈值把距离映射成 Accept / Noncommit / Reject。
最终 Judgment 不由 LLM 任意决定。
"""
from __future__ import annotations

from simulator.profile.cognitive_profile import apply_judgment_threshold, map_level

JUDGE_PROMPT = """The assistant's latest reply:
{reply}

Current user state (beliefs / desires / intentions):
{state_text}

Step 1: Extract the proposition the assistant is actually trying to make the user accept (one short sentence).
Step 2: Judge the distance between that proposition and the user's current beliefs.

Return exactly one JSON object:
{{
  "target": "Accepting 85 yuan is reasonable",
  "stance_distance": "low|medium|high",
  "support_quality": "low|moderate|high",
  "related_belief_ids": ["B1"]
}}

- stance_distance: how far the proposition is from the user's current relevant beliefs.
- support_quality: how well the reply supports the proposition.
- related_belief_ids: ids of existing beliefs most related to this proposition (may be [])."""


def control_judgment(llm, profile, state, reply: str) -> dict:
    out = llm.chat_json(
        [{"role": "user", "content": JUDGE_PROMPT.format(state_text=state.summary_text(), reply=reply)}],
        max_tok=300,
    )
    # 文档 §10 映射：low->0.2, medium->0.5, high->0.8
    d = map_level(out.get("stance_distance", "medium"))
    judgment = apply_judgment_threshold(d, profile.tau_A, profile.tau_R)
    return {
        "target": str(out.get("target", "")).strip() or None,
        "discrepancy": d,
        "support_quality": str(out.get("support_quality", "moderate")).strip().lower(),
        "judgment": judgment,
        "related_belief_ids": [str(i) for i in out.get("related_belief_ids", []) or []],
    }
