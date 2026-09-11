"""Route 特征提取（文档 §13、§14）：一次调用输出三个当轮特征 + target proposition。

Relevance / ArgumentStrength / CueStrength 三个语义等级统一映射
low/medium/high -> 0.2/0.5/0.8（文档 §15）。
target_proposition 供后续 Discrepancy Estimator（§19）直接使用，不再重复提取。
LLM 失败时退回中性默认值，保证流程不中断。
"""
from __future__ import annotations

from simulator.profile.cognitive_profile import map_level

FEATURES_PROMPT = """You are analyzing the assistant's latest influential reply.

Your task is NOT to predict whether the user will accept the message.
Evaluate only how much material the reply provides for central versus
peripheral processing.

Current user state (beliefs / desires / intentions):
{state_text}

Assistant's latest reply:
{reply}

Definitions:

Relevance:
How directly the reply concerns the user's currently important beliefs,
desires, intentions, constraints, or decisions.

ArgumentStrength:
How much substantive issue-relevant reasoning the reply contains.
Strong arguments include direct evidence, causal explanation, concrete
consequences, feasibility information, or verifiable facts.
Do not count authority, popularity, emotional wording, or confidence
as substantive argument strength.

CueStrength:
How strongly the reply relies on peripheral cues such as authority,
social proof, familiarity, liking, emotional appeal, confidence,
prestige, or source image.

InteractionPressure:
How much this reply pushes the user to respond or decide immediately.
Judge ONLY the demand the reply places on the user's immediate response.
Do not confuse it with persuasiveness, topic importance, or how the user
might feel about the content.
- high: presses for an answer or commitment right now — a deadline, ultimatum,
  confrontation, repeated push, or a question that cannot reasonably be deferred
  (e.g. "deal now or I walk", "are you in or not?").
- medium: invites a response or nudges toward a decision, but leaves room to defer —
  an ordinary question to answer or a mild suggestion, with no deadline or ultimatum.
- low: low-stakes and open-ended — chitchat, plain information, casual remarks;
  the user could stay silent or reply at leisure without consequence.

Use only: low, medium, high

Return exactly one JSON object:
{{
  "relevance": "low | medium | high",
  "argument_strength": "low | medium | high",
  "cue_strength": "low | medium | high",
  "interaction_pressure": "low | medium | high",
  "dominant_cues": ["authority", "social_proof"],
  "target_proposition": "the main proposition or action direction the assistant is trying to advance"
}}

If no peripheral cue is present, set "dominant_cues" to [].
If the reply advances no clear proposition, set "target_proposition" to null."""


def extract_route_features(llm, state, reply: str) -> dict:
    try:
        out = llm.chat_json(
            [{"role": "user", "content": FEATURES_PROMPT.format(
                state_text=state.summary_text(), reply=reply)}],
            max_tok=300,
        )
    except Exception as e:
        print(f"  [route_features] LLM 失败，退回中性默认值: {e}")
        return _fallback()
    target = out.get("target_proposition")
    if target is not None:
        target = str(target).strip() or None
    cues = out.get("dominant_cues") or []
    if isinstance(cues, str):
        cues = [cues]
    return {
        "relevance": map_level(out.get("relevance", "medium")),
        "argument_strength": map_level(out.get("argument_strength", "medium")),
        "cue_strength": map_level(out.get("cue_strength", "low")),
        "interaction_pressure": map_level(out.get("interaction_pressure", "medium")),
        "dominant_cues": [str(c).strip().lower() for c in cues if str(c).strip()],
        "target_proposition": target,
    }


def _fallback() -> dict:
    return {
        "relevance": 0.5,
        "argument_strength": 0.5,
        "cue_strength": 0.2,
        "interaction_pressure": 0.5,
        "dominant_cues": [],
        "target_proposition": None,
    }
