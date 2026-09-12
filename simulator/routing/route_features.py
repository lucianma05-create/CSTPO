"""Route 特征提取（文档 §13、§14）：一次调用输出三个当轮特征 + target proposition。

Relevance / ArgumentStrength / CueStrength 三个语义等级统一映射
low/medium/high -> 0.2/0.5/0.8（文档 §15）。
target_proposition 供后续 Discrepancy Estimator（§19）直接使用，不再重复提取。
LLM 失败时退回中性默认值，保证流程不中断。
"""
from __future__ import annotations

from simulator.profile.cognitive_profile import map_level

# 交互压力操作定义（TRIE 为规范所有者；SRR/ERR 合并调用复用，见 user_response.py）
INTERACTION_PRESSURE_DEF = """how much the reply demands the user's immediate response.
Judge ONLY the demand on immediate response — not persuasiveness, topic
importance, or how the user might feel. high = presses for an answer or
commitment right now (deadline, ultimatum, confrontation, repeated push, a
question that cannot be deferred). medium = an ordinary question or mild
suggestion with room to defer (no deadline). low = chitchat, plain
information, casual remarks (the user could stay silent without consequence)."""

FEATURES_PROMPT = """You are extracting semantic features from the assistant's latest reply.
You do NOT predict whether the user will accept it — only what processing
material the reply provides.

Current user state (beliefs / desires / intentions):
{state_text}

Assistant's latest reply:
{reply}

Operational definitions:

Relevance — how directly the reply concerns the user's currently important
beliefs, desires, intentions, constraints, or decisions.

ArgumentStrength — how much substantive issue-relevant reasoning the reply
contains (direct evidence, causal explanation, concrete consequences,
feasibility, verifiable facts). Authority, popularity, emotional wording,
and confidence do NOT count as argument strength.

CueStrength — how strongly the reply relies on peripheral cues: authority,
social proof, familiarity, liking, emotional appeal, confidence, prestige.

InteractionPressure — {pressure_def}

Use only: low, medium, high

Return exactly one JSON object:
{{
  "relevance": "low | medium | high",
  "argument_strength": "low | medium | high",
  "cue_strength": "low | medium | high",
  "interaction_pressure": "low | medium | high",
  {cues_field}
  "target_proposition": "the main proposition or action direction the assistant is trying to advance"
}}

If the reply advances no clear proposition, set "target_proposition" to null."""

CUES_SCHEMA = '"dominant_cues": ["authority", "social_proof"],\n  '


def extract_route_features(llm, state, reply: str, debug: bool = True) -> dict:
    """dominant_cues 为纯审计字段：debug=False 时不请求（§5 审计，不影响
    Route 计算——Route 只依赖三个等级特征）。"""
    try:
        out = llm.chat_json(
            [{"role": "user", "content": FEATURES_PROMPT.format(
                state_text=state.summary_text(), reply=reply,
                pressure_def=INTERACTION_PRESSURE_DEF,
                cues_field=CUES_SCHEMA if debug else "")}],
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
