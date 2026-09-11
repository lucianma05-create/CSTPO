"""Emotion 更新（文档 §16、§17）：Appraisal 决定目标情绪 + 惯性平滑。

v_hat = w1*GC + w2*CP + w3*FE，第一版 w1=w2=w3（文档 §16）。
v_{t+1} = (1-lambda_v) * v_t + lambda_v * v_hat_{t+1}

Arousal 的目标值公式文档未给出（依赖 EventIntensity、|ΔBDI|、uncertainty 等），
第一版由 LLM 的 emotion_proposal 提供 r_hat，程序做惯性平滑与范围钳制。
Influence / Elicit 用 lambda=0.4；Social 只允许低幅度变化，用 lambda=0.25。
category 不参与动力学，主要用于展示与生成（文档 §3.3）。
"""
from __future__ import annotations

from simulator.state.schema import EMOTION_CATEGORIES, Appraisal, Emotion

LAMBDA_INFLUENCE = 0.4
LAMBDA_SOCIAL = 0.25


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(float(v), lo), hi)


def update_emotion(
    prev: Emotion, proposal: dict, appraisal: Appraisal, mode: str
) -> tuple[Emotion, list[str]]:
    notes: list[str] = []
    lam = LAMBDA_SOCIAL if mode == "social" else LAMBDA_INFLUENCE
    prop = proposal.get("emotion_proposal", {}) or {}

    # 目标 valence 由 Appraisal 公式决定（文档 §16，w1=w2=w3）
    v_hat = _clamp(
        (appraisal.goal_congruence + appraisal.coping_potential + appraisal.future_expectancy) / 3.0,
        -1.0, 1.0,
    )
    # 目标 arousal 由 LLM 提议（公式未定义，文档 §16 留白）
    r_hat = _clamp(prop.get("arousal", prev.arousal), 0.0, 1.0)
    category = str(prop.get("category", prev.category)).strip().lower()
    if category not in EMOTION_CATEGORIES:
        notes.append(f"非法 emotion category {category!r}，退回 neutral")
        category = "neutral"

    valence = round((1 - lam) * prev.valence + lam * v_hat, 3)
    arousal = round((1 - lam) * prev.arousal + lam * r_hat, 3)
    next_emo = Emotion(valence=valence, arousal=arousal, category=category)
    return next_emo, notes
