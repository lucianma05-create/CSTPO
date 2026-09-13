"""Judgment 判定（文档 §20）：语义距离 + 用户阈值，程序侧纯阈值映射。

d_t <= tau_A -> Accept；d_t >= tau_R -> Reject；否则 Noncommit。
最终 Judgment 不是 LLM 任意决定的，而是 semantic estimation + user threshold
共同决定（文档 §19、§20）。
"""
from __future__ import annotations

from simulator.profile.cognitive_profile import apply_judgment_threshold


def control_judgment(profile, discrepancy: float) -> str:
    return apply_judgment_threshold(discrepancy, profile.tau_A, profile.tau_R)
