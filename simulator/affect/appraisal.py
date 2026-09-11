"""Appraisal（文档 §15）：当轮中间变量 (GC, CP, FE)，统一归一化到 [-1,1]。

第一版由 LLM 提出三个分量，程序只做范围钳制与合法性检查；
文档 §15.1 的加权公式 GC = sum(s_i r_i gc_i) / sum(s_i r_i) 依赖
逐 Desire 的事件标注，留待后续版本。
"""
from __future__ import annotations

from simulator.state.schema import Appraisal


def normalize_appraisal(proposal: dict) -> tuple[Appraisal, list[str]]:
    notes: list[str] = []
    prop = proposal.get("appraisal", {}) or {}

    def get_field(name: str) -> float:
        v = prop.get(name, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            notes.append(f"appraisal.{name} 非法值 {v!r}，取 0.0")
            return 0.0
        if not -1.0 <= v <= 1.0:
            notes.append(f"appraisal.{name}={v} 越界，钳制到 [-1,1]")
            return max(-1.0, min(1.0, v))
        return v

    appraisal = Appraisal(
        goal_congruence=round(get_field("goal_congruence"), 3),
        coping_potential=round(get_field("coping_potential"), 3),
        future_expectancy=round(get_field("future_expectancy"), 3),
    )
    return appraisal, notes
