"""Appraisal（文档 §32）：当轮中间变量 (GC, CP, FE)，统一归一化到 [-1,1]。

GC 半程序化（§32.1）：LLM 只输出逐 Desire 的语义标注 (relevance, gc_i)，
程序按公式 GC = Σ s_i·rel_i·gc_i / Σ s_i·rel_i 重算，权重取**更新后 BDI** 中
active Desire 的强度 s_i（§4 ActiveGoals = TopK(s_i·rel_i)，第一版 K=2）。
LLM 标注缺失/非法时退回 LLM 提议的 GC 值。
CP/FE 仍由 LLM 提议 + 程序钳制（文档未给公式）。
"""
from __future__ import annotations

from simulator.state.schema import Appraisal, UserState
from simulator.utils import clamp

ACTIVE_GOALS_K = 2   # 文档 §4：ActiveGoals = TopK(s_i * rel_i)，第一版 K=2


def compute_gc(state: UserState, assessments: list | None) -> tuple[float | None, list[str]]:
    """按 §32.1 公式用 (s_i, rel_i, gc_i) 重算 GC，返回 (gc, notes)。

    无有效标注时返回 (None, notes)，调用方退回 LLM 提议值。
    """
    notes: list[str] = []
    scored: list[tuple[str, float, float, float]] = []   # (id, s_i, rel_i, gc_i)
    for a in assessments or []:
        if not isinstance(a, dict):
            continue
        item = state.find(str(a.get("id", "")))
        if item is None or item.type != "desire" or not item.active:
            notes.append(f"desire_assessment 引用无效节点 {a.get('id')!r}，忽略")
            continue
        try:
            rel = clamp(float(a.get("relevance", 0.0)), 0.0, 1.0)
            gc = clamp(float(a.get("gc", 0.0)), -1.0, 1.0)
        except (TypeError, ValueError):
            notes.append(f"desire_assessment[{item.id}] 数值非法，忽略")
            continue
        scored.append((item.id, item.strength, rel, gc))
    if not scored:
        return None, notes
    # ActiveGoals = TopK(s_i * rel_i)，文档 §4 第一版 K=2
    top = sorted(scored, key=lambda x: x[1] * x[2], reverse=True)[:ACTIVE_GOALS_K]
    num = sum(s * rel * gc for _, s, rel, gc in top)
    den = sum(s * rel for _, s, rel, gc in top)
    if den <= 1e-9:
        notes.append("desire_assessment 有效权重为 0，GC 退回 LLM 提议值")
        return None, notes
    gc = clamp(num / den, -1.0, 1.0)
    notes.append(
        f"GC 按 §32.1 公式重算: Σ(s·rel·gc)/Σ(s·rel) = {gc:.3f} "
        f"(top{len(top)}: {[i for i, _, _, _ in top]})"
    )
    return gc, notes


def normalize_appraisal(proposal: dict, state: UserState | None = None) -> tuple[Appraisal, list[str]]:
    """钳制 LLM 提议值；state 传入时用 desire_assessment 按公式重算 GC。

    state 应为**更新后**的 UserState（Influence 在 Updater 之后调用），
    使 GC 权重反映实际生效的 Desire 强度（修复提案被截断时的时点错位）。
    """
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
            return clamp(v, -1.0, 1.0)
        return v

    gc = get_field("goal_congruence")
    if state is not None:
        gc_calc, g_notes = compute_gc(state, proposal.get("desire_assessment"))
        notes += g_notes
        if gc_calc is not None:
            if abs(gc_calc - gc) > 1e-6:
                notes.append(f"GC 提议值 {gc:.2f} -> 公式值 {gc_calc:.3f}")
            gc = gc_calc

    appraisal = Appraisal(
        goal_congruence=round(gc, 3),
        coping_potential=round(get_field("coping_potential"), 3),
        future_expectancy=round(get_field("future_expectancy"), 3),
    )
    return appraisal, notes
