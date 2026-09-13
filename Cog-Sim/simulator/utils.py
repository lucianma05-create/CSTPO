"""共享小工具（数值钳制等），供 state/affect 等模块复用。"""
from __future__ import annotations


def clamp(v: float, lo: float, hi: float) -> float:
    """把 v 钳制到 [lo, hi]。"""
    return min(max(float(v), lo), hi)
