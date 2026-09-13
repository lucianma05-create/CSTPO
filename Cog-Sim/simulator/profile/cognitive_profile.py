"""认知习惯参数：等级映射、Judgment 阈值、习惯卡片（文档 §6、§7、§15）。

Route 概率公式已移至 routing/route_controller.py（改进文档 §16、§17）。
"""
from __future__ import annotations

LEVEL_MAP = {"low": 0.2, "medium": 0.5, "high": 0.8}


def map_level(v, default: float = 0.5) -> float:
    return LEVEL_MAP.get(str(v).strip().lower(), default)


def apply_judgment_threshold(discrepancy, tau_A, tau_R):
    """文档 §6.2、§20: d<=tau_A -> Accept; d>=tau_R -> Reject; 否则 Noncommit。"""
    if discrepancy <= tau_A:
        return "accept"
    if discrepancy >= tau_R:
        return "reject"
    return "noncommit"


def build_habit_card(eta_R, tau_A, tau_R) -> str:
    """把数值参数编译成一次性自然语言认知习惯描述（文档 §7），整个 rollout 固定。"""
    parts = []
    if eta_R >= 0.65:
        parts.append(
            "You usually examine important claims carefully. You pay attention to "
            "whether evidence directly supports a claim. Popularity, authority, "
            "emotional language, or confidence alone rarely changes your core beliefs."
        )
    elif eta_R <= 0.35:
        parts.append(
            "You usually do not deeply analyze every argument. You are relatively "
            "sensitive to confidence, source credibility, social consensus, "
            "familiarity, emotional tone, and other simple cues."
        )
    else:
        parts.append(
            "You sometimes weigh evidence carefully and sometimes rely on simple cues, "
            "depending on how relevant and demanding the issue feels."
        )
    width = tau_R - tau_A
    if width <= 0.35:
        parts.append(
            "You are cautious about changing established views. A new claim must fit "
            "reasonably well with what you already believe, or provide sufficiently "
            "convincing reasons, before you accept it."
        )
    elif width >= 0.60:
        parts.append(
            "You are relatively open to new interpretations and proposals. You can "
            "provisionally incorporate a new view even when it differs somewhat from "
            "your current position."
        )
    else:
        parts.append(
            "You are moderately open to revising your views when a claim is well supported."
        )
    return " ".join(parts)
