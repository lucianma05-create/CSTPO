"""RJ Cognitive Contract（文档 §11）：(Route, Judgment) -> 更新约束。

Contract 是给 LLM 的自然语言约束；LIMITS 是给确定性 Updater 的数值上限。
两类约束分离：LLM 提出更新，程序决定更新是否合法（文档原则 3）。
"""
from __future__ import annotations

# 数值上限：每种 (route, judgment) 下 B/D/I 允许的最大 |Δstrength|
LIMITS = {
    ("central", "accept"):       {"belief": 1.0, "desire": 0.5, "intention": 0.8},
    ("central", "noncommit"):    {"belief": 0.4, "desire": 0.2, "intention": 0.3},
    ("central", "reject"):       {"belief": 0.4, "desire": 0.2, "intention": 0.5},
    ("peripheral", "accept"):    {"belief": 0.3, "desire": 0.1, "intention": 0.6},  # 核心节点
    ("peripheral", "noncommit"): {"belief": 0.2, "desire": 0.0, "intention": 0.2},
    ("peripheral", "reject"):    {"belief": 0.2, "desire": 0.0, "intention": 0.5},
}

CONTRACTS = {
    ("central", "accept"): """The user carefully considers the substantive argument and accepts it.

Core beliefs directly addressed by the argument may change substantially.
Related desires may change moderately if the new belief changes their
importance, feasibility, or relevance.
Intentions may change when supported by the resulting beliefs and desires.

Do not modify unrelated beliefs, desires, or intentions.""",

    ("central", "noncommit"): """The user carefully considers the argument but remains undecided.

A directly relevant belief may loosen slightly, but should not reverse.
Core desires should remain stable.
Intentions may only change slightly.
Do not create a strong new commitment this turn.""",

    ("central", "reject"): """The user carefully evaluates the argument and rejects it.

Do not move the user's core belief toward the rejected claim.
Existing beliefs may remain stable or strengthen.
A counter-belief may be added only when supported by the user's reasoning.
A target intention may weaken.""",

    ("peripheral", "accept"): """The user's positive reaction is mainly driven by a peripheral cue
rather than deep evaluation of the core argument.

A cue-related belief such as trust, popularity, authority, familiarity,
or social norm may strengthen.
Core issue beliefs should remain mostly stable.
Core desires should not substantially change.
A short-term intention may increase moderately.""",

    ("peripheral", "noncommit"): """The user barely processes the reply through peripheral cues and stays uncommitted.

Beliefs may move only slightly.
Desires do not change.
Intentions may change only slightly.
Main changes should be in appraisal and emotion.""",

    ("peripheral", "reject"): """The user rejects the reply mainly via a peripheral cue.

Trust / cue beliefs may weaken.
Core issue beliefs and desires stay basically unchanged.
A target intention may weaken.
Frustration and arousal may rise.""",
}


def build_rj_contract(route: str, judgment: str) -> str:
    key = (route, judgment)
    if key not in CONTRACTS:
        raise ValueError(f"未知 (route, judgment) 组合: {key}")
    return CONTRACTS[key]


def limits_for(route: str, judgment: str) -> dict:
    return dict(LIMITS[(route, judgment)])
