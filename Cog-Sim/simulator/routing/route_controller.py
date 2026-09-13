"""Route 决策（文档 §16、§17、§17.1、§17.2）：程序侧公式 + 采样。

CentralScore_t   = eta_R * Rel_t * Arg_t
PeripheralScore_t = (1 - eta_R) * Cue_t
p_C = CentralScore / (CentralScore + PeripheralScore + eps)

弱信号回退（§17.2）：CentralScore + PeripheralScore < delta 时，
该 Influence 回复既无 substantive argument 也无明显 cue，p_C 直接退回 eta_R。

两种采样模式：
- bernoulli（§17）：R_t ~ Bernoulli(p_C)，训练 rollout 保留随机性；
- deterministic（§17.1）：p_C >= 0.5 取 Central，正式 benchmark 可复现。
"""
from __future__ import annotations

import random

EPS = 1e-8
WEAK_SIGNAL_DELTA = 0.05   # 文档 §17.2

ROUTE_MODES = {"bernoulli", "deterministic"}


def compute_p_central(eta_R: float, rel: float, arg: float, cue: float) -> float:
    """文档 §16、§17 的归一化概率公式，含 §17.2 弱信号回退。"""
    central = eta_R * rel * arg
    peripheral = (1.0 - eta_R) * cue
    if central + peripheral < WEAK_SIGNAL_DELTA:
        return eta_R
    return central / (central + peripheral + EPS)


def decide_route(
    eta_R: float,
    rel: float,
    arg: float,
    cue: float,
    mode: str = "deterministic",
    rng: random.Random | None = None,
) -> tuple[str, float]:
    """返回 (route, p_central)。mode 取值见 ROUTE_MODES。"""
    if mode not in ROUTE_MODES:
        raise ValueError(f"未知 route 决策模式 {mode!r}，可选 {sorted(ROUTE_MODES)}")
    p_C = min(max(compute_p_central(eta_R, rel, arg, cue), 0.0), 1.0)
    if mode == "bernoulli":
        route = "central" if (rng or random).random() < p_C else "peripheral"
    else:
        route = "central" if p_C >= 0.5 else "peripheral"   # 文档 §17.1
    return route, round(p_C, 4)
