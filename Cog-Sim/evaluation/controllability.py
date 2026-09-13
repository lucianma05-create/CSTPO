"""Validation 4：Controllability。

固定 P/C_0/E_0/a_t，只改变 Θ_u=(η_R, τ_A, τ_R)，观察行为差异：
1. η_R ∈ {0.2, 0.5, 0.8} × {强论点/弱线索, 弱论点/强线索} → P(Central|η_R)、Route Separation；
2. τ 三档（Open/Moderate/Resistant，保证三 Judgment 分支可达）→ Judgment 分布与 Separation。
"""
from __future__ import annotations

from simulator.run_sim import SCENARIOS, build_state
from simulator.profile.cognitive_profile import build_habit_card
from simulator.simulator import UserSimulator
from simulator.llm import LLMClient

# 强论点/弱线索 与 弱论点/强线索 消息对（每任务一条，固定供对比）
MSG_TYPES = {
    "bargain": {
        "arg": "这车市场价95以上，85已经很便宜了。同款车在闲鱼成交记录都在90左右，昨天还有人出88我都没卖。",
        "cue": "别犹豫了，这车口碑特别好，我卖车这么多年回头客都来找我，大家一看就知道是好车。",
    },
    "donation": {
        "arg": "5元就能让孩子吃上一顿热饭，项目方公开每一笔采购清单和账目，钱花在哪都能查到。",
        "cue": "这个公益平台是知名大机构背书的，很多明星都在推，身边大家都说靠谱。",
    },
    "support": {
        "arg": "发一条消息只要两分钟，你上次和导师沟通还算顺利，先迈出最小的一步是最有效的做法。",
        "cue": "大家都说先动起来就好了，很多学长学姐都是这么过来的，你试试看。",
    },
}

ETA_VALUES = (0.2, 0.5, 0.8)

TAU_PROFILES = {
    "open": (0.25, 0.75),       # Accept: d=0.2；Noncommit: d=0.5；Reject: d=0.8
    "moderate": (0.35, 0.80),   # 同上，全部可达
    "resistant": (0.45, 0.80),  # 同上（tau_R<=0.8 保证 Reject 可达）
}

# τ 实验用三档消息：兼容（低 d）/ 中等（中 d）/ 冲突（高 d），覆盖三个离散 d 等级
TAU_MSGS = {
    "bargain": {
        "compatible": "行，就按80卖给你，你今天就能骑走。",
        "medium": "这车成色很新，85已经比市场价低了，你今晚就能骑走。",
        "conflicting": "85是最低价，我今天已经拒绝几个80的报价。",
    },
    "donation": {
        "compatible": "那就不劝你了，先不捐，等你看到具体项目的账目再决定。",
        "medium": "很多人都是从小额开始的，公益平台现在都能查到每一笔钱的去向。",
        "conflicting": "5元就能让一个孩子吃上一顿热饭，积少成多，效果是实打实的。",
    },
    "support": {
        "compatible": "那就先别发消息，等你准备好再说，这不是逃避。",
        "medium": "先给导师发一条消息只需要两分钟，回复一句进展顺利就够。",
        "conflicting": "其实先动起来就好，发一条消息成本很低，拖着只会更焦虑。",
    },
}


def _make_sim(task: str, llm: LLMClient | None = None,
              eta_R: float | None = None, tau: tuple | None = None) -> UserSimulator:
    state = build_state(SCENARIOS[task])
    if eta_R is not None:
        state.profile.eta_R = eta_R
    if tau is not None:
        state.profile.tau_A, state.profile.tau_R = tau
    state.habit_card = build_habit_card(state.profile.eta_R,
                                        state.profile.tau_A, state.profile.tau_R)
    return UserSimulator(state, llm or LLMClient())


def run_eta_experiment(task: str, msg_type: str, eta_R: float,
                       llm: LLMClient | None = None) -> dict:
    sim = _make_sim(task, llm, eta_R=eta_R)
    sim.simulate_turn(MSG_TYPES[task][msg_type])
    log = sim.logs[-1]
    return {"task": task, "msg_type": msg_type, "eta_R": eta_R,
            "mode": log.mode, "route": log.route, "p_central": log.p_central,
            "judgment": log.judgment, "discrepancy": log.discrepancy}


def run_tau_experiment(task: str, profile_name: str, msg_kind: str,
                       llm: LLMClient | None = None) -> dict:
    sim = _make_sim(task, llm, tau=TAU_PROFILES[profile_name])
    sim.simulate_turn(TAU_MSGS[task][msg_kind])
    log = sim.logs[-1]
    return {"task": task, "profile": profile_name, "msg_kind": msg_kind,
            "tau": TAU_PROFILES[profile_name],
            "mode": log.mode, "route": log.route, "judgment": log.judgment,
            "discrepancy": log.discrepancy}


def summarize_eta(results: list[dict]) -> dict:
    """P(Central | η_R) 按消息类型分组 + Route Separation。"""
    summary = {"by_msg_type": {}, "route_separation": None}
    for mt in ("arg", "cue"):
        by_eta = {}
        for r in results:
            if r["msg_type"] == mt:
                by_eta.setdefault(r["eta_R"], []).append(r["route"] == "central")
        p = {eta: round(sum(v) / len(v), 3) for eta, v in by_eta.items()}
        summary["by_msg_type"][mt] = p
        if p:
            summary[f"p_central_{mt}"] = p
    # Route Separation = P(Central|η=0.8) - P(Central|η=0.2)，按消息类型平均
    seps = []
    for mt in ("arg", "cue"):
        p = summary.get(f"p_central_{mt}", {})
        if 0.8 in p and 0.2 in p:
            seps.append(p[0.8] - p[0.2])
    if seps:
        summary["route_separation"] = round(sum(seps) / len(seps), 3)
    return summary


def summarize_tau(results: list[dict]) -> dict:
    """Judgment 分布按 (profile, msg_kind) + Separation（同消息下跨 profile 差异）。"""
    dist = {}
    for r in results:
        key = (r["profile"], r["msg_kind"])
        g = dist.setdefault(key, {"accept": 0, "noncommit": 0, "reject": 0,
                                  "n": 0, "d_vals": []})
        g[r["judgment"]] += 1
        g["n"] += 1
        g["d_vals"].append(r["discrepancy"])
    out = {}
    for (p, m), g in dist.items():
        out[f"{p}/{m}"] = {k: round(v / g["n"], 3) for k, v in g.items()
                           if k in ("accept", "noncommit", "reject")}
        out[f"{p}/{m}"]["d_mean"] = round(sum(g["d_vals"]) / len(g["d_vals"]), 3)
    # Separation：同一消息下 open 与 resistant 的 Accept/Reject 比例差（平均）
    for m in ("compatible", "medium", "conflicting"):
        o = out.get(f"open/{m}", {}), out.get(f"resistant/{m}", {})
    seps = []
    for m in ("compatible", "medium", "conflicting"):
        o, r = out.get(f"open/{m}", {}), out.get(f"resistant/{m}", {})
        if o and r:
            seps.append({"msg": m,
                         "accept_delta": round(r.get("accept", 0) - o.get("accept", 0), 3),
                         "reject_delta": round(o.get("reject", 0) - r.get("reject", 0), 3)})
    return {"judgment_distribution": out, "separation": seps}
