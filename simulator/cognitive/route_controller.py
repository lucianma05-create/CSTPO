"""Route 判断（文档 §9）：LLM 输出语义特征，程序结合 eta_R 产生最终 Route。

LLM 只做语义估计（issue_relevance / processing_demand / argument_content /
peripheral_cues / goal_relevance），不直接读抽象参数（文档 §4：数值保存在程序侧）。
"""
from __future__ import annotations

from simulator.profile.cognitive_profile import map_level, select_route

ROUTE_PROMPT = """You are analyzing how a user would process the assistant's latest reply.

Current user state (beliefs / desires / intentions):
{state_text}

Assistant's latest reply:
{reply}

Return exactly one JSON object:
{{
  "issue_relevance": "high|medium|low",
  "processing_demand": "high|medium|low",
  "argument_content": true,
  "peripheral_cues": ["authority"],
  "goal_relevance": "high|medium|low"
}}

- issue_relevance: how central the reply is to the user's current concerns.
- processing_demand: how hard the reply is to fully evaluate.
- argument_content: whether the reply contains substantive arguments or evidence.
- peripheral_cues: cues present in the reply (authority / social_proof / liking / confidence / familiarity / affective), or [].
- goal_relevance: how strongly the reply touches the user's current desires."""


def control_route(llm, profile, state, reply: str) -> dict:
    out = llm.chat_json(
        [{"role": "user", "content": ROUTE_PROMPT.format(state_text=state.summary_text(), reply=reply)}],
        max_tok=200,
    )
    rel = map_level(out.get("issue_relevance", "medium"))
    # 加工需求低 -> 能力高：ability = 1 - demand_level
    demand = map_level(out.get("processing_demand", "medium"))
    abi = round(1.0 - demand, 2)
    mot = map_level(out.get("goal_relevance", "medium"))
    route, p_C = select_route(profile.eta_R, rel, mot, abi)
    return {
        "route": route,
        "p_central": p_C,
        "issue_relevance": rel,
        "processing_demand": demand,
        "ability": abi,
        "motivation": mot,
        "argument_content": bool(out.get("argument_content", False)),
        "peripheral_cues": out.get("peripheral_cues", []) or [],
    }
