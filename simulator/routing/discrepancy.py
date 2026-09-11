"""Cognitive Discrepancy 估计（文档 §18、§19）：语义立场距离，不是文本相似度。

输入 target_proposition 来自 Route Feature Extractor（§14 输出），
LLM 输出 stance_distance 与 relevant_current_state（相关 B/D/I 节点 id）。
relevant_current_state 供 Deterministic Updater 做 Reject 方向检查（§31）
与相关节点定位；LLM 失败时退回 medium 距离 + 空节点列表，保证流程不中断。
"""
from __future__ import annotations

from simulator.profile.cognitive_profile import map_level

DISCREPANCY_PROMPT = """You are estimating the stance distance between an assistant's target
proposition and the user's current cognitive state.

This is NOT semantic text similarity.
Judge how far the target proposition is from what the user currently
believes, wants, or intends regarding the same issue.

Assistant's target proposition:
{target}

Current user state (beliefs / desires / intentions):
{state_text}

Definitions:

low:
The proposition is already broadly compatible with the user's current position.

medium:
The proposition differs meaningfully from the user's current position,
but does not directly overturn a strong existing belief or intention.

high:
The proposition directly conflicts with a strong existing belief,
goal priority, or intention.

Return exactly one JSON object:
{{
  "target_proposition": "the target proposition restated briefly",
  "relevant_current_state": ["B1", "D2", "I1"],
  "stance_distance": "low | medium | high",
  "reason": "one short sentence"
}}

"relevant_current_state" lists the ids of existing beliefs, desires, or
intentions most related to this proposition (may be [])."""


def estimate_discrepancy(llm, state, target: str | None) -> dict:
    """返回 {stance_distance, relevant_state_ids, target}。

    target 缺失（Route Feature Extractor 未提取到命题）时视为 medium 距离。
    """
    if not target:
        return {"stance_distance": 0.5, "relevant_state_ids": [], "target": None}
    try:
        out = llm.chat_json(
            [{"role": "user", "content": DISCREPANCY_PROMPT.format(
                target=target, state_text=state.summary_text())}],
            max_tok=300,
        )
    except Exception as e:
        print(f"  [discrepancy] LLM 失败，退回 medium 距离: {e}")
        return {"stance_distance": 0.5, "relevant_state_ids": [], "target": target}
    ids = out.get("relevant_current_state") or []
    if isinstance(ids, str):
        ids = [ids]
    # 只保留存在于当前状态的节点 id（LLM 可能幻觉出不存在/新编号的节点）
    valid = [i for i in (str(x).strip() for x in ids) if i and state.find(i) is not None]
    return {
        "stance_distance": map_level(out.get("stance_distance", "medium")),
        "relevant_state_ids": valid,
        "target": target,
    }
