"""三任务策略标签映射（SFT_SPEC_20260915 权威实现，构建时断言用）。

- ESConv：官方 8 类；数据变体归一化（Questions→Question、Other→Others）。
- P4G：官方 27 类收敛为 13 类（用户认可方案）：
  捐赠请求 5 合一、inquiry 3 合一、inquiry 反应 3 合一、会话功能 6 合一、
  说服手段 8 保留、off_task/other → other。
- CB：intent 词表 → 草案 v2 的 5 类；offer/accept/quit/reject 为结构化
  任务事件（不作策略标签）；unknown/None 排除出 SFT（不当负例）。
- 角色识别：CB buyer 用 scenario.kbs[].personal.Role（与 seed_adapter 一致）。

任何改动先改 SFT_SPEC 再改本文件；SFT/RL/推理三处共用。
"""
from __future__ import annotations

# ---------- ESConv ----------
ESCONV_CANONICAL = [
    "Question", "Restatement or Paraphrasing", "Reflection of feelings",
    "Self-disclosure", "Affirmation and Reassurance",
    "Providing Suggestions", "Information", "Others",
]
ESCONV_VARIANTS = {"Questions": "Question", "Other": "Others"}


def esconv_label(raw: str) -> str | None:
    lab = ESCONV_VARIANTS.get(raw, raw)
    return lab if lab in ESCONV_CANONICAL else None


# ---------- P4G（27 → 13） ----------
P4G_MAP = {
    # 捐赠请求类（5 合一）
    "proposition-of-donation": "donate_request",
    "ask-donation-amount": "donate_request",
    "ask-donate-more": "donate_request",
    "ask-not-donate-reason": "donate_request",
    "confirm-donation": "donate_request",
    # 说服手段（8 保留）
    "logical-appeal": "logical_appeal",
    "emotion-appeal": "emotion_appeal",
    "credibility-appeal": "credibility_appeal",
    "donation-information": "donation_information",
    "personal-story": "personal_story",
    "self-modeling": "self_modeling",
    "foot-in-the-door": "foot_in_the_door",
    "praise-user": "praise_user",
    # 询问（3 合一）
    "task-related-inquiry": "inquiry",
    "personal-related-inquiry": "inquiry",
    "source-related-inquiry": "inquiry",
    # 询问反应（3 合一）
    "positive-to-inquiry": "inquiry_response",
    "neutral-to-inquiry": "inquiry_response",
    "negative-to-inquiry": "inquiry_response",
    # 会话功能（6 合一）
    "greeting": "social",
    "closing": "social",
    "thank": "social",
    "you-are-welcome": "social",
    "acknowledgement": "social",
    "comment-partner": "social",
    # 其他
    "off-task": "other",
    "other": "other",
}
P4G_CANONICAL = sorted(set(P4G_MAP.values()))


def p4g_label(raw: str) -> str | None:
    return P4G_MAP.get(raw)


# ---------- CB（intent → 草案 v2 5 类） ----------
CB_TASK_EVENTS = {"offer", "accept", "quit", "reject"}   # 结构化任务事件，不作策略标签
CB_EXCLUDE = {"unknown", None}                            # 排除出 SFT，不当负例
CB_MAP = {
    "init-price": "propose_price",
    "counter-price": "propose_price",
    "vague-price": "propose_price",
    "inquiry": "inquire",
    "inform": "inform",
    "agree": "stance",
    "disagree": "stance",
    "insist": "stance",
    "intro": "social",
}
CB_CANONICAL = ["propose_price", "inquire", "inform", "stance", "social", "other"]


def cb_label(raw) -> str | None:
    """返回策略标签；任务事件/排除项返回 None（调用方区分事件与排除）。"""
    if raw in CB_TASK_EVENTS or raw in CB_EXCLUDE:
        return None
    return CB_MAP.get(raw, "other")


def cb_buyer_idx(dialogue: dict) -> int | None:
    """CB 对话中 buyer 的 agent 索引（scenario.kbs[].personal.Role）。"""
    for i, kb in enumerate(dialogue.get("scenario", {}).get("kbs", []) or []):
        if kb.get("personal", {}).get("Role") == "buyer":
            return i
    return None
