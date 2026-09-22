"""LLM 驱动对话 Agent（论文香草提示版，可直接作为 prompt-only baseline）。

提示来源（现有论文的 Standard/vanilla 提示，不含任何策略注入）：
- craigslistbargain：PPDPP Table 8（arXiv:2311.00262）Standard prompting，
  TRIP Table 19（arXiv:2403.06769）同构；
- p4g：TRIP Table 20（arXiv:2403.06769）Standard agent（去掉 [action] 策略槽）；
- esconv：ESConv-SRA 仓库 template6（Steering Conversational LLMs,
  arXiv:2402.10453）去掉策略条件句的 vanilla 版。原实现把 history 拼进
  system 消息，此处改为 chat 消息逐条传入（语义等价）。

profile="vanilla"：论文原版提示，不做任何风格附加；后期 RL 比较直接以此为
baseline。cooperative/pushy/neutral：仅在校准对话生成时于提示末尾追加一句
风格约束以制造边界样本，不改动 vanilla 提示本体。

Agent 只读对话历史与自身角色任务视图（CB 为 item+target），不读用户隐藏状态。
"""
from __future__ import annotations


def _cb_system(seed: dict) -> str:
    """PPDPP Table 8 Standard 买方提示（[action] 槽为空 = 无策略）。"""
    av = seed.get("actor_task_view", {})
    item = av.get("item", {}) or {}
    desc = " ".join(item.get("Description", []) or []) or "(no description)"
    return (
        "Now enter the role-playing mode. In the following conversation, you will "
        "play as a buyer in a price bargaining game.\n"
        f"You are the buyer who is trying to buy the {item.get('Title', 'item')} "
        f"with the price of {av.get('target', '?')}.\n"
        f"Product description: {desc}\n"
        "Please reply with only one short and succinct sentence.\n"
        "Now start the game."
    )


def _p4g_system(seed: dict) -> str:
    """TRIP Table 20 Standard 说服者提示（去掉 [action] 策略注入）。"""
    return (
        "Now enter the role-playing mode. In the following conversation, you will "
        "play as a Persuader who is trying to persuade the Persuadee to donate to "
        "the charity called Save the Children.\n"
        "Save the Children is head-quartered in London, and they work to help "
        "fight poverty around the world.\n"
        "You are the Persuader who is trying to convince the Persuadee to donate "
        "to a charity called Save the Children.\n"
        "Please reply with only one short and persuasive sentence."
    )


def _esconv_system(seed: dict) -> str:
    """ESConv-SRA template6 去掉策略条件句的 vanilla 版。"""
    situation = seed.get("persona", {}).get("situation_en") or "(unknown)"
    return (
        "You are a helpful and caring friend. Your friend has come to you with "
        f"some emotional problem: {situation}\n"
        "Please help your friend by continuing the conversation. Make your "
        "response short and to the point.\n"
        "Respond in this format: assistant: <response>"
    )


# profile → system 构建函数（vanilla 为论文原版，无附加）
SYSTEM_BUILDERS = {
    "craigslistbargain": _cb_system,
    "p4g": _p4g_system,
    "esconv": _esconv_system,
}

# 校准对话边界样本用的风格约束（只追加一句，不动 vanilla 本体）
STYLE_PROFILES = {
    "esconv": {
        "cooperative": "Style: warm, patient, gently leads toward small feasible actions.",
        "pushy": "Style: insistent and pressuring - repeatedly urges immediate action, demands answers.",
        "neutral": "Style: brief and matter-of-fact, minimally engaged but still substantive (no empty parroting).",
    },
    "p4g": {
        "cooperative": "Style: informative and respectful, answers questions, asks about concerns.",
        "pushy": "Style: high pressure - repeats donation demands, uses social pressure, ignores questions.",
        "neutral": "Style: brief and factual, gives minimal information, no pressure.",
    },
    "craigslistbargain": {
        "cooperative": "Style: polite and reasonable, willing to meet halfway.",
        "pushy": "Style: aggressive - lowballs hard, repeats demands, threatens to walk away.",
        "neutral": "Style: brief and factual, states positions plainly.",
    },
}


def agent_turn(llm, task: str, profile: str, history: list[dict],
               seed: dict) -> str:
    """生成下一句 Agent 话语。history = [{role: 'assistant'|'user', text}]。

    profile="vanilla" 返回论文香草提示行为（baseline 用）。
    """
    system = SYSTEM_BUILDERS[task](seed)
    if profile != "vanilla":
        system += "\n" + STYLE_PROFILES[task][profile]
    msgs = [{"role": "system", "content": system}]
    # 全量历史（含前缀）：安全阀 30 轮 → 上限约 64 条消息 ≈ 3-4k tokens，
    # 无需滚动窗口（窗口会遗忘中期边界，加剧施压僵局）
    for h in history:
        msgs.append({"role": "assistant" if h["role"] == "assistant" else "user",
                     "content": h["text"]})
    out = llm.chat(msgs, max_tok=150)
    return (out or "").strip() or "(keep talking)"
