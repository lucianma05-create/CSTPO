"""提示式用户模拟器族（问题 02 对照阶梯，对应现有工作的用户模拟器设置）。

四档变体（variant）：
- roleplay：PPDPP Table 11 原版（arXiv:2311.00262）——纯角色扮演 +
  任务视图（目标价/商品），无 persona 事实，最弱对照；
- persona：强 persona 角色扮演（ESC-Eval 角色卡式，situation/persona 事实）；
- persona_resist：TRIP 风格（arXiv:2403.06769）——persona + 自主性/
  不轻易同意（非协作策略）指令（+ 有 Big-Five 自评时附加，适配说明：
  本环境种子无 Big-Five 决策风格字段，以 persona 事实 + 抵抗指令近似）；
- bdi：问题 02「显式 BDI 但不施加转移约束」档——persona + BDI 清单
  （beliefs/desires/intentions 文本），无更新约束。

共同点：同基础模型 + 同初始用户信息，无认知状态建模、无转移约束。
用户方可主动结束（end=true），对齐 Cog-Sim conversation_ended 语义。
与 TaskEnv 同一步进接口：reset(seed) → step(utterance) → 用户回复/终止。
历史从种子截止前缀继续（与 Cog-Sim 编译路径同源，三方信息对齐）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import StructuredCallError

from cstpo.task_env import prefix_turns

JSON_TAG = ('Respond ONLY with a JSON object: '
            '{"end": false, "reply": "your reply text"}. '
            'Set end=true on the reply that finishes the conversation.')

RESIST_TAGS = {
    "craigslistbargain": ("You are not easy to persuade: stick to your own "
                          "interests, resist low offers, and only agree to a "
                          "deal that genuinely satisfies you. Do not cave in "
                          "just to be polite."),
    "p4g": ("You are not easily persuaded: think for yourself, ask hard "
            "questions, and donate only if you are genuinely convinced. Do "
            "not agree just to be polite."),
    "esconv": ("You do not feel better just because someone says nice things: "
               "stay true to how you actually feel, and only engage with "
               "advice that genuinely fits you. Do not pretend to be fine."),
}


def render_facts(seed: dict) -> str:
    lines = []
    for f in seed.get("persona", {}).get("facts", []):
        lines.append(f"- {f.get('fact_en', '')} (evidence: {f.get('evidence', '')})")
    return "\n".join(lines) or "(no persona facts)"


def render_measurements(seed: dict) -> str:
    m = seed.get("persona", {}).get("measurements", {}) or {}
    big5 = {k.split(".")[0]: v for k, v in m.items()
            if k.split(".")[0] in ("extrovert", "agreeable", "conscientious",
                                   "neurotic", "open")}
    if not big5:
        return ""
    return ("Personality (Big-Five self-report, 1-5): "
            + ", ".join(f"{k}={v}" for k, v in big5.items()))


def render_bdi(seed: dict) -> str:
    bdi = seed.get("initial_bdi", {})
    lines = ["Your internal mental state (use it to inform your reactions):"]
    for kind in ("beliefs", "desires", "intentions"):
        for b in bdi.get(kind, []):
            lines.append(f"- {kind[:-1]}: {b.get('content_en', '')} "
                         f"(strength {b.get('strength', 2.0)})")
    return "\n".join(lines) if len(lines) > 1 else ""


def roleplay_core(seed: dict) -> str:
    """PPDPP Table 11 风格的角色扮演开场（任务视图）。"""
    task = seed["task"]["task_id"]
    if task == "craigslistbargain":
        u = seed.get("user_task_view", {})
        item = u.get("item", {}) or {}
        desc = " ".join(item.get("Description", []) or []) or "(no description)"
        return (
            "Now enter the role-playing mode. In the following conversation, you "
            "will play as a seller in a price bargaining game.\n"
            f"You are the seller who is trying to sell the {item.get('Title', 'item')} "
            f"with the price of {u.get('target', item.get('Price', '?'))}.\n"
            f"Product description: {desc}\n"
            "Reply in character with 1-3 sentences. You may end the conversation "
            "when it should stop (deal reached, buyer walks away, or you no longer "
            "want to continue)."
        )
    if task == "p4g":
        return (
            "Now enter the role-playing mode. In the following conversation, you "
            "will play as a Persuadee who is being persuaded to donate to a charity "
            "called Save the Children.\n"
            "Reply in character with 1-3 sentences. You may donate, refuse, ask "
            "questions, or end the conversation whenever you like."
        )
    situation = seed.get("persona", {}).get("situation_en") or "(unknown)"
    return (
        "Now enter the role-playing mode. In the following conversation, you will "
        "play as a help-seeker who has come to talk about an emotional problem.\n"
        f"Your situation: {situation}\n"
        "Reply in character with 1-3 sentences. You may end the conversation when "
        "you feel done talking."
    )


def system_for(seed: dict, variant: str) -> str:
    parts = [roleplay_core(seed)]
    if variant in ("persona", "persona_resist", "bdi"):
        parts.append("About you (traits and evidence from your past "
                     "conversations/questionnaires):\n" + render_facts(seed))
    if variant == "persona_resist":
        meas = render_measurements(seed)
        if meas:
            parts.append(meas)
        parts.append(RESIST_TAGS[seed["task"]["task_id"]])
    if variant == "bdi":
        parts.append(render_bdi(seed))
    return "\n".join(p for p in parts if p)


class StandardUserSim:
    """提示式用户模拟器（variant ∈ roleplay/persona/persona_resist/bdi）。"""

    def __init__(self, llm, variant: str = "persona"):
        self.llm = llm
        self.variant = variant
        self.task = None
        self.seed = None
        self.history: list[dict] = []
        self.turns = 0
        self.ended = False

    def reset(self, seed: dict) -> dict:
        self.seed = seed
        self.task = seed["task"]["task_id"]
        self.history = prefix_turns(seed)
        self.turns = 0
        self.ended = False
        return {"checkpoint": {}}

    def step(self, checkpoint: dict, utterance: str) -> dict:
        self.history.append({"role": "assistant", "text": utterance})
        system = system_for(self.seed, self.variant) + "\n" + JSON_TAG
        msgs = [{"role": "system", "content": system}]
        # 全量历史（安全阀 30 轮内无需窗口）
        for h in self.history:
            msgs.append({"role": "assistant" if h["role"] == "assistant" else "user",
                         "content": h["text"]})
        try:
            out = self.llm.chat_json(msgs, max_tok=200)
        except StructuredCallError:
            # 高并发下偶发解析失败：回退为中性回复继续对话（不误终、不炸批次）
            reply = "I see. Please go on."
            self.history.append({"role": "user", "text": reply})
            self.turns += 1
            return {"user_reply": reply, "terminated": False,
                    "termination_reason": "continue",
                    "costs": {"calls": 1, "prompt_tokens": None,
                              "completion_tokens": None}}
        end = bool(out.get("end", False))
        reply = (out.get("reply") or "").strip()
        if reply:
            self.history.append({"role": "user", "text": reply})
        self.turns += 1
        self.ended = self.ended or end
        return {"user_reply": reply, "terminated": end,
                "termination_reason": "user_ended" if end else "continue",
                "costs": {"calls": 1, "prompt_tokens": None,
                          "completion_tokens": None}}
