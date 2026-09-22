"""Policy 抽象：基线统一接口。

每个 baseline 只实现 turn(seed, turns) -> str：给定完整种子与当前对话
（含前缀），返回下一句 agent 话语。回合循环、环境、裁判、记账全部由
runner 负责，与主方法同一口径。

后处理统一在策略内部完成：strip_role_prefix（ESConv 模板要求
"assistant: <response>" 格式，RL 评估口径会剥掉角色前缀，基线保持一致；
无前缀时为空操作）。本地工具不从 cstpo/rl/verl_cstpo_agent_loop.py 导入
（该模块依赖 verl，baseline 不引入）。
"""
from __future__ import annotations

import re

_ROLE_PREFIX = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)


def strip_role_prefix(text: str) -> str:
    """剥掉行首 "assistant: / user: " 角色前缀（与 RL 评估口径一致）。

    循环到稳定（模型会输出 "assistant: assistant: ..." 嵌套前缀），
    语义与 cstpo/rl/verl_cstpo_agent_loop.py 的 strip_role_prefix 相同。
    """
    prev = None
    while prev != text:
        prev = text
        text = _ROLE_PREFIX.sub("", text or "", count=1)
    return text.strip() or "(keep talking)"


def strip_strategy_label(text: str, task: str) -> str:
    """两字段输出解析：首行若是该任务已知策略标签则剥离，否则整段视为
    话语。"<unlabeled>" 与 "标签_变体" 也按标签剥离。

    语义与 verl_cstpo_agent_loop.py 的 strip_strategy_label 相同（本地
    实现，不 import verl 模块）。
    """
    from cstpo.core.label_maps import (CB_CANONICAL, ESCONV_CANONICAL,
                                       P4G_CANONICAL)

    known = {"esconv": set(ESCONV_CANONICAL), "p4g": set(P4G_CANONICAL),
             "craigslistbargain": set(CB_CANONICAL)}.get(task, set())
    text = (text or "").strip() or "(keep talking)"
    if "\n" in text:
        first, rest = text.split("\n", 1)
        first_s = first.strip()
        is_label = (first_s in known or first_s == "<unlabeled>"
                    or any(first_s.startswith(k + "_") for k in known))
        if is_label:
            return rest.strip() or "(keep talking)"
    return text


class Policy:
    """策略基类。name/backbone 进入 manifest 与记录。"""

    name: str = "policy"
    backbone: str = "unknown"

    def turn(self, seed: dict, turns: list[dict]) -> str:
        """生成下一句 agent 话语。turns = 前缀 + 已生成的回合（{role,text}）。"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} backbone={self.backbone}>"
