"""策略注册表：--method 名 → Policy 类。M2/M3 在此追加新基线。"""
from __future__ import annotations

from baseline.policies.policy import Policy
from baseline.policies.dialogxpert_policy import DialogXpertPolicy
from baseline.policies.prompt_policies import (AnEPolicy, MiPromptPolicy,
                                               ProactivePolicy, ProCoTPolicy,
                                               StandardPolicy)
from baseline.policies.sft_policy import SftPolicy

REGISTRY: dict[str, type[Policy]] = {
    "standard": StandardPolicy,
    "proactive": ProactivePolicy,
    "procot": ProCoTPolicy,
    "ane": AnEPolicy,
    "mi_prompt": MiPromptPolicy,
    "sft_labeled": SftPolicy,
    "dialogxpert": DialogXpertPolicy,
}


def make_policy(method: str, llm) -> Policy:
    """按方法名构造策略实例（每个 case 一个独立实例，账本干净）。"""
    if method not in REGISTRY:
        raise ValueError(f"未知方法 {method!r}，可用：{sorted(REGISTRY)}")
    return REGISTRY[method](llm)
