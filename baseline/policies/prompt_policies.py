"""提示族策略（API 底座 deepseek-flash，零训练）。

- StandardPolicy：论文香草提示直生成（cstpo/core/agent.py 的 vanilla，
  提示来源 PPDPP Table 8 / TRIP Table 20 / ESConv-SRA template6）；
- ProactivePolicy：Deng 2023a 主动提示 p(a,r|D,C,A)——先选策略再回应；
- ProCoTPolicy：Deng 2023a ProCoT p(t,a,r|D,C,A)——进度分析 + 目标 +
  策略 + 回应；模板结构取自 vendor/LLM-Proactive/negotiate/prompt.txt；
- AnEPolicy：Ask-an-Expert（Zhang 2023b）——先写专家建议再回应；
- MiPromptPolicy：Chen 2023 混合主动策略提示——两步：模型先选策略标签，
  再把标签映射为自然语言指令注入第二次调用生成回应。

适配口径（与原仓库的差异，README 有记录）：
- 原仓库把完整 CoT 输出直接当回复喂给用户模拟器；本实现只提取最终
  response 段（"the response is" 之后）作为话语，避免元文本污染 Cog-Sim
  对话与 judge 判分；
- 策略词表用本仓库三任务 canonical 标签（label_maps），不用原仓库的
  CB 卖方策略集（我们 Actor 是买方，且词表需与 SFT/RL 一致）。
"""
from __future__ import annotations

import re

from baseline.policies.policy import Policy, strip_role_prefix

# 原仓库 prompt.txt 的回应标记（模型输出 "..., the response is ..."）
_RESP_MARKERS = ("the response is", "the response:", "response is:",
                 "response:", "based on the expert's answer, your response is",
                 "based on that, your response is")


def extract_response(text: str) -> str:
    """取最终回应段：最后一个回应标记之后的文本。

    无回应标记时（模型不按格式输出）：
    - 若存在专家建议标题（AnE 模型常把回应放前面、建议放后面），取标题
      之前的文本；
    - 否则原样返回。
    原仓库将完整 CoT 输出当回复；我们只把最终回应送入对话（口径偏差，
    见模块 docstring）。
    """
    t = strip_role_prefix(text)
    low = t.lower()
    idx, hit = -1, ""
    for m in _RESP_MARKERS:
        i = low.rfind(m)
        if i > idx:
            idx, hit = i, m
    if idx >= 0:
        rest = t[idx + len(hit):]
    else:
        advice_idx = None
        for pat in ("**advice", "advice from", "expert advice",
                    "expert's advice", "the expert would", "expert's answer"):
            i = low.find(pat)
            if i >= 0 and (advice_idx is None or i < advice_idx):
                advice_idx = i
        if advice_idx is not None and advice_idx > 0:
            rest = t[:advice_idx]
        else:
            rest = t
    rest = rest.strip()
    # 剥掉包裹引号、冒号与 markdown 加粗残余（"**"、星号、反引号）
    rest = rest.strip("\"'“”‘’")
    rest = re.sub(r"^[\*\s:]+", "", rest)
    rest = re.sub(r"[\*\s]+$", "", rest)
    out = strip_role_prefix(rest)
    if not out or re.fullmatch(r"[\W_]*", out):  # 空白/纯标点（如 "."）兜底
        return "(keep talking)"
    return out


def _vocab(task: str) -> list[str]:
    """三任务 canonical 策略词表（与 SFT/RL 同一 label_maps）。"""
    from cstpo.core.label_maps import (CB_CANONICAL, ESCONV_CANONICAL,
                                       P4G_CANONICAL)
    return {"esconv": ESCONV_CANONICAL, "p4g": P4G_CANONICAL,
            "craigslistbargain": CB_CANONICAL}[task]


def _task_preamble(seed: dict) -> str:
    """任务角色前导（agent.py 的 vanilla 提示去掉末尾约束句）。"""
    from cstpo.core.agent import SYSTEM_BUILDERS

    task = seed["task"]["task_id"]
    base = SYSTEM_BUILDERS[task](seed)
    # 去掉 vanilla 的短句约束（与 CoT 指令冲突），保留角色与任务视图
    for tail in ("Please reply with only one short and succinct sentence.\n"
                 "Now start the game.",
                 "Please reply with only one short and persuasive sentence.",
                 "Please help your friend by continuing the conversation. "
                 "Make your response short and to the point.\n"
                 "Respond in this format: assistant: <response>"):
        base = base.replace(tail, "")
    return base.strip()


_STRATEGY_HEAD = "first select the most appropriate strategy from the list {}"
_ANALYSIS_HEAD = ("first analyse the current progress and consider an "
                  "appropriate goal, then select the most appropriate "
                  "strategy from the list {}")


def _procot_instruction(task: str, with_analysis: bool) -> str:
    head = _ANALYSIS_HEAD if with_analysis else _STRATEGY_HEAD
    if with_analysis:
        return (
            "Given the task and the conversation history, in order to achieve "
            "the goal, " + head.format(_vocab(task)) + ". "
            "The reply should start with the analysis of the current progress "
            "and an appropriate goal, and then follow by 'To reach this goal, "
            "the most appropriate strategy is []. Based on the selected "
            "strategy, the response is'.")
    return (
        "Given the task and the conversation history, in order to achieve the "
        "goal, " + head.format(_vocab(task)) + ". "
        "The reply should state the selected strategy and then follow by "
        "'Based on the selected strategy, the response is'.")


def _chat(llm, system: str, history: list[dict], max_tok: int = 300) -> str:
    msgs = [{"role": "system", "content": system}]
    for h in history:
        msgs.append({"role": "assistant" if h["role"] == "assistant" else "user",
                     "content": h["text"]})
    return (llm.chat(msgs, max_tok=max_tok) or "").strip()


class _PromptPolicyBase(Policy):
    """共用件：系统提示 + 历史 → 输出 → 回应提取。"""

    backbone = "deepseek-flash"

    def __init__(self, llm):
        self.llm = llm

    def _history(self, seed, turns):
        from cstpo.core.task_env import prefix_turns
        return prefix_turns(seed) + turns

    def _system(self, seed: dict) -> str:
        raise NotImplementedError


class StandardPolicy(_PromptPolicyBase):
    """香草提示直生成：无策略注入、无 CoT。"""

    name = "standard"

    def turn(self, seed: dict, turns: list[dict]) -> str:
        from cstpo.core.agent import agent_turn

        task = seed["task"]["task_id"]
        out = agent_turn(self.llm, task, "vanilla",
                         self._history(seed, turns), seed)
        return strip_role_prefix(out) or "(keep talking)"


class ProactivePolicy(_PromptPolicyBase):
    """Deng 2023a 主动提示：a→r（先选策略再回应，无分析步）。"""

    name = "proactive"

    def _system(self, seed):
        return (_task_preamble(seed) + "\n"
                + _procot_instruction(seed["task"]["task_id"],
                                      with_analysis=False))

    def turn(self, seed, turns):
        out = _chat(self.llm, self._system(seed),
                    self._history(seed, turns))
        return extract_response(out)


class ProCoTPolicy(_PromptPolicyBase):
    """Deng 2023a ProCoT：t→a→r（分析 + 目标 + 策略 + 回应）。"""

    name = "procot"

    def _system(self, seed):
        return (_task_preamble(seed) + "\n"
                + _procot_instruction(seed["task"]["task_id"],
                                      with_analysis=True))

    def turn(self, seed, turns):
        out = _chat(self.llm, self._system(seed),
                    self._history(seed, turns))
        return extract_response(out)


class AnEPolicy(_PromptPolicyBase):
    """Ask-an-Expert（Zhang 2023b）：先写专家建议，再按其生成回应。"""

    name = "ane"

    _EXPERT = {"esconv": "a psychological counselling expert",
               "p4g": "a persuasion expert",
               "craigslistbargain": "a negotiation expert"}

    def _system(self, seed):
        task = seed["task"]["task_id"]
        return (
            _task_preamble(seed) + "\n"
            f"Before responding, consult {self._EXPERT[task]}: first write "
            "down the advice this expert would give for the best next step in "
            "the current situation. Then, based on the expert's answer, "
            "generate your response. The reply should end with 'Based on the "
            "expert's answer, the response is'.")

    def turn(self, seed, turns):
        out = _chat(self.llm, self._system(seed),
                    self._history(seed, turns))
        return extract_response(out)


class MiPromptPolicy(_PromptPolicyBase):
    """Chen 2023 混合主动策略提示（单步链，挂在与 Proactive 同构的选择后）。

    链式结构：先选策略标签 → 把标签译为自然语言指令（MI-Prompt 映射）→
    按指令生成回应。与文献 "+ MI-Prompt" 附加件用法一致。
    """

    name = "mi_prompt"

    def _system(self, seed):
        task = seed["task"]["task_id"]
        return (
            _task_preamble(seed) + "\n"
            "Given the task and the conversation history, in order to achieve "
            "the goal, " + _STRATEGY_HEAD.format(_vocab(task)) + ". "
            "Then translate the selected strategy into a brief "
            "natural-language instruction for what to say next, and generate "
            "the response by exactly following that instruction. "
            "The reply should state the selected strategy and the translated "
            "instruction, and then follow by 'the response is'.")

    def turn(self, seed, turns):
        out = _chat(self.llm, self._system(seed),
                    self._history(seed, turns))
        return extract_response(out)
