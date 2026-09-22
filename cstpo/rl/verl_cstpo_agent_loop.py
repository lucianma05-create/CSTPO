"""CSTPO 多轮 agent loop（verl 0.8 agent_loop 接口）。

每轮：策略模型生成两字段输出（策略标签 + 话语）→ 剥离标签 → TaskEnv /
Cog-Sim simulator 生成用户回复 → 增量拼接（user token mask=0）→ 直至环境
终止或 max_assistant_turns。完整对话经 extra_fields 传给 reward loop 做终局
judge 聚合（异步，不阻塞生成循环）。

注册方式（verl rollout.agent）：
  agent_loop_config_path: cstpo/configs/cstpo_agent_loop.yaml
  default_agent_loop: cstpo_agent

第一版简化（相对 verl_env_adapter.run_episode）：
  - 不接 15 轮 judger（对话以 max_assistant_turns=12 硬顶）；
  - trie 约束采样后续接入（vllm logits processor，见 trie_qwen）。
"""
from __future__ import annotations

import json
import re
import sys
from functools import partial
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "Cog-Sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

from cstpo.core.label_maps import CB_CANONICAL, ESCONV_CANONICAL, P4G_MAP
from cstpo.core.task_env import TaskEnv, prefix_turns

KNOWN_STRATEGY_LABELS = {
    "esconv": frozenset(ESCONV_CANONICAL),
    "p4g": frozenset(P4G_MAP.values()),
    "cb": frozenset(CB_CANONICAL),
    # 别名：seed task_id 用的是全名（build_prompt/ab_validate 的 task 串）
    "craigslistbargain": frozenset(CB_CANONICAL),
}


def strip_strategy_label(text: str, task: str) -> str:
    """两字段输出解析：首行若是该任务的已知策略标签，剥离；否则整段视为话语。

    "<unlabeled>" 也按标签剥离（SFT 全量口径下模型可能吐出——p4g/cb 大量
    目标回合本身无标签，防止其泄漏进模拟器话语）。"""
    text = (text or "").strip() or "(keep talking)"
    if "\n" in text:
        first, rest = text.split("\n", 1)
        first_s = first.strip()
        known = KNOWN_STRATEGY_LABELS.get(task, ())
        # 标签变体识别（无约束推理的格式漂移，如 "propose_price_other"）：
        # 首行以已知标签开头且后接 "_" 也视为标签行剥离
        is_label = first_s in known or first_s == "<unlabeled>" \
            or any(first_s.startswith(k + "_") for k in known)
        if is_label:
            return rest.strip() or "(keep talking)"
    return text


_ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)


def build_label_trie(tokenizer, labels) -> dict:
    """choice 约束的 token 前缀 trie：prefix tuple → 允许的下一 token 集合。

    供 actor 侧对标签段做约束内重归一化（技术债 #8：π_θ 与 π_old 都在
    约束条件分布上计算，标签参与 loss 后 ratio 良定义）。"""
    trie = {}
    for lab in labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        for k in range(len(ids)):
            trie.setdefault(tuple(ids[:k]), set()).add(ids[k])
    return trie


def strip_role_prefix(text: str) -> str:
    """两段式协议下模型会模仿模板在话语前写 'assistant:'/'user:' 前缀
    （判别实验三档 17-18/19 存在，与权重/温度/rep 均无关）。剥离后再进
    模拟器与 judge；token 侧保持原样（mask 与生成一致）。"""
    prev = None
    while prev != text:
        prev = text
        text = _ROLE_PREFIX_RE.sub("", text, count=1)
    return text.strip() or "(keep talking)"


def strip_embedded_labels(text: str, task: str) -> str:
    """话语正文中段模仿两字段格式写出的标签行剥离（v1c 轨迹实测 1.2%
    消息含独立标签行，如 "What kind...\nOthers\nSorry..."——模型在自由
    生成段提前写了下一轮的标签）。只删首行之外恰好等于已知标签词的独立
    行；首行标签由 strip_strategy_label 处理。仅作用于进模拟器/judge 的
    文本，token 侧不动。"""
    known = KNOWN_STRATEGY_LABELS.get(task, ())
    lines = text.split("\n")
    kept = [lines[0]] + [ln for ln in lines[1:] if ln.strip() not in known]
    return "\n".join(kept).strip() or "(keep talking)"


@register("cstpo_agent")
class CstpoAgentLoop(AgentLoopBase):
    """CSTPO 对话轨迹生成循环：agent 话语 ↔ Cog-Sim simulator 用户回复。"""

    # 每轮话语生成 token 上限（GEN_CONFIG.max_new_tokens 口径）
    TURN_MAX_TOKENS = 128
    # 策略标签段 token 上限（最长标签 "Restatement or Paraphrasing" < 8）
    LABEL_MAX_TOKENS = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns or 12
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        seed = json.loads(kwargs["seed_json"])
        task = seed["task"]["task_id"]
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        # 每轮 cap：vllm 默认 max_tokens=min(response_length, 剩余预算)，
        # 不 cap 的话首轮就会吃光 12 轮的总预算（见 vllm_async_server.py:490）
        sampling_params = {**sampling_params, "max_tokens": self.TURN_MAX_TOKENS}
        # SFT 推理口径对齐：verl agent loop 硬编码 repetition_penalty=1.0
        # （experimental/agent_loop/agent_loop.py:500，rollout 配置传值无效）。
        # 1.15 是 SFT 三件套口径——判别实验证实其为跨轮固定回复循环的解药
        # （HF-merge 权重 + rep=1.0 时 greedy 12/19 循环，rep=1.15 时 0/19）。
        sampling_params = {**sampling_params, "repetition_penalty": 1.15}
        # URL 退化硬禁（与 A/B/eval 的 bad_words 同口径；数据清洗治本，
        # 此处兜底残留/泛化）
        sampling_params["bad_words"] = [
            w for w in ("URL", " Url", " url") if self.tokenizer.encode(w)][:3]
        # 硬预算：verl 把 response_ids 装进 response_length 定长缓冲，溢出
        # 直接崩（"Sizes of tensors must match... 4105 vs 4096"，v3 踩中）。
        # 每轮最坏增量 ≈ 标签 8 + 换行 1 + 话语 128 + 用户回复模板 ~300，
        # 提前 500 token 刹车，保证最终 mask ≤ response_length。
        self._turn_budget = max(self.response_length - 500, 512)

        # all_ids 累积一切 token（原始 prompt + 各轮生成 + user 回复）；
        # 最终输出按 tool_agent 约定拆分：prompt_ids = 原始 prompt 部分，
        # response_ids = 全部 mask 覆盖部分（含 user token，mask=0 区分）。
        all_ids = await self.apply_chat_template(messages)
        response_mask: list[int] = []

        # 环境：前缀轮次 + simulator（Cog-Sim，llm=None 内部默认 DeepSeek 客户端）
        turns = prefix_turns(seed)
        env = TaskEnv(max_turns=self.max_assistant_turns)
        cp = env.reset(seed)["checkpoint"]
        termination = None
        num_turns = 0
        label_log: list[str] = []  # 诊断：每轮标签段原文
        # 约束内 ratio（#8）：记录标签段的响应内偏移 + 每个位置的允许 token 集，
        # 供 actor 侧在 choice 掩码下重归一化 logprob
        label_trie = build_label_trie(self.tokenizer,
                                      sorted(KNOWN_STRATEGY_LABELS[task]))
        label_offsets: list[int] = []      # 响应内偏移（从 0 起）
        label_allowed: list[list[int]] = []  # 每位置对应的 trie 允许集
        prompt_len = len(all_ids)  # 此刻 response_mask 为空，all_ids 即 prompt

        for _ in range(self.max_assistant_turns):
            if len(response_mask) >= self._turn_budget:
                termination = "response_length_cap"
                break
            # 1a. 策略标签段：choice 约束（vllm structured_outputs → xgrammar）
            label_params = {
                **sampling_params,
                "max_tokens": self.LABEL_MAX_TOKENS,
                "structured_outputs": {"choice": sorted(KNOWN_STRATEGY_LABELS[task])},
            }
            label_out = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=all_ids,
                sampling_params=label_params,
            )
            label_ids = label_out.token_ids
            label_text = self.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
            if label_text not in KNOWN_STRATEGY_LABELS[task]:
                # 兜底（约束失效/截断垃圾时）：必须用合法标签的 token 替换，
                # 否则垃圾 token 进 all_ids → 话语段从垃圾后级联乱码
                label_text = sorted(KNOWN_STRATEGY_LABELS[task])[0]
            # 标签段 token 规范化：vllm choice 采样可能产出与 trie 规范编码
            # 不同的 token 路径（前导空格/换行等变体，decode+strip 后仍是
            # 合法标签所以上面的兜底不触发）——导致 actor 侧 choice 掩码的
            # 允许集不含实际 token → logprob=-inf → -inf-(-inf)=NaN 污染
            # ratio（正式 run actor 指标全 NaN 的根因，仪器实测每标签一个
            # -inf）。统一按解码文本重编码为规范路径——与 build_label_trie
            # 同源，掩码允许集必然包含实际 token。
            label_ids = self.tokenizer.encode(label_text, add_special_tokens=False)
            label_log.append(label_text)
            # 约束内 ratio（#8）：记录位置与允许集（按最终 label_ids，兜底后
            # 的标签字符串必在标签集内，每个位置的 token 必在 trie 允许集内）
            seg_start = len(response_mask)
            for k in range(len(label_ids)):
                allowed = label_trie.get(tuple(label_ids[:k]),
                                         label_trie.get(tuple(), set()))
                label_offsets.append(seg_start + k)
                label_allowed.append(sorted(allowed))
            all_ids += label_ids
            # 标签 token 参与 loss（#8：π_old/π_θ 都在约束下重算后，
            # ratio 良定义，标签获得梯度——策略选择可优化）
            response_mask += [1] * len(label_ids)

            # 1b. 话语段：自由生成（prompt 续上换行，模型从 \n 后开始说）
            nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            utter_params = {**sampling_params, "max_tokens": self.TURN_MAX_TOKENS}
            utter_out = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=all_ids + nl_ids,
                sampling_params=utter_params,
            )
            ids = utter_out.token_ids
            utter = strip_role_prefix(self.tokenizer.decode(ids, skip_special_tokens=True))
            utter = strip_embedded_labels(utter, task)
            all_ids += nl_ids + ids
            response_mask += [0] * len(nl_ids) + [1] * len(ids)
            if len(response_mask) >= self.response_length:
                termination = "response_length_cap"
                break

            # 2. 环境步进（同步 LLM 调用 → executor 避免阻塞事件循环）
            step = await self.loop.run_in_executor(None, partial(env.step, cp, utter))
            cp = step["checkpoint"]
            turns.append({"role": "assistant", "text": utter})
            turns.append({"role": "user", "text": step["user_reply"]})
            num_turns += 1
            if step["terminated"]:
                termination = step["termination_reason"]
                break

            # 3. 增量 token（user 回复 mask=0）
            # 修复（2026-09-18 审计）：旧实现 append template([assistant:utter,
            # user:reply]) 的 system 剥离块——该块含完整 assistant 回合，导致
            # 模型上下文里自己的话语出现两次（token 级重复，跨轮循环的嫌疑
            # 来源之一）。正确增量 = 关闭当前 assistant 回合 + 仅 user 回合 +
            # 下一轮生成提示，与 SFT 训练样本结构逐 token 对齐。
            close_ids = self.tokenizer.encode("<|im_end|>\n",
                                              add_special_tokens=False)
            user_ids = await self.apply_chat_template(
                [{"role": "user", "content": step["user_reply"]}],
                remove_system_prompt=True)
            all_ids += close_ids + user_ids
            response_mask += [0] * (len(close_ids) + len(user_ids))
            if len(response_mask) >= self.response_length:
                termination = "response_length_cap"
                break

        # 拆分：prompt_ids = 原始 prompt；response_ids = mask 覆盖的全部 token
        split = len(all_ids) - len(response_mask)
        return AgentLoopOutput(
            prompt_ids=all_ids[:split],
            response_ids=all_ids[split:],
            response_mask=response_mask,
            response_logprobs=None,
            num_turns=num_turns,
            extra_fields={
                "dialogue": turns,
                "seed_json": kwargs["seed_json"],
                "termination": termination,
                "labels": label_log,
                # verl agent_loop worker patch 透传（诊断归因）
                "global_step": kwargs.get("__global_step__", -1),
                "is_validate": kwargs.get("__is_validate__", False),
                # 约束内 ratio（#8）：标签段位置与允许 token 集（actor 侧重归一化用）
                "label_offsets": label_offsets,
                "label_allowed": label_allowed,
                "prompt_len": prompt_len,
            },
            metrics=AgentLoopMetrics(),
        )
