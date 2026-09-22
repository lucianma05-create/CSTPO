"""SFT 策略：只读加载现有 labeled-SFT 合并权重，greedy 三件套推理。

- 权重：/publicdata/model/CSTPO/<task>（每任务独立模型，懒加载 + 缓存；
  三个 8B bf16 模型约 48GB，单张 A100 可容纳）；
- 生成配方：与主方法 SFT 评估口径完全一致（复制自
  cstpo/sft/ab_validate.py 的 pipe：greedy + repetition_penalty 1.15 +
  no_repeat_ngram_size 4 + 特殊 token 抑制 + "URL" bad_words 硬禁 +
  取第一段 + 剥标签/角色前缀）；
- 线程：生成串行（_GEN_LOCK，同 ab_validate），与环境/裁判的 API 调用
  无 GPU 冲突。
"""
from __future__ import annotations

import os
import threading

from baseline.policies.policy import (Policy, strip_role_prefix,
                                      strip_strategy_label)

CKPT_BASE = os.environ.get("CSTPO_SFT_CKPT", "/publicdata/model/CSTPO")

_MODELS: dict[str, callable] = {}   # task -> pipe(msgs, max_new_tokens) -> str
_LOAD_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()


def _pick_free_gpu(min_free_gb: float = 18.0) -> int | None:
    """选剩余显存最大的 torch 设备（torch 枚举，避免 nvidia-smi 编号与
    cuda:N 编号错位——本机 cuda:4 是 4GB 的 A400）。"""
    import torch

    best, best_free = None, 0.0
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        free_gb = free / 2 ** 30
        if free_gb > best_free:
            best, best_free = i, free_gb
    return best if best_free >= min_free_gb else None


def _load_pipe(task: str):
    """懒加载单任务 SFT 模型（三个任务各自独立权重目录，各选最空闲卡）。"""
    import torch  # noqa: F401  确认 torch 可用再加载
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = os.path.join(CKPT_BASE, task)
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"SFT 权重目录不存在：{ckpt}")
    gpu = _pick_free_gpu()
    if gpu is None:
        raise RuntimeError("没有剩余 >=18GB 显存的 GPU，SFT 评估无法加载")
    device = f"cuda:{gpu}"
    print(f"[sft_policy] {task} 加载到 {device}（自动选空闲卡）", flush=True)
    tk = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    # 禁掉 Qwen3 masked 特殊 token 与 "URL" 退化词（ab_validate 同款）
    bad_ids = []
    for tok in ("<|url|>", "<|code|>", "<|audio|>", "<|video|>", "<|quote|>",
                "<think>", "</think>"):
        t = tk.convert_tokens_to_ids(tok)
        if isinstance(t, int) and t >= 0:
            bad_ids.append(t)
    bad_words = [tk.encode(w, add_special_tokens=False)
                 for w in ("URL", " Url", " url")]
    bad_words = [w for w in bad_words if w]

    def pipe(msgs, max_new_tokens=150):
        text = tk.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        ids = tk(text, return_tensors="pt").to(model.device)
        with _GEN_LOCK:
            out = model.generate(
                **ids, max_new_tokens=min(max_new_tokens, 128),
                do_sample=False, repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                suppress_tokens=bad_ids if bad_ids else None,
                bad_words_ids=bad_words)
        gen = out[0][ids["input_ids"].shape[1]:]
        text = tk.decode(gen, skip_special_tokens=True)
        # 取第一个非空段落：防模型续写下一轮对话（多轮泄漏）
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return parts[0] if parts else text

    return pipe


class SftPolicy(Policy):
    """labeled-SFT 检查点策略（不训练新权重，只读推理）。"""

    name = "sft_labeled"

    def __init__(self, llm=None):
        self.llm = llm  # 未使用（本地模型推理）
        self.backbone = f"{CKPT_BASE}/<task>"

    def _pipe(self, task: str):
        if task not in _MODELS:
            with _LOAD_LOCK:
                if task not in _MODELS:
                    _MODELS[task] = _load_pipe(task)
        return _MODELS[task]

    def turn(self, seed: dict, turns: list[dict]) -> str:
        from cstpo.core.agent import SYSTEM_BUILDERS
        from cstpo.core.task_env import prefix_turns

        task = seed["task"]["task_id"]
        msgs = [{"role": "system", "content": SYSTEM_BUILDERS[task](seed)}]
        for h in prefix_turns(seed) + turns:
            msgs.append({"role": "assistant" if h["role"] == "assistant"
                         else "user", "content": h["text"]})
        out = self._pipe(task)(msgs)
        return strip_role_prefix(strip_strategy_label(out, task))
