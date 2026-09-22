"""baseline 侧采样 LLM：LLMClient 硬编码 temperature=0，无法做候选多样性
采样，故在 baseline/ 内用 openai SDK 直接实现带温度的调用（只读外部）。
"""
from __future__ import annotations

import os

from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"


class SamplingLLM:
    """带 temperature 的 chat 调用（候选生成与情绪推断用）。

    后端可切：默认 deepseek API；CSTPO_VLLM_URL 存在时走本地 vLLM
    （策略骨干换 Qwen3-14B 时使用，保持与其他方法同骨干）。
    """

    def __init__(self, temperature: float = 0.7, base_url: str | None = None,
                 model: str | None = None):
        vllm_url = os.environ.get("CSTPO_VLLM_URL")
        if base_url is None and vllm_url:
            base_url, model = vllm_url, os.environ.get(
                "CSTPO_VLLM_MODEL", "/publicdata/model/Qwen3-14B")
        if base_url is None:
            base_url, model = BASE_URL, MODEL
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError("缺少 DEEPSEEK_API_KEY（先跑 runner 的 load_api_key）")
            api_key = key
        else:
            api_key = "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600,
                             max_retries=2)
        self.model = model or MODEL
        self.temperature = temperature
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, messages: list[dict], max_tok: int = 200) -> str:
        kwargs = {"model": self.model, "messages": messages,
                  "max_tokens": max_tok, "temperature": self.temperature}
        # deepseek 走 extra_body 关 thinking；本地 vLLM（Qwen3 思考模型）
        # 走 chat_template_kwargs（与 SFT 侧 enable_thinking=False 一致）
        if os.environ.get("CSTPO_VLLM_URL"):
            kwargs["extra_body"] = {"chat_template_kwargs":
                                    {"enable_thinking": False}}
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = self.client.chat.completions.create(**kwargs)
        self.calls += 1
        if resp.usage:
            self.prompt_tokens += resp.usage.prompt_tokens
            self.completion_tokens += resp.usage.completion_tokens
        return (resp.choices[0].message.content or "").strip()
