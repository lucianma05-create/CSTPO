"""本地 vLLM 客户端：与 LLMClient 同接口（chat/计数器），供策略层使用。

- 环境/裁判（Cog-Sim 用户、judge rev4）仍走 deepseek LLMClient，保持
  同条件口径；只有"策略骨干"换成本地 Qwen3-14B。
- temperature 可配（评估 greedy=0；DialogXpert 候选采样=0.7）。
"""
from __future__ import annotations

import os

from openai import OpenAI

DEFAULT_URL = os.environ.get("CSTPO_VLLM_URL", "http://127.0.0.1:8001/v1")
DEFAULT_MODEL = os.environ.get("CSTPO_VLLM_MODEL", "/publicdata/model/Qwen3-14B")


class VLLMClient:
    def __init__(self, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600,
                             max_retries=2)
        self.model = model
        self.temperature = temperature
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, messages: list[dict], max_tok: int = 512,
             json_mode: bool = False) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tok,
            temperature=self.temperature,
            # Qwen3 思考模型必须显式关 thinking，否则输出 <think> 推理块
            # 污染对话（与 SFT 侧 enable_thinking=False 口径一致）。
            # 旧版 openai SDK 不支持顶层 chat_template_kwargs，走 extra_body。
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        self.calls += 1
        if resp.usage:
            self.prompt_tokens += resp.usage.prompt_tokens
            self.completion_tokens += resp.usage.completion_tokens
        return (resp.choices[0].message.content or "").strip()
