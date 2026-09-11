"""DeepSeek 兼容接口封装（OpenAI SDK）。

读取项目根目录 .env 的 DEEPSEEK_API_KEY，base_url 固定为 api.deepseek.com。
本环境可用模型：deepseek-flash / deepseek-v4-pro，默认 deepseek-flash。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

MODEL_FLASH = "deepseek-flash"
MODEL_PRO = "deepseek-v4-pro"


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _extract_json(text: str):
    """从模型输出提取 JSON，容忍 markdown 代码块与前后缀文字。"""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"无法解析 JSON: {text[:200]}")


class LLMClient:
    """极薄封装：chat 返回纯文本，chat_json 返回解析后的 dict。"""

    def __init__(self, model: str | None = None):
        env = _load_env()
        api_key = os.environ.get("DEEPSEEK_API_KEY", env.get("DEEPSEEK_API_KEY", ""))
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY（.env 或环境变量）")
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key,
            timeout=120,
            max_retries=2,
        )
        self.model = model or env.get("MODEL", MODEL_FLASH)

    def chat(self, messages, max_tok: int = 512, json_mode: bool = False) -> str:
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_tok)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            # 关闭推理链，避免 reasoning_content 挤占输出预算（与 CogWM 一致）
            r = self.client.chat.completions.create(
                temperature=0.0,
                extra_body={"thinking": {"type": "disabled"}},
                **kwargs,
            )
        except Exception:
            r = self.client.chat.completions.create(temperature=0.0, **kwargs)
        return (r.choices[0].message.content or "").strip()

    def chat_json(self, messages, max_tok: int = 512) -> dict:
        out = _extract_json(self.chat(messages, max_tok=max_tok, json_mode=True))
        if not isinstance(out, dict):
            raise ValueError(f"JSON 输出不是对象: {type(out)}")
        return out
