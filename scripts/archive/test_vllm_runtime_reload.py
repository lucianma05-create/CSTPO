"""vllm 运行时 load_weights 重放：干净权重经运行时更新后是否腐败。

背景：RL 里同步 dump 证明 verl 发送的 merged 权重数值干净（≈SFT 基座），
但 gs=2 起服务端输出 100% CJK。本实验在 standalone vllm 引擎上复刻：
初始加载（磁盘 safetensors）→ 生成 → 运行时 load_weights（同一份干净权重）
→ 生成 → 再 load 数次。若 load_weights 后输出腐败 → vllm 运行时权重更新
路径（vllm 0.28 + Qwen3 fused QKV）是元凶。
"""
import asyncio
import re

from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt

BASE = "/publicdata/model/CSTPO/esconv"
tk = AutoTokenizer.from_pretrained(BASE)
CJK = re.compile(r"[一-鿿]")
LANG = re.compile(r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ฀-๿぀-ヿ가-힯]")

MSGS = [{"role": "system", "content": "You are a helpful and caring friend."},
        {"role": "user", "content": "I am feeling sad. My boyfriend left me."}]
PROMPT_IDS = tk.encode(
    tk.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True,
                           enable_thinking=False), add_special_tokens=False)


async def gen(engine, i):
    out = None
    async for o in engine.generate(TokensPrompt(prompt_token_ids=PROMPT_IDS),
                                   SamplingParams(max_tokens=128, temperature=0),
                                   request_id=f"r{i}"):
        out = o
    return out.outputs[0].text


async def main():
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=BASE, enforce_eager=True, gpu_memory_utilization=0.4,
        max_model_len=8192))
    t = await gen(engine, 0)
    print(f"初始加载生成: {t[:70]!r} | CJK={bool(CJK.search(t))} 脚本={bool(LANG.search(t))}",
          flush=True)
    # 走 weights_path（RPC 传大权重列表有序列化问题；worker 内从磁盘读等价）
    for k in range(4):
        await engine.collective_rpc("reload_weights", kwargs={"weights_path": BASE})
        t = await gen(engine, 1 + k)
        print(f"第{k+1}次 reload_weights(磁盘) 后: {t[:70]!r} | CJK={bool(CJK.search(t))} 脚本={bool(LANG.search(t))}",
              flush=True)

    # verl 的真实路径：model.load_weights(list[(name, tensor)])（内存权重直灌）
    def _load_inmem(worker):
        from safetensors.torch import load_file
        w = list(load_file(BASE + "/model.safetensors").items())
        return worker.model_runner.model.load_weights(w)

    for k in range(2):
        await engine.collective_rpc(_load_inmem)
        t = await gen(engine, 10 + k)
        print(f"第{k+1}次 model.load_weights(内存) 后: {t[:70]!r} | CJK={bool(CJK.search(t))} 脚本={bool(LANG.search(t))}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
