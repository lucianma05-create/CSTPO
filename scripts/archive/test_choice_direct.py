"""根因验证：绕过 OpenAI 端点，直接 AsyncLLM + SamplingParams 构造
（verl generate 端点同款路径），测 choice 约束是否仍生效。

若此路径下标签段生成满 max_tokens 垃圾 → verl 端点的 structured_outputs
未走 Processor 验证 → 约束失效 → RL 乱码根因坐实。
"""
import asyncio
import sys

sys.path.insert(0, "/data/user21300120/mmh/CSTPO")

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine.async_llm import AsyncLLM

MODEL = "/publicdata/model/Qwen3-8B"
ESCONV_LABELS = ["Question", "Restatement or Paraphrasing", "Reflection of feelings",
                 "Self-disclosure", "Affirmation and Reassurance",
                 "Providing Suggestions", "Information", "Others"]


async def main():
    engine_args = AsyncEngineArgs(model=MODEL, gpu_memory_utilization=0.25,
                                  max_model_len=8192, enforce_eager=True)
    vllm_config = engine_args.create_engine_config()
    llm = AsyncLLM.from_vllm_config(vllm_config=vllm_config)

    tk = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "system", "content": "You are a helpful and caring friend."},
            {"role": "user", "content": "I am feeling sad. My boyfriend left me."}]
    text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    prompt_ids = tk.encode(text, add_special_tokens=False)

    for trial in range(3):
        params = SamplingParams(
            max_tokens=8, temperature=0.2,
            structured_outputs=StructuredOutputsParams(choice=ESCONV_LABELS),
        )
        async for out in llm.generate({"prompt_token_ids": prompt_ids},
                                      sampling_params=params, request_id=f"t{trial}"):
            final = out
        ids = final.outputs[0].token_ids
        txt = tk.decode(ids, skip_special_tokens=True)
        print(f"trial {trial}: {len(ids)} tokens, text={txt!r}")
        ok = txt.strip() in ESCONV_LABELS
        print(f"  合法标签: {ok}")


if __name__ == "__main__":
    asyncio.run(main())
