"""保存 HF merge_and_unload 的 SFT 权重（含 tokenizer），供 vllm 加载测试。"""
import sys
sys.path.insert(0, "/data/user21300120/mmh/CSTPO")
from pathlib import Path
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SRC = "/publicdata/model/Qwen3-8B"
ADAPTER = "/data/user21300120/mmh/CSTPO/outputs/sft/esconv/checkpoint-604"
OUT = Path("/tmp/qwen3_esconv_sft_merged")

m = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto".__class__ and __import__("torch").bfloat16,
                                        attn_implementation="sdpa")
m = PeftModel.from_pretrained(m, ADAPTER, is_trainable=False)
m = m.merge_and_unload()
m.save_pretrained(OUT, safe_serialization=True)
tk = AutoTokenizer.from_pretrained(SRC)
tk.save_pretrained(OUT)
print("merged model + tokenizer saved to", OUT)
