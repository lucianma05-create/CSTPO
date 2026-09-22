"""验证1：actor checkpoint（LoRA）经 HF merge 后的权重是否干净。

裁决逻辑：
- HF merge（已验证干净的合并路径）+ 本地生成若无 CJK/乱码 →
  RL 更新无罪，腐败在 verl 同步链（get_per_tensor_param→vllm load_weights）；
- 若 HF merge 后同样出 CJK/乱码 → 训练更新本身有问题。

用法：python scripts/test_actor_ckpt_clean.py <ckpt_dir> [n_seeds]
生成口径与 ab_validate 一致：greedy + 三件套（rep 1.15/ngram 4/suppress）。
"""
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "Cog-Sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pyarrow.parquet as pq
from cstpo.core.label_maps import ESCONV_CANONICAL
from cstpo.core.task_env import TaskEnv, prefix_turns

CKPT = Path(sys.argv[1])
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE = "/publicdata/model/CSTPO/esconv"
LABELS = ESCONV_CANONICAL

tk = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            device_map="cuda")
model = PeftModel.from_pretrained(model, CKPT)
model = model.merge_and_unload()
model.eval()

bad_ids = []
for tok in ("<|url|>", "<|code|>", "<|audio|>", "<|video|>", "<|quote|>",
            "<|think|>", "</think>"):
    t = tk.convert_tokens_to_ids(tok)
    if isinstance(t, int) and t >= 0:
        bad_ids.append(t)

ROLE_RE = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)


def gen(messages):
    text = tk.apply_chat_template(messages, tokenize=False,
                                  add_generation_prompt=True,
                                  enable_thinking=False)
    ids = tk(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=128, do_sample=False,
                         repetition_penalty=1.15, no_repeat_ngram_size=4,
                         suppress_tokens=bad_ids if bad_ids else None)
    gen_ids = out[0][ids["input_ids"].shape[1]:]
    raw = tk.decode(gen_ids, skip_special_tokens=True).strip() or "(keep talking)"
    # 两字段：首行是合法标签则剥掉
    if "\n" in raw:
        first, rest = raw.split("\n", 1)
        if first.strip() in LABELS:
            raw = rest.strip() or "(keep talking)"
    prev = None
    while prev != raw:
        prev = raw
        raw = ROLE_RE.sub("", raw, count=1)
    return raw.strip() or "(keep talking)"


def one(row):
    seed = json.loads(row["seed_json"])
    messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
    env = TaskEnv(max_turns=12)
    cp = env.reset(seed)["checkpoint"]
    turns = prefix_turns(seed)
    termination = None
    for _ in range(12):
        utter = gen(messages)
        step = env.step(cp, utter)
        cp = step["checkpoint"]
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": step["user_reply"]})
        messages.append({"role": "assistant", "content": utter})
        messages.append({"role": "user", "content": step["user_reply"]})
        if step["terminated"]:
            termination = step["termination_reason"]
            break
    return {"seed_id": row["seed_id"], "dialogue": turns, "termination": termination}


def main():
    rows = pq.read_table(ROOT / "data" / "mcppo_esconv" / "train.parquet").to_pylist()[:N_SEEDS]
    print(f"ckpt={CKPT} base={BASE} seeds={len(rows)}", flush=True)
    out = [one(r) for r in rows]
    LANG = re.compile(r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ฀-๿぀-ヿ가-힯]")
    CJK = re.compile(r"[一-鿿]")
    garb = cjk = bang = 0
    for r in out:
        at = [t["text"] for t in r["dialogue"] if t["role"] == "assistant"]
        if any(LANG.search(t) for t in at):
            garb += 1
        if any(CJK.search(t) for t in at):
            cjk += 1
        if any("!" in t and t.count("!") / len(t) > 0.5 for t in at):
            bang += 1
    print(f"结果（{len(out)} 条）：乱码脚本 {garb} / CJK {cjk} / bang {bang}", flush=True)
    for r in out:
        at = [t["text"] for t in r["dialogue"] if t["role"] == "assistant"]
        print(f"  {r['seed_id']}: {r['termination']}, turns={len(r['dialogue'])}",
              f"| 首句: {at[0][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
