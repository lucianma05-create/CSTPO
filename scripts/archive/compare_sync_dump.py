"""验证1+2 合一：同步链 dump 比对 + actor 权重干净性测试。

输入：logs/sync_dump/{sync_merged.pt, adapter.pt}（每次权重同步时由
transformer_impl.py patch 落盘，最后一次同步=训练后最终权重）。

1. 逐张量比对：verl 发送的 merged dict vs 手工 HF 合并（base + B@A*alpha/r）
   - 一致 → verl 检索无错，腐败嫌疑在 vllm load_weights 侧
   - 不一致 → merge 检索本身出错
2. 用手工合并的 actor 权重做多轮生成（greedy+三件套），查 CJK/乱码：
   - 干净 → 更新无罪（同步链有罪）
   - 脏   → RL 更新本身有问题
"""
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "Cog-Sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pyarrow.parquet as pq
from cstpo.core.label_maps import ESCONV_CANONICAL
from cstpo.core.task_env import TaskEnv, prefix_turns

BASE = "/publicdata/model/CSTPO/esconv"
DUMP = ROOT / "logs" / "sync_dump"
RANK, ALPHA = 64, 16
SCALE = ALPHA / RANK

tk = AutoTokenizer.from_pretrained(BASE)
base_sd = torch.load(BASE + "/model.safetensors") if False else None
from safetensors.torch import load_file
base_sd = load_file(BASE + "/model.safetensors")

sync = torch.load(DUMP / "sync_merged.pt")
adapter = torch.load(DUMP / "adapter.pt")
print(f"sync_merged 键数 {len(sync)}，adapter 键数 {len(adapter)}", flush=True)

# 手工合并：base + B@A*scale（peft 默认 use_rslora=False → merged = base + B@A*(alpha/r)）
# adapter 键含嵌套 FSDP 标记（model.layers.N._fsdp_wrapped_module.self_attn...），
# 先归一化再解析
def _norm(k):
    return k.replace("_fsdp_wrapped_module.", "")

manual = {k: v.clone() for k, v in base_sd.items()}
applied = 0
delta_mags = []
for k, v in adapter.items():
    k2 = _norm(k)
    m = re.match(r"base_model\.model\.(model\.layers\.\d+\.\S+?)\.lora_([AB])\.default\.weight", k2)
    if not m:
        continue
    target, ab = m.group(1), m.group(2)
    target_key = f"{target}.weight"
    if target_key not in manual:
        print(f"  缺目标键: {target_key}", flush=True)
        continue
    if ab == "A":
        continue
    a_key = k.replace("lora_B", "lora_A")
    if a_key not in adapter:
        continue
    applied += 1
    delta = (v.float() @ adapter[a_key].float()) * SCALE
    delta_mags.append(delta.abs().max().item())
    manual[target_key] = (manual[target_key].float() + delta)

print(f"adapter 目标模块数: {applied}", flush=True)
if delta_mags:
    print(f"B@A*scale 最大幅值: max={max(delta_mags):.6f} mean={sum(delta_mags)/len(delta_mags):.6f}",
          flush=True)

# 逐张量比对 sync vs manual
diffs = []
for k in sorted(sync):
    if k not in manual:
        diffs.append((k, "MISSING-IN-BASE", 0))
        continue
    d = (sync[k].float() - manual[k].float()).abs().max().item()
    if d > 1e-3:
        diffs.append((k, d, manual[k].numel()))
print(f"超差张量数: {len(diffs)}/{len(sync)}", flush=True)
for k, d, n in sorted(diffs, key=lambda x: -x[1])[:10]:
    print(f"  {k}: maxdiff={d:.6f} numel={n}", flush=True)

# 生成测试：把 manual 权重装进模型
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            device_map="cuda")
model.load_state_dict({k: v.to(torch.bfloat16) for k, v in manual.items()
                       if k in model.state_dict()}, strict=False)
model.eval()
bad_ids = [tk.convert_tokens_to_ids(t) for t in
           ("<|url|>", "<|code|>", "<|audio|>", "<|video|>", "<|quote|>",
            "<|think|>", "</think>")]
bad_ids = [t for t in bad_ids if isinstance(t, int) and t >= 0]
ROLE_RE = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)
LABELS = ESCONV_CANONICAL


def gen(messages):
    text = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    ids = tk(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=128, do_sample=False,
                         repetition_penalty=1.15, no_repeat_ngram_size=4,
                         suppress_tokens=bad_ids if bad_ids else None)
    raw = tk.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() \
        or "(keep talking)"
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = pq.read_table(ROOT / "data" / "mcppo_esconv" / "train.parquet").to_pylist()[:n]
    print(f"生成测试 {len(rows)} 种子...", flush=True)
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
    print(f"生成结果（{len(out)} 条）：乱码脚本 {garb} / CJK {cjk} / bang {bang}", flush=True)
    for r in out:
        at = [t["text"] for t in r["dialogue"] if t["role"] == "assistant"]
        print(f"  {r['seed_id']}: {r['termination']} turns={len(r['dialogue'])}",
              f"| 首句: {at[0][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
