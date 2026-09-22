"""判别实验：HF-merge 权重 + vllm 8001 + 真实 esconv 训练种子 × 两段式多轮。

与 RL rollout（轮 7）唯一差异 = 权重来源：
  - 本实验：save_merged_sft.py 的 HF merge_and_unload（8001 服务）
  - RL：verl lora.merge 路径（base + adapter，FSDP merge patch）
协议完全复刻 CstpoAgentLoop（标签 choice 约束 → "\\n" → 自由话语，
T=0.2 top_p=0.95，max_tokens 8/128，enable_thinking=False，同种子同环境）。

裁决：本实验干净 → verl merge 权重路径是 token soup 元凶；
      本实验同样乱 → 腐败在 vllm 协议层（约束/缓存/并发）。
"""
import concurrent.futures
import json
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "Cog-Sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pyarrow.parquet as pq
from openai import OpenAI
from transformers import AutoTokenizer

from cstpo.core.label_maps import ESCONV_CANONICAL
from cstpo.core.task_env import TaskEnv, prefix_turns

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="x")
MODEL = "/tmp/qwen3_esconv_sft_merged"
tk = AutoTokenizer.from_pretrained("/publicdata/model/Qwen3-8B")
LABELS = sorted(ESCONV_CANONICAL)
FALLBACK = LABELS[0]
NL = tk.encode("\n", add_special_tokens=False)
IMEND_NL = tk.encode("<|im_end|>\n", add_special_tokens=False)
# 变体开关：TEMP=0 贪心口径；REP=1.15 与 SFT 三件套对齐
TEMP = float(os.environ.get("TEMP", "0.2"))
REP = float(os.environ.get("REP", "1.0"))
OUT_NAME = os.environ.get("OUT_NAME", "discrim_vllm_merged.jsonl")


def template_ids(msgs, strip_system=False):
    text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    ids = tk.encode(text, add_special_tokens=False)
    if strip_system:  # verl remove_system_prompt 语义：去掉开头 system 块
        for i in range(len(ids) - len(IMEND_NL) + 1):
            if ids[i:i + len(IMEND_NL)] == IMEND_NL:
                ids = ids[i + len(IMEND_NL):]
                break
    return ids


def gen(prompt_ids, max_tokens, choice=None):
    extra = {"repetition_penalty": REP}
    if choice:
        extra["structured_outputs"] = {"choice": choice}
    r = client.completions.create(model=MODEL, prompt=prompt_ids,
                                  max_tokens=max_tokens, temperature=TEMP,
                                  top_p=0.95, extra_body=extra)
    return r.choices[0].text


def one(row):
    seed = json.loads(row["seed_json"])
    messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
    all_ids = template_ids(messages)
    env = TaskEnv(max_turns=12)
    cp = env.reset(seed)["checkpoint"]
    turns = prefix_turns(seed)
    termination = None
    for _ in range(12):
        # 标签段：choice 约束
        label = gen(all_ids, 8, choice=LABELS).strip()
        if label not in LABELS:
            label = FALLBACK
        all_ids += tk.encode(label, add_special_tokens=False)
        # 话语段：自由生成
        utter = gen(all_ids + NL, 128).strip() or "(keep talking)"
        all_ids += NL + tk.encode(utter, add_special_tokens=False)
        step = env.step(cp, utter)
        cp = step["checkpoint"]
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": step["user_reply"]})
        if step["terminated"]:
            termination = step["termination_reason"]
            break
        all_ids += template_ids(
            [{"role": "assistant", "content": utter},
             {"role": "user", "content": step["user_reply"]}], strip_system=True)
    return {"seed_id": row["seed_id"], "dialogue": turns, "termination": termination}


def main():
    rows = pq.read_table(ROOT / "data" / "mcppo_esconv" / "train.parquet").to_pylist()
    print(f"种子数: {len(rows)}，8 并发复刻轮 7 协议（TEMP={TEMP}, REP={REP}）...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(one, rows))
    out_path = ROOT / "logs" / OUT_NAME
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"落盘: {out_path}")

    # 内联审计（与 rollout 审计同口径）
    def garb(t):
        non = sum(1 for c in t if ord(c) > 127 and c not in "’“”—–")
        return len(t) > 10 and non / len(t) > 0.2

    stats = {"bang": 0, "garb": 0, "role_leak": 0, "loop_in": 0, "loop_x": 0, "cjk": 0}
    for r in out:
        at = [t["text"] for t in r["dialogue"] if t["role"] == "assistant"]
        for t in at:
            if "!" in t and t.count("!") / len(t) > 0.5:
                stats["bang"] += 1
                break
        for t in at:
            if garb(t):
                stats["garb"] += 1
                break
        for t in at:
            if re.match(r"^\s*(assistant|user)\s*:", t):
                stats["role_leak"] += 1
                break
        for t in at:
            sents = [s.strip() for s in re.split(r"[.!?]\s+", t) if s.strip()]
            if len(sents) >= 3 and max(sents.count(s.lower()) for s in sents) >= 3:
                stats["loop_in"] += 1
                break
        # 跨轮固定回复：同一条助手话语一字不差出现 >=3 次
        dup = [a for a, n in __import__("collections").Counter(at).items() if n >= 3]
        if dup:
            stats["loop_x"] += 1
        for t in at:
            if re.search(r"[一-鿿]", t):
                stats["cjk"] += 1
                break
    print("审计（轨迹级）:", stats, f"/ {len(out)}")
    for r in out:
        print(f"  {r['seed_id']}: {r['termination']}, turns={len(r['dialogue'])}")


if __name__ == "__main__":
    main()
