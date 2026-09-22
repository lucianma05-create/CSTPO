"""标签探针（v2 重训验收第 1 步）：greedy 无约束首行分布。

判据（预注册）：首行合法率 ≥50% 且最大标签占比 ≤80% = PASS。
v1 对照：合法率 0/251（完全不吐标签）；RL rollout 对照：99.4% Information。

用法：python scripts/probe_labels_v2.py [--task esconv] [--n-seeds 5] [--turns 6]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from cstpo.core.agent import SYSTEM_BUILDERS  # noqa: E402
from cstpo.core.label_maps import CB_CANONICAL, ESCONV_CANONICAL, P4G_CANONICAL  # noqa: E402
from cstpo.core.task_env import prefix_turns  # noqa: E402

CKPT = Path(__import__("os").environ.get("AB_CKPT", "/publicdata/model/CSTPO_v2"))
LABEL_SETS = {"esconv": ESCONV_CANONICAL, "p4g": P4G_CANONICAL,
              "craigslistbargain": CB_CANONICAL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="esconv")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--turns", type=int, default=6)
    args = ap.parse_args()
    task = args.task
    labels = LABEL_SETS[task]
    tk = AutoTokenizer.from_pretrained("/publicdata/model/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        str(CKPT / task), torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    bad = [tk.convert_tokens_to_ids(t) for t in
           ("<|url|>", "<|code|>", "<|audio|>", "<|video|>", "<|quote|>",
            "<|think|>", "</think>")]
    bad = [t for t in bad if isinstance(t, int) and t >= 0]
    bad_words = [tk.encode(w, add_special_tokens=False)
                 for w in ("URL", " Url", " url")]
    bad_words = [w for w in bad_words if w]

    seed_prefix = {"esconv": "esconv_", "p4g": "p4g_",
                   "craigslistbargain": "cb_"}[task]
    seeds = [p for p in sorted((ROOT / "data" / "seeds_draft" / task).glob("*.json"))
             if p.stem.startswith(seed_prefix)][:args.n_seeds]
    dist = {}
    total = legal = 0
    for fp in seeds:
        seed = json.loads(fp.read_text())
        turns = prefix_turns(seed)
        for _ in range(args.turns):
            msgs = [{"role": "system", "content": SYSTEM_BUILDERS[task](seed)}]
            for h in turns:
                msgs.append({"role": "assistant" if h["role"] == "assistant"
                             else "user", "content": h["text"]})
            text = tk.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
            ids = tk(text, return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=128, do_sample=False,
                                 repetition_penalty=1.15, no_repeat_ngram_size=4,
                                 suppress_tokens=bad or None,
                                 bad_words_ids=bad_words)
            raw = tk.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True)
            first = raw.strip().split("\n", 1)[0].strip()
            total += 1
            if first in labels:
                legal += 1
                dist[first] = dist.get(first, 0) + 1
            turns.append({"role": "assistant", "text": raw})
            turns.append({"role": "user", "text": "(continue)"})
    legal_rate = legal / max(total, 1)
    top = max(dist.values()) / max(legal, 1) if dist else 0
    verdict = "PASS" if legal_rate >= 0.5 and top <= 0.8 else "FAIL"
    print(f"[{task}] 首行合法率 {legal}/{total} ({legal_rate:.1%})  "
          f"最大标签占比 {top:.1%} → {verdict}")
    print(f"[{task}] 标签分布:", dict(sorted(dist.items(), key=lambda x: -x[1])[:10]))


if __name__ == "__main__":
    main()
