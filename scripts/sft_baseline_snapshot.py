"""SFT 基线的采样分布快照（MC_PPO_EVAL_PLAN §6.2 前置诊断）。

裁决"训练第一步后 reward 崩"的两种假设：
  A. RL 确实毁模型（更新微小但行为崩）——若基线多次采样稳定 ≥0.2，
     则 RL 有问题；
  B. reward 稀疏 + 采样噪声（epoch0 的 0.17 是噪声高点）——若基线
     多次采样均值 ≈0.1 且方差大，则训练"崩"是测量假象，问题在
     reward 分辨率。

口径与 RL rollout 一致：温度 0.5 / top_p 0.95 / 每轮 max 128 tokens /
12 轮 / 首行标签剥离 / 终局 judge 3 次聚合。
输出：data/sft_baseline_snapshot/{task}.jsonl + 汇总打印。
"""
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from cstpo.core.agent import SYSTEM_BUILDERS  # noqa: E402
from cstpo.core.judge import judge_with_aggregation  # noqa: E402
from cstpo.core.task_env import TaskEnv, prefix_turns  # noqa: E402
from cstpo.rl.verl_cstpo_agent_loop import strip_strategy_label  # noqa: E402
from simulator.llm import LLMClient  # noqa: E402

TASK = "esconv"
# 权重来源参数化：v3 基线快照用 EVAL_BASE（merged 权重目录，无 adapter）
ADAPTER = Path(os.environ["ADAPTER"]) if os.environ.get("ADAPTER") else None
VAL_SEEDS = sorted((ROOT / "data/seeds_draft/esconv").glob("*.json"))[20:40]
N_REPEATS = int(os.environ.get("N_REPEATS", "2"))
MAX_TURNS = 12
TEMPERATURE = float(os.environ.get("TEMP", "0.5"))  # TEMP=0 → greedy + 三件套


def main():
    tk = AutoTokenizer.from_pretrained("/publicdata/model/Qwen3-8B")
    if ADAPTER:
        model = AutoModelForCausalLM.from_pretrained(
            "/publicdata/model/Qwen3-8B", torch_dtype=torch.bfloat16,
            attn_implementation="sdpa")
        model = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False)
        model = model.merge_and_unload().cuda().eval()
    else:
        base = os.environ.get("EVAL_BASE", "/publicdata/model/CSTPO_v3/esconv")
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model = model.cuda().eval()
    print("模型加载完成（SFT adapter 已合并）", flush=True)

    out_dir = ROOT / "data" / "sft_baseline_snapshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"t{TEMPERATURE:g}" if TEMPERATURE > 0 else "greedy"
    tag = os.environ.get("OUT_TAG", tag)  # 版本化：v3/v4 快照不覆盖历史基线
    out_path = out_dir / f"{TASK}_{tag}.jsonl"
    results = []
    with open(out_path, "w") as f:
        for fp in VAL_SEEDS:
            seed = json.loads(fp.read_text())
            for rep in range(N_REPEATS):
                turns = prefix_turns(seed)
                env = TaskEnv(max_turns=MAX_TURNS, debug=False)
                cp = env.reset(seed)["checkpoint"]
                n_turns = 0
                for _ in range(MAX_TURNS):
                    msgs = [{"role": "system", "content": SYSTEM_BUILDERS[TASK](seed)}]
                    for h in turns:
                        msgs.append({"role": "assistant" if h["role"] == "assistant"
                                     else "user", "content": h["text"]})
                    text = tk.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
                    ids = tk(text, return_tensors="pt").to(model.device)
                    gen_kwargs = dict(max_new_tokens=128)
                    if TEMPERATURE > 0:
                        gen_kwargs.update(do_sample=True, temperature=TEMPERATURE,
                                          top_p=0.95)
                    else:
                        # greedy 口径 = A/B 验证三件套
                        bad_ids = []
                        for tok in ("<|url|>", "<|code|>", "<|audio|>", "<|video|>",
                                    "<|quote|>", "<think>", "</think>"):
                            t = tk.convert_tokens_to_ids(tok)
                            if isinstance(t, int) and t >= 0:
                                bad_ids.append(t)
                        gen_kwargs.update(do_sample=False, repetition_penalty=1.15,
                                          no_repeat_ngram_size=4,
                                          suppress_tokens=bad_ids or None)
                    out = model.generate(**ids, **gen_kwargs)
                    gen = out[0][ids["input_ids"].shape[1]:]
                    raw = tk.decode(gen, skip_special_tokens=True)
                    utter = strip_strategy_label(raw, TASK)
                    r = env.step(cp, utter)
                    cp = r["checkpoint"]
                    turns.append({"role": "assistant", "text": utter})
                    turns.append({"role": "user", "text": r["user_reply"]})
                    n_turns += 1
                    if r["terminated"]:
                        break
                situation = seed.get("persona", {}).get("situation_en")
                emotion = seed.get("initial_emotion", {}).get("category")
                agg = judge_with_aggregation(LLMClient(), TASK, turns,
                                             situation, emotion)
                rec = {"seed_id": seed["seed_id"], "rep": rep,
                       "n_turns": n_turns, "E": agg["E"], "A": agg["A"],
                       "G": (agg["E"] + agg["A"]) / 8.0}
                results.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                gs = [x["G"] for x in results]
                print(f"{seed['seed_id']}#{rep}: G={rec['G']:.3f} "
                      f"(E={rec['E']} A={rec['A']}, {n_turns}轮) | "
                      f"累计 mean={sum(gs)/len(gs):.3f} n={len(gs)}",
                      flush=True)

    gs = [x["G"] for x in results]
    print(f"\n=== SFT 基线（温度 0.5）val 20 种子 × {N_REPEATS} 次 ===")
    print(f"G mean={sum(gs)/len(gs):.4f}  min={min(gs):.4f}  max={max(gs):.4f}")
    nonzero = [g for g in gs if g > 0]
    print(f"非零占比 {len(nonzero)}/{len(gs)}，非零均值 {sum(nonzero)/len(nonzero):.4f}")


if __name__ == "__main__":
    main()
