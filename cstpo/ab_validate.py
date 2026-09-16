"""SFT 后 A/B 验证（SFT_SPEC §6）：校准留出 30 条上对比 SFT 模型 vs 香草提示。

双方在相同种子/前缀/模拟器（Cog-Sim）上各跑一遍对话（自由结束 + judger），
judge_with_aggregation（3 次聚合）评分：
- ESConv：E/A 均值对比；
- P4G：commitment 率对比；
- CB：deal 率 + 平均训练回报（raw_SL clip，需种子双方 target）。

SFT 模型侧：加载 merge 后的完整权重（checkpoints/sft/<task>/），用与香草
提示完全相同的消息格式（system + 历史）生成下一句（自回归，max 150 tokens）。

用法：cd CSTPO && CUDA_VISIBLE_DEVICES=0 python -m cstpo.ab_validate --task esconv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient

from cstpo.agent import agent_turn
from cstpo.cost_ledger import RecordingLLM
from cstpo.judge import judge_with_aggregation
from cstpo.stale_judger import check_stale, should_check
from cstpo.task_env import TaskEnv, prefix_turns

CALIB = ROOT / "data" / "judge_calibration" / "heldout"
CKPT = Path("/publicdata/model/CSTPO")
SAFETY_CAP = 30


def load_seeds(task, split="heldout"):
    """校准留出对话所用的种子（按 dialogue 文件的 seed 引用取）。"""
    out = {}
    for fp in (CALIB / task).glob("*.json"):
        d = json.loads(fp.read_text())
        sid = d["seed"]["seed_id"]
        if sid not in out:
            seed = json.loads((ROOT / "data" / "seeds_draft" / task /
                               f"{sid}.json").read_text())
            out[sid] = seed
    return list(out.values())


def run_side(task, seeds, model_pipe, use_sft):
    """跑一侧（SFT 模型或香草提示）全部留出对话。model_pipe=None → 香草提示。"""
    results = {}
    for seed in seeds:
        llm = RecordingLLM()
        env = TaskEnv(llm=llm, max_turns=SAFETY_CAP)
        cp = env.reset(seed)["checkpoint"]
        turns = prefix_turns(seed)
        r, n = None, 0
        reason = None
        while not (r and r["terminated"]):
            if should_check(n):
                j = check_stale(LLMClient(), turns, task)
                if j["stale"] and j["final_line"]:
                    turns.append({"role": "user", "text": j["final_line"]})
                    reason = "judger_stale_end"
                    break
            if use_sft:
                utter = model_turn(model_pipe, task, turns, seed)
            else:
                utter = agent_turn(LLMClient(), task, "vanilla", turns, seed)
            r = env.step(cp, utter)
            cp = r["checkpoint"]
            turns.append({"role": "assistant", "text": utter})
            turns.append({"role": "user", "text": r["user_reply"]})
            n += 1
        results[seed["seed_id"]] = {
            "turns": turns, "termination": reason or r["termination_reason"],
            "seed": seed}
    return results


def model_turn(pipe, task, turns, seed):
    """SFT 模型生成：与推理设置一致的消息格式，无标签前缀（话语 SFT 口径
    下标签由 RL 阶段 trie 采样；SFT 验证阶段直接生成话语）。"""
    from cstpo.agent import SYSTEM_BUILDERS
    system = SYSTEM_BUILDERS[task](seed)
    msgs = [{"role": "system", "content": system}]
    for h in turns:
        msgs.append({"role": "assistant" if h["role"] == "assistant" else "user",
                     "content": h["text"]})
    out = pipe(msgs, max_new_tokens=150)
    return (out or "").strip() or "(keep talking)"


def score_side(task, results):
    out = {}
    for sid, d in results.items():
        situation = d["seed"].get("persona", {}).get("situation_en")
        emotion = d["seed"].get("initial_emotion", {}).get("category")
        agg = judge_with_aggregation(LLMClient(), task, d["turns"],
                                     situation, emotion)
        out[sid] = agg
    return out


def summarize(task, sft_scores, base_scores):
    if task == "esconv":
        for name, sc in (("SFT", sft_scores), ("香草", base_scores)):
            e = sum(v["E"] for v in sc.values()) / len(sc)
            a = sum(v["A"] for v in sc.values()) / len(sc)
            g = (e + a) / 8
            print(f"  {name}: E={e:.2f} A={a:.2f} G={g:.3f}")
        return
    if task == "p4g":
        for name, sc in (("SFT", sft_scores), ("香草", base_scores)):
            c = sum(bool(v["commitment"]) for v in sc.values())
            print(f"  {name}: commitment {c}/{len(sc)}")
        return
    # CB：deal 率 + 训练回报（种子双方 target）
    for name, sc in (("SFT", sft_scores), ("香草", base_scores)):
        deals = sum(bool(v["deal"]) for v in sc.values())
        print(f"  {name}: deal {deals}/{len(sc)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["esconv", "p4g", "craigslistbargain"])
    args = ap.parse_args()

    # UUID 选空闲最大的 A100（与 train_sft 同口径，避开序映射陷阱）
    import os, subprocess
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free,uuid",
             "--format=csv,noheader"], capture_output=True, text=True).stdout
        best, best_free = None, -1
        for line in out.strip().splitlines():
            name, free, uuid = [x.strip() for x in line.split(",")]
            if "A100" in name and int(free.split()[0]) > best_free:
                best, best_free = uuid, int(free.split()[0])
        if best is None:
            raise RuntimeError("无可用 A100")
        os.environ["CUDA_VISIBLE_DEVICES"] = best
        print(f"[自动选卡] {best[:8]}…", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tk = AutoTokenizer.from_pretrained(str(CKPT / args.task))
    model = AutoModelForCausalLM.from_pretrained(
        str(CKPT / args.task), torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # 禁掉 Qwen3 的 masked 特殊 token（<|url|> 等）：训练数据中的 URL 被
    # 转成这些 token，模型学会吐 "URL URL..." 的退化循环
    bad_ids = []
    for tok in ("<|url|>", "<|code|>", "<|audio|>", "<|video|>", "<|quote|>"):
        t = tk.convert_tokens_to_ids(tok)
        if isinstance(t, int) and t >= 0:
            bad_ids.append(t)
    if bad_ids:
        print(f"[生成配置] suppress {len(bad_ids)} 个特殊 token", flush=True)

    def pipe(msgs, max_new_tokens=150):
        text = tk.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
        ids = tk(text, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             repetition_penalty=1.15,
                             suppress_tokens=bad_ids if bad_ids else None)
        gen = out[0][ids["input_ids"].shape[1]:]
        return tk.decode(gen, skip_special_tokens=True)

    seeds = load_seeds(args.task)
    print(f"[{args.task}] 留出种子 {len(seeds)} 个，跑双侧对话...", flush=True)
    sft_res = run_side(args.task, seeds, pipe, use_sft=True)
    base_res = run_side(args.task, seeds, None, use_sft=False)
    print("judge 评分（3 次聚合）...", flush=True)
    sft_scores = score_side(args.task, sft_res)
    base_scores = score_side(args.task, base_res)
    out = ROOT / "data" / "sft" / "ab_results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.task}.json").write_text(json.dumps(
        {"sft": sft_scores, "vanilla": base_scores,
         "sft_termination": {k: v["termination"] for k, v in sft_res.items()},
         "vanilla_termination": {k: v["termination"] for k, v in base_res.items()}},
        ensure_ascii=False, indent=1) + "\n")
    print(f"\n[{args.task}] A/B 结果:")
    summarize(args.task, sft_scores, base_scores)


if __name__ == "__main__":
    main()
