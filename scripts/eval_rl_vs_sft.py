"""RL vs SFT 正式评估（MC_PPO_EVAL_PLAN §3 主指标）。

RL 策略来源：正式轮最后一次权重同步的 dump（logs/sync_dump/adapter.pt +
sync_merged.pt 目录由 DUMP_DIR 指定）——手工 HF 合并（base + B@A*alpha/r），
与 compare_sync_dump.py 同口径。评估口径与 sft_baseline_snapshot.py 完全一致：
greedy + 三件套、12 轮、首行标签剥离、judge 3 次聚合、esconv_21-40 × 2 次。

并行化：对话全量并行（max_workers 线程，本地生成持锁串行，模拟器/judge
网络调用全并发）。输出：data/rl_eval/rl_esconv_greedy.jsonl + 与基线
（esconv_greedy.jsonl）按种子配对的 ΔG、配对 t 检验、提前终止率、标签分布。
"""
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from cstpo.core.agent import SYSTEM_BUILDERS  # noqa: E402
from cstpo.core.judge import judge_with_aggregation  # noqa: E402
from cstpo.core.label_maps import ESCONV_CANONICAL  # noqa: E402
from cstpo.core.task_env import TaskEnv, prefix_turns  # noqa: E402
from cstpo.rl.verl_cstpo_agent_loop import strip_role_prefix, strip_strategy_label  # noqa: E402
from simulator.llm import LLMClient  # noqa: E402

TASK = "esconv"
BASE = os.environ.get("EVAL_BASE", "/publicdata/model/CSTPO_v3/esconv")
DUMP_DIR = Path(os.environ.get("DUMP_DIR", "logs/sync_dump"))
N_REPEATS = int(os.environ.get("N_REPEATS", "2"))
N_WORKERS = int(os.environ.get("N_WORKERS", "8"))
MAX_TURNS = 12
RANK, ALPHA = 64, 16
SCALE = ALPHA / RANK

gen_lock = threading.Lock()
write_lock = threading.Lock()


def load_rl_model(device):
    """base + 最终同步 dump 的 LoRA 手工合并（与 compare_sync_dump 同逻辑）。"""
    adapter = torch.load(DUMP_DIR / "adapter.pt", map_location="cpu", weights_only=False)
    base_sd = load_file(BASE + "/model.safetensors")
    manual = {k: v.clone() for k, v in base_sd.items()}

    def _norm(k):
        return k.replace("_fsdp_wrapped_module.", "")

    n_applied = 0
    for k, v in adapter.items():
        k2 = _norm(k)
        m = re.match(r"base_model\.model\.(model\.layers\.\d+\.\S+?)\.lora_([AB])\.default\.weight", k2)
        if not m:
            continue
        target, ab = m.group(1), m.group(2)
        if ab == "A":
            continue
        a_key = k.replace("lora_B", "lora_A")
        if a_key not in adapter:
            continue
        target_key = f"{target}.weight"
        if target_key not in manual:
            continue
        n_applied += 1
        manual[target_key] = manual[target_key].float() + (v.float() @ adapter[a_key].float()) * SCALE
    print(f"adapter 合并目标模块 {n_applied}，dtype bf16", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                device_map=device)
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in manual.items()
                           if k in model.state_dict()}, strict=False)
    return model.eval()


def run_one(entry, rep, tk, model, bad_ids, bad_words):
    seed = entry["seed"]
    turns = prefix_turns(seed)
    env = TaskEnv(max_turns=MAX_TURNS, debug=False)
    cp = env.reset(seed)["checkpoint"]
    n_turns = 0
    first_lines = []
    for _ in range(MAX_TURNS):
        msgs = [{"role": "system", "content": SYSTEM_BUILDERS[TASK](seed)}]
        for h in turns:
            msgs.append({"role": "assistant" if h["role"] == "assistant"
                         else "user", "content": h["text"]})
        text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        ids = tk(text, return_tensors="pt").to(model.device)
        with gen_lock:
            out = model.generate(**ids, max_new_tokens=128, do_sample=False,
                                 repetition_penalty=1.15, no_repeat_ngram_size=4,
                                 suppress_tokens=bad_ids or None,
                                 bad_words_ids=bad_words)
        gen = out[0][ids["input_ids"].shape[1]:]
        raw = tk.decode(gen, skip_special_tokens=True)
        first_lines.append(raw.strip().split("\n", 1)[0].strip())
        utter = strip_role_prefix(strip_strategy_label(raw, TASK))
        r = env.step(cp, utter)
        cp = r["checkpoint"]
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": r["user_reply"]})
        n_turns += 1
        if r["terminated"]:
            break
    situation = seed.get("persona", {}).get("situation_en")
    emotion = seed.get("initial_emotion", {}).get("category")
    agg = judge_with_aggregation(LLMClient(), TASK, turns, situation, emotion)
    return {"seed_id": seed["seed_id"], "rep": rep, "n_turns": n_turns,
            "E": agg["E"], "A": agg["A"], "G": (agg["E"] + agg["A"]) / 8.0,
            "first_lines": first_lines, "dialogue": turns}


def main():
    tk = AutoTokenizer.from_pretrained("/publicdata/model/Qwen3-8B")
    bad_ids = [tk.convert_tokens_to_ids(t) for t in
               ("<|url|>", "<|code|>", "<|audio|>", "<|video|>", "<|quote|>",
                "<|think|>", "</think>")]
    bad_ids = [t for t in bad_ids if isinstance(t, int) and t >= 0]
    bad_words = [tk.encode(w, add_special_tokens=False)
                 for w in ("URL", " Url", " url")]
    bad_words = [w for w in bad_words if w]
    model = load_rl_model("cuda")
    print("RL 模型加载完成", flush=True)

    # 冻结评估协议：种子必须来自冻结 parquet 的 seed_json，不能 glob
    # seeds_draft（扩种合并后文件名字典序错位，且种子文件被 refix 更新）。
    # EVAL_PARQUET 可指定最终测试集（data/mcppo_esconv/test.parquet）。
    import pyarrow.parquet as pq
    parquet_path = ROOT / os.environ.get(
        "EVAL_PARQUET", "data/mcppo_esconv/val.parquet")
    eval_table = pq.read_table(parquet_path)
    eval_rows = eval_table.to_pylist()
    val_seeds = [{"fp": None, "seed": json.loads(r["seed_json"]), "id": r["seed_id"]}
                 for r in eval_rows]
    out_dir = ROOT / "data" / "rl_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = os.environ.get("EVAL_OUT", "rl_esconv_greedy.jsonl")
    out_path = out_dir / out_name
    tasks = [(entry, rep) for entry in val_seeds for rep in range(N_REPEATS)]
    print(f"{len(tasks)} 条对话全量并行（{N_WORKERS} workers）...", flush=True)

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(run_one, entry, rep, tk, model, bad_ids, bad_words): (entry["id"], rep)
                for entry, rep in tasks}
        for fut in futs:
            rec = fut.result()
            results.append(rec)
            done += 1
            gs = [x["G"] for x in results]
            print(f"[{done}/{len(tasks)}] {rec['seed_id']}#{rec['rep']}: G={rec['G']:.3f} "
                  f"(E={rec['E']} A={rec['A']}, {rec['n_turns']}轮) | "
                  f"累计 mean={sum(gs)/len(gs):.3f}", flush=True)
    with write_lock, open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 标签统计（greedy 无约束首行）
    labels_ok = labels_total = 0
    label_dist = {}
    for r in results:
        for first in r["first_lines"]:
            labels_total += 1
            if first in ESCONV_CANONICAL:
                labels_ok += 1
                label_dist[first] = label_dist.get(first, 0) + 1

    # 与基线配对比较
    base_file = os.environ.get("BASELINE_FILE", "esconv_greedy_v3.jsonl")
    base_recs = [json.loads(l) for l in open(ROOT / "data" / "sft_baseline_snapshot"
                                             / base_file)]
    by_seed = {}
    for rec in base_recs:
        by_seed.setdefault(rec["seed_id"], []).append(rec["G"])
    base_mean = {s: sum(g) / len(g) for s, g in by_seed.items()}
    rl_mean = {}
    for rec in results:
        rl_mean.setdefault(rec["seed_id"], []).append(rec["G"])
    rl_mean = {s: sum(g) / len(g) for s, g in rl_mean.items()}
    seeds = sorted(set(base_mean) & set(rl_mean))
    diffs = [rl_mean[s] - base_mean[s] for s in seeds]
    n = len(diffs)
    if n == 0:
        g_rl = sum(r["G"] for r in results) / len(results)
        early = sum(1 for r in results if r["n_turns"] < 4) / len(results)
        print("\n=== RL 评估（无配对基线，仅汇总）===")
        print(f"G_rl={g_rl:.4f}  标签合法率 {labels_ok}/{labels_total}  提前终止率 {early:.1%}")
        print("标签分布（greedy 无约束）:", dict(sorted(label_dist.items(),
              key=lambda x: -x[1])[:8]))
        return
    import statistics
    dG = sum(diffs) / n
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    t = dG / (sd / (n ** 0.5)) if sd > 0 else float("inf")
    try:
        from scipy import stats as sp
        p = sp.ttest_rel([rl_mean[s] for s in seeds], [base_mean[s] for s in seeds]).pvalue
    except ImportError:
        p = float("nan")
    early = sum(1 for r in results if r["n_turns"] < 4) / len(results)
    g_rl = sum(r["G"] for r in results) / len(results)
    g_base = sum(base_mean.values()) / len(base_mean)
    print("\n=== RL vs SFT（esconv val 20 种子 × 2 次，greedy+三件套）===")
    print(f"G_rl={g_rl:.4f}  G_sft={g_base:.4f}  ΔG={dG:+.4f}  配对t={t:.3f}  p={p:.4f}")
    print(f"标签合法率 {labels_ok}/{labels_total}  提前终止率 {early:.1%}")
    print("标签分布（greedy 无约束）:", dict(sorted(label_dist.items(),
          key=lambda x: -x[1])[:8]))
    print(f"有效判据：ΔG>0 且 p<0.1；有意义判据：ΔG>+0.03")


if __name__ == "__main__":
    main()
