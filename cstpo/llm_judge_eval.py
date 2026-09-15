"""LLM judge 评估（flash）：漂移/僵化 + 五档行为相似度。

1. 漂移/僵化（P2 盲评 rubric 的 LLM 版，合理性评估口径）：
   对每条模拟对话，判用户行为的两类问题——
   drift：无对话证据支撑的态度/情绪/立场改变（0=无 1=轻微 2=明确）；
   rigidity：面对清晰相关的新证据/实质让步/可行步骤未做合理反应（0/1/2）。
2. 相似度对比：同一 (任务,种子,风格) 下五档模拟续写两两配对（10 对），
   flash 判两条续写中用户行为的相似度（0-1）。回答"哪个提示式基线
   最接近 Cog-Sim"。

只评用户侧行为；judge 用 flash（deepseek-flash），结果标注为合理性评估，
正式真实性结论需人工盲评（04§19）。

用法：cd CSTPO && python -m cstpo.llm_judge_eval --workers 200
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient, StructuredCallError

OUT = ROOT / "data" / "diversity_probe"
# 目录内的模块产物（非对话文件），加载器必须排除
ARTIFACTS = ("summary.json", "snr_analysis.json", "llm_judge_eval.json",
             "polarity_by_condition.json")
SIMS = ("cogsim", "std_roleplay", "std_persona", "std_persona_resist", "std_bdi")

DRIFT_SYSTEM = """You are a dialogue evaluator. The user in this conversation is
simulated. Judge two things about the USER's behavior only (not the assistant's):

1) DRIFT - unwarranted change: did the user change attitude, emotion, or
position WITHOUT supporting evidence in the dialogue? A change is warranted
when the assistant gave new information, a genuine concession, or a concrete
actionable step that explains it. 0 = no unwarranted change; 1 = minor or
unclear; 2 = clear unwarranted change.

2) RIGIDITY - warranted-evidence blindness: when the assistant DID provide
clear relevant new evidence, a genuine concession, or a concrete actionable
step, did the user fail to respond in any reasonable way? 0 = responded
appropriately; 1 = partially or unclear; 2 = clearly rigid (ignored/deflected
without addressing it).

Quote the user utterances (with turn indices) that support each judgment.
Return exactly one JSON object:
{"drift": 0, "rigidity": 0, "drift_quotes": [{"turn": 3, "quote": "..."}],
 "rigidity_quotes": [{"turn": 5, "quote": "..."}]}"""


def judge_drift(args):
    d, llm = args
    dialogue = "\n".join(f"[{i}] {t['role']}: {t['text']}"
                         for i, t in enumerate(d["turns"]))
    try:
        out = llm.chat_json(
            [{"role": "system", "content": DRIFT_SYSTEM},
             {"role": "user", "content": f"Task type: {d['task']}\n\nDialogue:\n{dialogue}"}],
            max_tok=600)
    except StructuredCallError:
        return {"drift": None, "rigidity": None, "error": "parse_failed"}
    return {"drift": int(out.get("drift", -1)) if out.get("drift") is not None else None,
            "rigidity": int(out.get("rigidity", -1)) if out.get("rigidity") is not None else None,
            "drift_quotes": out.get("drift_quotes", []),
            "rigidity_quotes": out.get("rigidity_quotes", [])}


SIM_SYSTEM = """You are a dialogue evaluator. Below are TWO continuations of
the SAME conversation (same starting prefix and situation, two different
simulated users). Rate how similar the USER's behavior is between the two
continuations: conversational style, stance/resistance level, emotional
reactions, and what they eventually do. Focus ONLY on the user messages.
Return exactly one JSON object: {"similarity": 0.75} where 0 = completely
different behavior and 1 = essentially the same behavior."""


def judge_similarity(args):
    (key, turns_a, turns_b, task), llm = args
    def render(turns):
        return "\n".join(f"{t['role']}: {t['text']}" for t in turns)
    try:
        out = llm.chat_json(
            [{"role": "system", "content": SIM_SYSTEM},
             {"role": "user", "content": (
                 f"Task type: {task}\n\nContinuation A:\n{render(turns_a)}\n\n"
                 f"Continuation B:\n{render(turns_b)}")}],
            max_tok=100)
    except StructuredCallError:
        return key, None
    try:
        return key, float(out.get("similarity", -1))
    except (TypeError, ValueError):
        return key, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=200)
    args = ap.parse_args()

    files = [p for p in OUT.rglob("*.json") if p.name not in ARTIFACTS]
    dialogues = [json.loads(p.read_text()) for p in files]
    print(f"加载 {len(dialogues)} 条对话", flush=True)

    # 1) 漂移/僵化
    print("漂移/僵化判定（flash）...", flush=True)
    dres = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_drift, (d, LLMClient())): i
                for i, d in enumerate(dialogues)}
        for k, fut in enumerate(as_completed(futs)):
            i = futs[fut]
            d = dialogues[i]
            key = f"{d['sim']}/{d['task']}/{d['style']}_{d['seed']['seed_id']}"
            try:
                dres[key] = fut.result()
            except Exception as e:  # 单条失败不炸批次
                dres[key] = {"drift": None, "rigidity": None,
                             "error": f"{type(e).__name__}"}
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(dialogues)}", flush=True)

    # 2) 五档相似度（同一任务/种子/风格下两两配对）
    print("五档相似度判定（flash）...", flush=True)
    groups = {}
    for d in dialogues:
        groups.setdefault((d["task"], d["seed"]["seed_id"], d["style"]), {})[d["sim"]] = d
    pairs = []
    for (task, sid, style), by_sim in groups.items():
        sims = [s for s in SIMS if s in by_sim]
        for a_i in range(len(sims)):
            for b_i in range(a_i + 1, len(sims)):
                sa, sb = sims[a_i], sims[b_i]
                pairs.append(((task, sid, style, sa, sb),
                              by_sim[sa]["turns"][by_sim[sa].get("prefix_n", 0):],
                              by_sim[sb]["turns"][by_sim[sb].get("prefix_n", 0):],
                              task))
    print(f"  {len(pairs)} 对", flush=True)
    sres = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judge_similarity, (p, LLMClient())) for p in pairs]
        for k, fut in enumerate(as_completed(futs)):
            try:
                key, v = fut.result()
            except Exception:  # 单条失败不炸批次
                continue
            sres[key] = v
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(pairs)}", flush=True)

    # 汇总
    agg_drift = {}
    for (sim, style), vals in _group_scores(dialogues, dres).items():
        agg_drift[(sim, style)] = vals
    sim_sim = {s: [] for s in SIMS}
    for (task, sid, style, sa, sb), v in sres.items():
        if v is not None:
            sim_sim[sa].append(v)
            sim_sim[sb].append(v)
    pair_mean = {}
    for i in range(len(SIMS)):
        for j in range(i + 1, len(SIMS)):
            sa, sb = SIMS[i], SIMS[j]
            vals = [v for (t_, s_, st_, a, b), v in sres.items()
                    if ((a == sa and b == sb) or (a == sb and b == sa))
                    and v is not None]
            pair_mean[f"{sa}~{sb}"] = (sum(vals) / len(vals)) if vals else None

    out = {
        "n_dialogues": len(dialogues),
        "n_pairs": len(pairs),
        "per_dialogue": dres,
        "pairwise_similarity": {str(k): v for k, v in sres.items()},
        "agg": {
            "drift_by_sim_style": {f"{k[0]}|{k[1]}": v for k, v in agg_drift.items()},
            "mean_similarity_by_sim": {s: (sum(v) / len(v)) if v else None
                                       for s, v in sim_sim.items()},
            "pair_mean": pair_mean,
        },
    }
    (OUT / "llm_judge_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n")

    print("\n===== 漂移/僵化（0-2，越低越好） =====")
    for sim in SIMS:
        rows = [(k[1], v) for k, v in agg_drift.items() if k[0] == sim]
        for style, v in sorted(rows):
            print(f"  {sim:18s} {style:11s} drift={v['drift']:.2f} "
                  f"rigidity={v['rigidity']:.2f} n={v['n']}")
    print("\n===== 五档行为相似度（均值） =====")
    for s, v in sim_sim.items():
        print(f"  {s:18s} mean_sim={sum(v)/len(v):.3f} (n={len(v)})")
    print("\n===== 两两相似度 =====")
    for k, v in pair_mean.items():
        print(f"  {k:22s} {v:.3f}" if v is not None else f"  {k:22s} None")
    print(f"\n产物: {OUT / 'llm_judge_eval.json'}")


def _group_scores(dialogues, dres):
    by_key = {}
    for d in dialogues:
        key = f"{d['sim']}/{d['task']}/{d['style']}_{d['seed']['seed_id']}"
        r = dres.get(key, {})
        if r.get("drift") is None or r.get("rigidity") is None:
            continue
        by_key.setdefault((d["sim"], d["style"]), []).append((r["drift"], r["rigidity"]))
    out = {}
    for k, vals in by_key.items():
        out[k] = {"drift": sum(a for a, b in vals) / len(vals),
                  "rigidity": sum(b for a, b in vals) / len(vals),
                  "n": len(vals)}
    return out


if __name__ == "__main__":
    main()
