"""Cog-Sim 反应多样性探测（问题 02 迷你先导，2026-09-15）。

设置：3 任务 × 3 种子（dev 前 3，取自 90 冻结种子）× 4 风格
（vanilla/cooperative/pushy/neutral）× 5 用户模拟器 = 180 条对话：
cogsim（TaskEnv）+ 提示式四档（standard_user_sim.py 的 roleplay/persona/
persona_resist/bdi，对应 PPDPP Table 11 / ESC-Eval 角色卡 / TRIP / 问题 02
「显式 BDI 无约束」档）。
自由结束：TaskEnv(max_turns=30) 仅作安全阀（非协议轮数）；提示式
模拟器由用户方 end 标记结束。

指标（三层）：
- 结局层：终止方式分布、平均轮数（TRIP AT 口径）、judge 结局分布；
- 文本层：用户回复极性分布（flash 三分类，对齐 Value-Reinforcement 的
  worse/same/better 口径）、平均长度、distinct-1/2；
- 状态层（仅 Cog-Sim）：emotion valence/arousal 终点与轨迹均值。
多样性量化：各指标风格条件分布 vs 池化分布的 Jensen-Shannon 散度
（两模拟器同口径）；方向性 sanity：pushy 的负极性占比与早退轮数应不低于
cooperative。

用法：cd CSTPO && python -m cstpo.eval.diversity_probe --workers 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient

from cstpo.core.agent import agent_turn
from cstpo.core.cost_ledger import RecordingLLM
from cstpo.core.judge import judge_cb, judge_esconv, judge_p4g
from cstpo.core.stale_judger import check_stale, should_check
from cstpo.core.standard_user_sim import StandardUserSim
from cstpo.core.task_env import TaskEnv, prefix_turns

OUT = ROOT / "data" / "diversity_probe"
# 目录内的模块产物（非对话文件），所有加载器必须排除
ARTIFACTS = ("summary.json", "snr_analysis.json", "llm_judge_eval.json",
             "polarity_by_condition.json")
TASKS = ("esconv", "p4g", "craigslistbargain")
STYLES = ("vanilla", "cooperative", "pushy", "neutral")
SIM_VARIANTS = ("roleplay", "persona", "persona_resist", "bdi")
SAFETY_CAP = 30  # 安全阀（非协议轮数），正常应由用户方结束


def load_seeds(task: str, n: int, offset: int = 0) -> list[dict]:
    seeds = sorted((ROOT / "data" / "seeds_draft" / task).glob("*.json"))
    seeds = [p for p in seeds if p.name != "manifest.json"]
    return [json.loads(p.read_text()) for p in seeds[offset:offset + n]]


def seed_ref(seed: dict) -> dict:
    return {"seed_id": seed["seed_id"],
            "situation": seed["persona"].get("situation_en"),
            "emotion": seed["initial_emotion"]["category"]}


def run_cogsim(args) -> dict:
    task, seed, style, use_judger, llm = args
    env = TaskEnv(llm=llm, max_turns=SAFETY_CAP)
    cp = env.reset(seed)["checkpoint"]
    pre = prefix_turns(seed)  # 从截止前缀继续（actor 可见历史）
    turns = list(pre)
    states, r = [], None
    n = 0
    reason = None
    while not (r and r["terminated"]):
        if use_judger and should_check(n):
            j = check_stale(LLMClient(), turns, task)
            if j["stale"] and j["final_line"]:
                turns.append({"role": "user", "text": j["final_line"]})
                reason = "judger_stale_end"
                break
        utter = agent_turn(llm, task, style, turns, seed)
        r = env.step(cp, utter)
        cp = r["checkpoint"]
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": r["user_reply"]})
        n += 1
        sim = r.get("sim")
        st = getattr(sim, "state", None)
        if st is not None and hasattr(st, "emotion"):
            states.append({"valence": st.emotion.valence,
                           "arousal": st.emotion.arousal,
                           "category": st.emotion.category})
    return {"task": task, "seed": seed_ref(seed), "style": style, "sim": "cogsim",
            "termination": reason or (r["termination_reason"] if r else "?"),
            "n_turns": n, "prefix_n": len(pre), "turns": turns, "states": states}


def run_standard(args) -> dict:
    task, seed, style, variant, use_judger, llm = args
    sim = StandardUserSim(llm, variant=variant)
    sim.reset(seed)
    pre = prefix_turns(seed)  # agent 与模拟器同从前缀继续
    turns = list(pre)
    r, n = None, 0
    reason = None
    while True:
        if use_judger and should_check(n):
            j = check_stale(LLMClient(), turns, task)
            if j["stale"] and j["final_line"]:
                turns.append({"role": "user", "text": j["final_line"]})
                reason = "judger_stale_end"
                break
        utter = agent_turn(llm, task, style, turns, seed)
        r = sim.step({}, utter)
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": r["user_reply"]})
        n += 1
        if r["terminated"] or n >= SAFETY_CAP:
            break
    if reason:
        term = reason
    elif r and r["terminated"]:
        term = r["termination_reason"]
    else:
        term = "safety_cap"
    return {"task": task, "seed": seed_ref(seed), "style": style,
            "sim": f"std_{variant}",
            "termination": term, "n_turns": n, "prefix_n": len(pre),
            "turns": turns, "states": []}


def judge_dialogue(args):
    d, llm = args
    if d["task"] == "esconv":
        v = judge_esconv(llm, d["turns"], d["seed"]["situation"],
                         d["seed"]["emotion"])
        return {"key": f"E{v.E}A{v.A}"}
    if d["task"] == "p4g":
        v = judge_p4g(llm, d["turns"])
        return {"key": f"commit={v.commitment}"}
    v = judge_cb(llm, d["turns"])
    return {"key": f"deal={v.deal}"}


SENT_SYSTEM = ('Classify the sentiment polarity of this message. '
               'Respond ONLY with a JSON object: {"polarity": "negative"}')


def sent_polarity(args):
    text, llm = args
    out = llm.chat_json([{"role": "system", "content": SENT_SYSTEM},
                         {"role": "user", "content": text}], max_tok=50)
    p = (out.get("polarity") or "neutral").strip().lower()
    return p if p in ("negative", "neutral", "positive") else "neutral"


def js_div(p: dict, q: dict) -> float:
    """两经验分布（键已对齐）的 Jensen-Shannon 散度（natural log）。"""
    keys = sorted(set(p) | set(q))
    pp = [p.get(k, 0.0) for k in keys]
    qq = [q.get(k, 0.0) for k in keys]
    m = [(a + b) / 2 for a, b in zip(pp, qq)]
    kl = lambda a, b: sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)
    return (kl(pp, m) + kl(qq, m)) / 2


def norm_counter(c: Counter) -> dict:
    tot = sum(c.values()) or 1
    return {k: v / tot for k, v in c.items()}


def summarize(results: list[dict]) -> dict:
    by_sim = {}
    for sim in sorted({r["sim"] for r in results}):
        rows = [r for r in results if r["sim"] == sim]
        per_style = {}
        for style in STYLES:
            rs = [r for r in rows if r["style"] == style]
            if not rs:
                continue
            user_replies = [t["text"] for r in rs
                            for t in r["turns"][r.get("prefix_n", 0):]
                            if t["role"] == "user" and t["text"]]
            words = [w for x in user_replies for w in x.split()]
            per_style[style] = {
                "n": len(rs),
                "termination": dict(Counter(r["termination"] for r in rs)),
                "mean_turns": sum(r["n_turns"] for r in rs) / len(rs),
                "n_sent": len(user_replies),
                "mean_len": sum(len(x.split()) for x in user_replies) / max(len(user_replies), 1),
                "distinct1": len(set(words)) / max(len(words), 1),
                "distinct2": len(set(zip(words, words[1:]))) / max(len(words) - 1, 1),
            }
        # JS：风格条件分布 vs 池化（结局/终止/极性三张表）
        js = {}
        for metric in ("judge", "termination", "polarity"):
            pool = Counter()
            style_dists = {}
            for r in rows:
                pool[r[metric]] += 1
            pool_n = norm_counter(pool)
            for style, rs2 in per_style.items():
                c = Counter(r[metric] for r in rows if r["style"] == style)
                style_dists[style] = norm_counter(c)
            js[metric] = {s: js_div(d, pool_n) for s, d in style_dists.items()}
            js[f"{metric}_mean"] = sum(js[metric].values()) / max(len(js[metric]), 1)
        by_sim[sim] = {"per_style": per_style, "js": js,
                       "judge_dist": norm_counter(Counter(r["judge"] for r in rows)),
                       "termination_dist": norm_counter(Counter(r["termination"] for r in rows)),
                       "polarity_dist": norm_counter(Counter(r["polarity"] for r in rows))}
    # 状态层（仅 cogsim）：valence 终点均值按风格
    states_by_style = {}
    for r in results:
        if r["sim"] == "cogsim" and r["states"]:
            states_by_style.setdefault(r["style"], []).append(
                r["states"][-1]["valence"])
    by_sim["cogsim"]["valence_end_mean"] = {
        s: sum(v) / len(v) for s, v in states_by_style.items() if v}
    return by_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--pilot", action="store_true",
                    help="先导：每任务 1 种子 × 2 风格（cooperative/pushy）")
    ap.add_argument("--judger", action="store_true",
                    help="15 轮后冗余 judger（默认关；judger_probe 已验证）")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="种子起始偏移（分批扩种用）")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="每任务种子数（默认 3；扩种时配合 --seed-offset）")
    ap.add_argument("--summary-only", action="store_true",
                    help="只对已有产物重算合并 summary（不生成对话）")
    args = ap.parse_args()
    if args.summary_only:
        results = []
        for fp in OUT.rglob("*.json"):
            if fp.name in ARTIFACTS:
                continue
            results.append(json.loads(fp.read_text()))
        print(f"summary-only：加载 {len(results)} 条已有对话")
        summary = summarize(results)
        (OUT / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
        print(f"[合并] 结局/JS/状态层汇总已写入 {OUT / 'summary.json'}")
        for sim, s in summary.items():
            print(f"[{sim}] judge_JS={s['js']['judge_mean']:.4f} "
                  f"termination_JS={s['js']['termination_mean']:.4f} "
                  f"polarity_JS={s['js']['polarity_mean']:.4f} "
                  f"termination={s['termination_dist']}")
        return
    styles = ("cooperative", "pushy") if args.pilot else STYLES
    n_seeds = 1 if args.pilot else args.n_seeds

    # 跳过已落盘完成的对话（崩溃续跑；同名文件即同 (sim,task,style,seed)）
    done = set()
    for fp in OUT.rglob("*.json"):
        if fp.name in ARTIFACTS:
            continue
        done.add((fp.parent.parent.name, fp.parent.name, fp.stem))

    jobs = []
    for task in TASKS:
        for seed in load_seeds(task, n_seeds, args.seed_offset):
            for style in styles:
                key_c = ("cogsim", task, f"{style}_{seed['seed_id']}")
                if key_c not in done:
                    jobs.append((run_cogsim,
                                 (task, seed, style, args.judger, RecordingLLM()),
                                 "cogsim"))
                for variant in SIM_VARIANTS:
                    key_v = (f"std_{variant}", task, f"{style}_{seed['seed_id']}")
                    if key_v not in done:
                        jobs.append((run_standard,
                                     (task, seed, style, variant, args.judger,
                                      RecordingLLM()),
                                     f"std_{variant}"))
    if not jobs:
        print("无待生成对话（全部已落盘）", flush=True)
    else:
        print(f"启动 {len(jobs)} 条自由结束对话（workers={args.workers}）...")
    def out_path(r):
        d = OUT / r["sim"] / r["task"]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{r['style']}_{r['seed']['seed_id']}.json"

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(j[0], j[1]) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            try:
                r = fut.result()
            except Exception as e:  # 单条失败不炸批次：跳过，重跑时 skip-existing 补
                print(f"  [跳过] 对话生成异常: {type(e).__name__}: {e}", flush=True)
                continue
            results.append(r)
            out_path(r).write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  已落盘 {i + 1}/{len(jobs)}", flush=True)
    # 统一从磁盘装载全部对话（含既有批次），只对缺 judge 的做 judge/极性
    all_results = [json.loads(fp.read_text()) for fp in OUT.rglob("*.json")
                   if fp.name not in ARTIFACTS]
    pending = [r for r in all_results if "judge" not in r]
    sims = sorted({r["sim"] for r in all_results})
    print("终止分布:", {s: dict(Counter(r["termination"] for r in all_results
                                       if r["sim"] == s)) for s in sims},
          flush=True)

    # judge + 极性
    print(f"judge 与极性判定（逐条回写）：{len(pending)} 条待判 / "
          f"共 {len(all_results)} 条", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        jfuts = {ex.submit(judge_dialogue, (r, LLMClient())): r for r in pending}
        for k, fut in enumerate(as_completed(jfuts)):
            r = jfuts[fut]
            try:
                r["judge"] = fut.result()["key"]
            except Exception as e:
                print(f"  [跳过] judge 异常: {type(e).__name__}", flush=True)
                r["judge"] = "judge_failed"
            out_path(r).write_text(
                json.dumps(r, ensure_ascii=False, indent=1) + "\n")
            if (k + 1) % 60 == 0:
                print(f"  judge {k + 1}/{len(pending)}", flush=True)
    def dialogue_polarity(args):
        r, llm = args
        pols = [sent_polarity((t["text"], llm))
                for t in r["turns"][r.get("prefix_n", 0):]
                if t["role"] == "user" and t["text"]]
        return pols

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pfuts = {ex.submit(dialogue_polarity, (r, LLMClient())): r for r in pending}
        for k, fut in enumerate(as_completed(pfuts)):
            r = pfuts[fut]
            try:
                pols = fut.result()
            except Exception as e:
                print(f"  [跳过] 极性异常: {type(e).__name__}", flush=True)
                continue
            r["polarity"] = (Counter(pols).most_common(1)[0][0]
                             if pols else "neutral")
            r["polarity_dist"] = dict(Counter(pols))
            out_path(r).write_text(
                json.dumps(r, ensure_ascii=False, indent=1) + "\n")
            if (k + 1) % 60 == 0:
                print(f"  极性 {k + 1}/{len(pending)}", flush=True)

    summary = summarize(all_results)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print("\n===== 结局层 =====")
    for sim, s in summary.items():
        print(f"[{sim}] 结局分布: {s['judge_dist']} | 终止: {s['termination_dist']}")
        print(f"[{sim}] 极性分布: {s['polarity_dist']}")
        for style, m in s["per_style"].items():
            print(f"  {style:12s} n={m['n']} 轮均={m['mean_turns']:.1f} "
                  f"distinct1={m['distinct1']:.3f} 终止={m['termination']}")
    print("\n===== 多样性（风格 vs 池化的 JS） =====")
    for sim, s in summary.items():
        print(f"[{sim}] judge_JS={s['js']['judge_mean']:.4f} "
              f"termination_JS={s['js']['termination_mean']:.4f} "
              f"polarity_JS={s['js']['polarity_mean']:.4f}")
    if "valence_end_mean" in summary["cogsim"]:
        print("\n===== 状态层（Cog-Sim 终点 valence 均值） =====")
        print(summary["cogsim"]["valence_end_mean"])
    total_turns = sum(r["n_turns"] for r in all_results)
    print(f"\n产物: {OUT}/（{len(all_results)} 条对话，共 {total_turns} 轮 + summary.json）")


if __name__ == "__main__":
    main()
