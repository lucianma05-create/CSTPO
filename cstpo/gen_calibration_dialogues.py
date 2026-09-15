"""P1 校准对话生成（04§19 修订版：共 60 条）与 judge smoke。

- Agent：默认 LLM Agent（`cstpo/agent.py`，论文香草提示 + cooperative/pushy/
  neutral 风格画像轮流使用制造边界样本；`--scripted` 回退话术池版），
  用冻结种子在真实模拟器上跑到终止（user_ended / time_limit / 8 轮上限）。
- 分配：ESConv 20 开发 + 20 留出；P4G/CB 各 5 开发 + 5 留出。
- judge smoke：留出对话用 judge 独立评分 3 次（flash），报告重复一致性与
  分数分布；开发对话评分 1 次供提示修订。

用法：cd CSTPO && python -m cstpo.gen_calibration_dialogues --workers 30
      （--judge-only 只对已有对话重跑 smoke，judge 提示迭代用）
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient

from cstpo.cost_ledger import RecordingLLM, attribute_calls
from cstpo.judge import judge_cb, judge_esconv, judge_p4g, reward_for
from cstpo.task_env import TaskEnv, prefix_turns

OUT = ROOT / "data" / "judge_calibration"
MAX_TURNS = 8

# 脚本化 Agent 画像：每画像一组循环使用的话语（覆盖成功/失败/边界）
AGENT_PROFILES = {
    "esconv": {
        "cooperative": [
            "That sounds really hard. What part worries you the most?",
            "It's okay to feel that way. You've been carrying a lot.",
            "What is one small thing you could try this week?",
            "That sounds like a reasonable first step.",
            "I hear you. Would talking to your advisor help at all?",
            "Maybe you could write down the smallest possible step.",
            "You've managed hard things before. What helped then?",
            "I'm here to listen - tell me more about it.",
        ],
        "pushy": [
            "You really should see a counselor about this.",
            "Just send the message today - you'll feel better.",
            "Stop overthinking and just do it.",
            "You need to take action now, not later.",
            "Why haven't you tried a counselor yet?",
            "This is simple: message your advisor today.",
        ],
        "neutral": [
            "I see. Anything else on your mind?",
            "Okay. How are you feeling right now?",
            "That's understandable.",
            "Hmm, I see what you mean.",
        ],
    },
    "p4g": {
        "cooperative": [
            "Could you tell me what you think about donating to help children?",
            "A small donation of 10 dollars would help a child in need.",
            "Every dollar goes to programs that support children's education.",
            "Would you consider donating 10 dollars today?",
            "Thank you for listening - even 5 dollars makes a difference.",
            "No problem, take your time to decide.",
        ],
        "pushy": [
            "You should donate right now - children are waiting.",
            "10 dollars is nothing for you. Just donate.",
            "Everyone else is donating. Don't miss out.",
            "If you don't donate, who will help these children?",
            "Come on, donate 10 dollars now.",
        ],
        "neutral": [
            "I see. That's good to know.",
            "Sure, let me know if you have questions.",
            "Okay, thank you for your time.",
        ],
    },
    "craigslistbargain": {
        "cooperative": [
            "What's the lowest price you could accept?",
            "Could you go a bit lower? I'm on a budget.",
            "That might work. Would you include the accessories?",
            "I can offer a bit more if you can meet me halfway.",
            "Let's agree on that price then.",
        ],
        "pushy": [
            "My offer is final - take it or leave it.",
            "That price is too high. Lower it.",
            "I'm not paying a dollar more.",
            "Deal or no deal?",
            "Fine, forget it. Goodbye.",
        ],
        "neutral": [
            "Okay, I see. Let me think about it.",
            "Hmm, that's interesting. Tell me more.",
            "Thanks, that's helpful to know.",
        ],
    },
}


def dialogue_seed_ref(seed: dict) -> dict:
    return {"seed_id": seed["seed_id"],
            "situation": seed["persona"].get("situation_en"),
            "emotion": seed["initial_emotion"]["category"]}


def run_dialogue(args, scripted: bool = False) -> dict:
    """跑一条对话到终止。默认 LLM Agent；scripted=True 用话术池。"""
    task, seed, profile, did, llm = args
    env = TaskEnv(llm=llm, max_turns=MAX_TURNS)
    out = env.reset(seed)
    cp = out["checkpoint"]
    turns = prefix_turns(seed)  # 从截止前缀继续（actor 可见历史）
    actor_costs = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    r = None
    if not scripted:
        from cstpo.agent import agent_turn
        from simulator.llm import LLMClient as _LC
        actor_llm = _LC()
    pool = AGENT_PROFILES[task][profile]
    i = 0
    while not (r and r["terminated"]):
        if r:
            cp = r["checkpoint"]
        if scripted:
            utter = pool[i % len(pool)]
        else:
            utter = agent_turn(actor_llm, task, profile, turns, seed)
            actor_costs["calls"] += 1
            actor_costs["prompt_tokens"] += actor_llm.prompt_tokens
            actor_costs["completion_tokens"] += actor_llm.completion_tokens
        i += 1
        r = env.step(cp, utter)
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": r["user_reply"]})
    n_pre_ass = len([t for t in turns[:len(prefix_turns(seed))]
                     if t["role"] == "assistant"])
    return {"task": task, "dialogue_id": did, "profile": profile,
            "turns": turns, "termination": r["termination_reason"],
            "n_turns": len([t for t in turns if t["role"] == "assistant"]) - n_pre_ass,
            "prefix_n": len(prefix_turns(seed)),
            "seed": dialogue_seed_ref(seed), "costs": r["costs"],
            "actor_costs": actor_costs}


def judge_dialogue(args):
    did, turns, task, situation, emotion, llm = args
    if task == "esconv":
        v = judge_esconv(llm, turns, situation, emotion)
        return did, {"E": v.E, "A": v.A,
                     "evidence_sufficient": v.evidence_sufficient,
                     "reward": reward_for(task, v)}
    if task == "p4g":
        v = judge_p4g(llm, turns)
        return did, {"commitment": v.commitment, "withdrawn": v.withdrawn,
                     "conditional": v.conditional, "amount": v.amount,
                     "reward": reward_for(task, v)}
    v = judge_cb(llm, turns)
    seed = None  # CB 收益需双方 target；烟测阶段仅报 deal/price
    return did, {"deal": v.deal, "final_price": v.final_price,
                 "parse_error": v.parse_error, "currency": v.currency}


def run_smoke(plans, results, workers, term=None, out=OUT):
    """留出 3 次独立评分 + 开发 1 次；报告重复一致性与分数分布。"""
    print("\njudge smoke（留出 3 次独立评分）...")
    jobs = []
    for p, res in zip(plans, results):
        rep = 3 if p[4] == "heldout" else 1
        for _ in range(rep):
            jobs.append((res["dialogue_id"], res["turns"], p[0],
                         res["seed"].get("situation"),
                         res["seed"].get("emotion"), LLMClient()))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        jres = list(ex.map(judge_dialogue, jobs))
    by_did = {}
    for did, v in jres:
        by_did.setdefault(did, []).append(v)
    # 一致性：留出对话 3 次评分的一致率
    for task in ("esconv", "p4g", "craigslistbargain"):
        rows = [(did, vs) for did, vs in by_did.items()
                if did.startswith(task) and "heldout" in did]
        exact = 0
        for did, vs in rows:
            if task == "esconv":
                pairs = {(v["E"], v["A"]) for v in vs}
                exact += len(pairs) == 1
            else:
                k = "deal" if task == "craigslistbargain" else "commitment"
                pairs = {v[k] for v in vs}
                exact += len(pairs) == 1
        print(f"  {task}: 留出 {len(rows)} 条，3 次评分完全一致 {exact} 条")
    # 分布
    dist = {}
    for did, vs in by_did.items():
        task = did.split("_")[0]
        v = vs[0]
        if task == "esconv":
            key = f"E{v['E']}A{v['A']}"
        elif task == "craigslistbargain":
            key = f"deal={v['deal']}"
        else:
            key = f"commit={v['commitment']}"
        dist[key] = dist.get(key, 0) + 1
    print(f"  分数分布（首次评分）: {dist}")
    # 写入 smoke 结果
    (out / "smoke_summary.json").write_text(
        json.dumps({"termination_dist": term, "by_dialogue": by_did},
                   ensure_ascii=False, indent=1) + "\n")
    print(f"\n产物: {out}/{{dev,heldout}}/ 与 smoke_summary.json")
    return by_did


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--pilot", action="store_true",
                    help="先导模式：每任务 2 条（1 开发 + 1 留出），打印完整对话")
    ap.add_argument("--scripted", action="store_true",
                    help="回退到脚本化话术池 Agent（默认 LLM Agent）")
    ap.add_argument("--judge-only", action="store_true",
                    help="不生成对话，对 data/judge_calibration 下已有对话重跑 smoke")
    args = ap.parse_args()

    if args.judge_only:
        results, plans = [], []
        for split in ("dev", "heldout"):
            for task in ("esconv", "p4g", "craigslistbargain"):
                for fp in sorted((OUT / split / task).glob("*.json")):
                    res = json.loads(fp.read_text())
                    results.append(res)
                    plans.append((res["task"], None, res.get("profile"),
                                  res["dialogue_id"], split))
        print(f"judge-only：加载 {len(results)} 条已有对话")
        run_smoke(plans, results, args.workers)
        return

    if args.pilot:
        alloc = {"esconv": (1, 1), "p4g": (1, 1), "craigslistbargain": (1, 1)}
    else:
        alloc = {"esconv": (20, 20), "p4g": (5, 5), "craigslistbargain": (5, 5)}
    plans = []
    for task, (n_dev, n_ho) in alloc.items():
        seeds = sorted((ROOT / "data" / "seeds_draft" / task).glob("*.json"))
        seeds = [p for p in seeds if p.name != "manifest.json"]
        dev = [json.loads(p.read_text()) for p in seeds[:n_dev]]
        ho = [json.loads(p.read_text()) for p in seeds[n_dev:n_dev + n_ho]]
        for i, (seed, split) in enumerate(
                [(s, "dev") for s in dev] + [(s, "heldout") for s in ho]):
            profiles = ("cooperative", "pushy", "neutral")
            plans.append((task, seed, profiles[i % 3], f"{task}_{split}_{i:02d}",
                          split))

    llms = [RecordingLLM() for _ in plans]
    print(f"生成 {len(plans)} 条对话（workers={args.workers}，"
          f"Agent={'scripted' if args.scripted else 'llm'}）...", flush=True)
    base = OUT / "pilot" if args.pilot else OUT

    def out_path(split, task, did):
        d = base / split / task
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{did}.json"

    results = []
    jobs = [(p[0], p[1], p[2], p[3], llm) for p, llm in zip(plans, llms)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(lambda a: run_dialogue(a, scripted=args.scripted), j)
                for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            results.append(res)
            out_path("heldout" if res["dialogue_id"].split("_")[1] == "heldout"
                     else "dev", res["task"], res["dialogue_id"]).write_text(
                json.dumps(res, ensure_ascii=False, indent=2) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  已落盘 {i + 1}/{len(jobs)}", flush=True)
    term = {}
    for res in results:
        term[res["termination"]] = term.get(res["termination"], 0) + 1
    print(f"终止分布: {term}")
    if args.pilot:
        for res in results:
            print(f"\n===== {res['dialogue_id']}（{res['profile']}，"
                  f"{res['n_turns']} 轮，{res['termination']}）=====")
            for t in res["turns"]:
                print(f"[{t['role']}] {t['text'][:120]}")
    total_calls = sum(c.calls for c in llms)
    total_tok = sum(c.prompt_tokens + c.completion_tokens for c in llms)
    print(f"生成成本: {total_calls} 调用 / {total_tok} tokens")

    # judge smoke：留出 3 次 + 开发 1 次
    run_smoke(plans, results, args.workers, term=term, out=base)


if __name__ == "__main__":
    main()
