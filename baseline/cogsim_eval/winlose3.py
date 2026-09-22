"""三维修度 win/lose 盲评（基于剂量-反应实验轨迹）。

对每个 (任务, 种子, 剂量档)，cogsim 的回复与每档对照的回复两两配对
（两条措辞各配一次），flash 当评委盲评三个维度（左右顺序随机化）：
- consistency：与历史/画像/当前语境的一致性；
- appropriateness：反应分寸与 agent 该句剂量的匹配度；
- naturalness：语言自然度（像不像真人随手打的字）。

聚合：按维度 × 对手 × 任务，W/T/L + (W−L)/n，种子聚类 bootstrap CI。
产物：winlose3.json + 打印表。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient  # noqa: E402

from baseline.runner import load_api_key  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "cogsim_eval"
OPPONENTS = ["std_persona", "std_persona_resist", "std_bdi"]
DIMS = ["consistency", "appropriateness", "naturalness"]

SYSTEM = """You are comparing two user replies in a dialogue. Both replies are
written by DIFFERENT simulated users responding to the SAME situation and the
SAME agent utterance. Compare them on three dimensions:

1. consistency: which reply fits better with the user's earlier words,
   profile, and situation (no contradiction with history or persona)?
2. appropriateness: which reply's response level better matches the weight of
   the agent's utterance (e.g., a vague utterance should not trigger a big
   change of attitude; a strong, substantive offer or evidence should get a
   substantive response)?
3. naturalness: which reply sounds more like something a real person would
   actually type in a chat (casual wording, natural phrasing, human
   imperfection) rather than mechanical or formulaic?

For each dimension answer "A" if reply 1 is better, "B" if reply 2 is
better, or "tie" if equally good. Do not judge which reply is more polite.
Return exactly one JSON object:
{"consistency": "A", "appropriateness": "B", "naturalness": "tie",
 "reason": "one short sentence"}"""


def load_cells() -> list[dict]:
    """合并 dose_probe（证据梯度）与 dose_cb（cb 价格梯度）为统一格子。

    返回 [{task, seed_id, level, prefix, utterances: [..], replies:
    {sim: [reply, ...]}}]，只保留 cogsim 与三档对照齐全的格。"""
    d_ev = json.loads((OUT_DIR / "dose_probe.json").read_text())
    d_cb = json.loads((OUT_DIR / "dose_cb.json").read_text())
    seeds_cache = {}

    def seed_json(task, sid):
        key = (task, sid)
        if key not in seeds_cache:
            p = ROOT / "data" / "seeds_draft" / task / f"{sid}.json"
            seeds_cache[key] = json.loads(p.read_text())
        return seeds_cache[key]

    def prefix_of(task, sid):
        from cstpo.core.task_env import prefix_turns
        return prefix_turns(seed_json(task, sid))

    cells, seen = [], set()
    for task in ("esconv", "p4g"):
        inter = d_ev["interventions"]
        for key, recs in d_ev["arms"].items():
            t, sid, sim, lv = key.split("|")
            cell_key = (task, sid, lv)
            if cell_key in seen:
                continue
            seen.add(cell_key)
            utt = inter.get(f"{task}|{sid}", {}).get(lv, [])
            replies = {}
            ok = True
            for s in ["cogsim"] + OPPONENTS:
                rs = d_ev["arms"].get(f"{task}|{sid}|{s}|{lv}", [])
                rs = [r["reply"] for r in rs if r.get("reply")]
                if not rs:
                    ok = False
                replies[s] = rs
            if ok and utt:
                cells.append({"task": task, "seed_id": sid, "level": lv,
                              "prefix": prefix_of(task, sid),
                              "utterances": utt, "replies": replies,
                              "asking": None})
    inter_cb = d_cb["interventions"]
    for key, recs in d_cb["arms"].items():
        sid, sim, lv = key.split("|")
        cell_key = ("craigslistbargain", sid, lv)
        if cell_key in seen:
            continue
        seen.add(cell_key)
        utt = inter_cb.get(sid, {}).get(lv, [])
        replies = {}
        ok = True
        for s in ["cogsim"] + OPPONENTS:
            rs = d_cb["arms"].get(f"{sid}|{s}|{lv}", [])
            rs = [r["reply"] for r in rs if r.get("reply")]
            if not rs:
                ok = False
            replies[s] = rs
        if ok and utt:
            seed = seed_json("craigslistbargain", sid)
            asking = (seed["persona"].get("item_facts") or {}).get("price") \
                or seed["actor_task_view"].get("item", {}).get("Price")
            cells.append({"task": "craigslistbargain", "seed_id": sid,
                          "level": lv,
                          "prefix": prefix_of("craigslistbargain", sid),
                          "utterances": utt, "replies": replies,
                          "asking": asking})
    return cells


def judge_pair(llm: LLMClient, cell: dict, opp: str, i: int, rng) -> dict:
    """配对评一条：cogsim 回复 vs opp 回复（左右随机化），返回
    {dim: +1 cogsim 胜 / -1 输 / 0 平}。"""
    reply_c = cell["replies"]["cogsim"][i]
    reply_o = cell["replies"][opp][i]
    swap = rng.random() < 0.5
    r1, r2 = (reply_o, reply_c) if swap else (reply_c, reply_o)
    ctx = "\n".join(f"{t['role']}: {t['text']}" for t in cell["prefix"])
    task_hint = {"esconv": "emotional support conversation",
                 "p4g": "charity donation persuasion",
                 "craigslistbargain": "price negotiation"}[cell["task"]]
    ask = ""
    if cell["asking"]:
        ask = f"\nListing asking price: {cell['asking']}"
    user = (f"Task: {task_hint}{ask}\n\nConversation so far:\n{ctx}\n\n"
            f"Agent's utterance: {cell['utterances'][i]}\n\n"
            f"Reply 1: {r1}\n\nReply 2: {r2}")
    out = llm.chat_json(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user}], max_tok=200)
    res = {}
    for dim in DIMS:
        v = (out.get(dim) or "tie").strip().upper()
        if v not in ("A", "B"):
            v = "tie"
        better = r1 if v == "A" else (r2 if v == "B" else None)
        if better is None:
            res[dim] = 0
        elif better == reply_c:
            res[dim] = 1
        else:
            res[dim] = -1
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=240)
    args = ap.parse_args()
    load_api_key()
    cells = load_cells()
    # 每任务限种子数（取前 n 个种子）
    per_task = {}
    for c in cells:
        per_task.setdefault(c["task"], {})[c["seed_id"]] = None
    keep_seeds = {t: sorted(ids)[: args.n_seeds] for t, ids in per_task.items()}
    cells = [c for c in cells if c["seed_id"] in keep_seeds[c["task"]]]
    rng = random.Random(20260922)
    jobs = []
    for c in cells:
        n = min(len(c["replies"]["cogsim"]), 2)
        for opp in OPPONENTS:
            for i in range(min(n, len(c["replies"][opp]))):
                jobs.append((c, opp, i))
    print(f"[winlose3] 格 {len(cells)}，配对 {len(jobs)}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for c, opp, i in jobs:
            futs[pool.submit(judge_pair, LLMClient(), c, opp, i, rng)] \
                = (c, opp, i)
        done = 0
        for fut in as_completed(futs):
            c, opp, i = futs[fut]
            try:
                res = fut.result()
                res.update({"task": c["task"], "seed_id": c["seed_id"],
                            "level": c["level"], "opp": opp})
                results.append(res)
            except Exception as e:
                print(f"[fail] {c['task']}|{c['seed_id']}|{opp}: {e}",
                      flush=True)
            done += 1
            if done % 300 == 0:
                print(f"[winlose3] {done}/{len(jobs)}", flush=True)
    (OUT_DIR / "winlose3.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1))
    # 聚合
    boot_rng = np.random.default_rng(20260922)
    print(f"\n{'维度':<16}{'对手':<20}{'任务':<16}{'W':>4}{'T':>4}{'L':>4}"
          f"{'W-L':>8}{'CI':>18}")
    for dim in DIMS:
        for opp in OPPONENTS:
            for task in ("esconv", "p4g", "craigslistbargain"):
                by_seed = {}
                for r in results:
                    if r["opp"] == opp and r["task"] == task:
                        by_seed.setdefault(r["seed_id"], []).append(r[dim])
                seed_means = [np.mean(v) for v in by_seed.values()]
                if not seed_means:
                    continue
                v = np.asarray(seed_means)
                n = len(v)
                boots = np.array([v[boot_rng.integers(0, n, n)].mean()
                                  for _ in range(2000)])
                lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
                w = sum(1 for x in seed_means if x > 0)
                t = sum(1 for x in seed_means if x == 0)
                l = sum(1 for x in seed_means if x < 0)
                print(f"{dim:<16}{opp:<20}{task:<16}{w:>4}{t:>4}{l:>4}"
                      f"{v.mean():>+8.2f}"
                      f"{f'[{lo:+.2f},{hi:+.2f}]':>18}")


if __name__ == "__main__":
    main()
