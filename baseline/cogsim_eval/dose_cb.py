"""cb 剂量-反应 v2：价格让步梯度（替代信息证据梯度）。

谈判中改变卖家态度的杠杆是价格与让步，剂量按买方报价的让步幅度四档：
- L0 纯询问（不提价格，只问信息）；
- L1 轻微还价（略低于要价，象征性试探）；
- L2 实质还价（明显低于要价的合理报价）；
- L3 大让步（接近要价的高报价 + 即刻付款承诺）。

措辞按种子真实价格结构生成（要价 = persona.item_facts.price；
买家目标 = actor_task_view.target），每档 2 措辞，同一快照分支 ×
四档模拟器。

反应 = 卖方让步评分（0-3：0 无让步/拒绝/不回应价格；1 轻微让步；
2 实质让步；3 接受或接近接受），替代态度分。Cog-Sim 额外记录状态版
反应（BDI 总 |delta|）。

预注册判据：①cogsim L0 反应不高于最强对照 + 0.2（不轻易被说服）；
②cogsim 区分度指数（L3−L0）显著高于最强对照（不固化）。
"""
from __future__ import annotations

import argparse
import copy
import json
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
SIMS = ["cogsim", "std_persona", "std_persona_resist", "std_bdi"]
LEVELS = ["L0", "L1", "L2", "L3"]

_INTERVENTION_SYSTEM = (
    "You are writing what a BUYER says to a SELLER in a price negotiation, "
    "at the same moment, differing only in concession strength. Return "
    "exactly one JSON object with keys L0,L1,L2,L3; each maps to a list of "
    "exactly two short utterances. Concession = offering a HIGHER price: "
    "the three priced levels must have strictly increasing prices "
    "(L1's price < L2's price < L3's price).\n"
    "L0: pure questions about the item (condition, details) with NO price "
    "mentioned at all;\n"
    "L1: slight concession — a price about 4-5% below the asking price "
    "(the lowest of the three priced offers);\n"
    "L2: moderate concession — a price about 1-2% below the asking price "
    "(clearly higher than L1's price);\n"
    "L3: strong concession — a price AT the asking price or slightly above "
    "it, PLUS an immediate-payment commitment (cash today).")

_CONCESSION_SYSTEM = (
    "Read the negotiation so far (a buyer and a seller haggling over a "
    "price), then the buyer's latest utterance, then the seller's reply. "
    "The listing/asking price is given to you as a reference point. Judge "
    "how much the SELLER conceded on price in this reply. "
    "0 = no concession (refuses, ignores the price, or holds the asking "
    "price); 1 = slight concession (moves a little below the asking price, "
    "or signals flexibility); 2 = substantive concession (offers a clearly "
    "lower price); 3 = acceptance or near-acceptance — this includes "
    "expressing surprise or verifying the buyer's seriousness AND THEN "
    "moving into the closing flow (agreeing, accepting, scheduling pickup "
    "or payment, or asking how to complete the deal). A seller who only "
    "hesitates without moving toward closing stays at 0. Score "
    "conservatively. Return exactly one JSON "
    'object: {"concession": 0, "reason": "one short sentence"}.')


def gen_interventions(llm: LLMClient, seed: dict) -> dict:
    from cstpo.core.task_env import prefix_turns

    item = seed["persona"].get("item_facts", {}) or {}
    asking = item.get("price") or seed["actor_task_view"].get("item", {}).get(
        "Price")
    target = seed["actor_task_view"].get("target")
    user = (f"Item: {item.get('title')}, listed price {asking} (use this as "
            f"the asking-price anchor unless the SELLER states a different "
            f"asking price in the conversation below — then use the seller's "
            f"stated price). Buyer's target price: {target}.\nConversation "
            f"so far:\n"
            + "\n".join(f"{t['role']}: {t['text']}"
                        for t in prefix_turns(seed)))
    out = llm.chat_json(
        [{"role": "system", "content": _INTERVENTION_SYSTEM},
         {"role": "user", "content": user}], max_tok=600)
    return {lv: [str(x).strip() for x in (out.get(lv) or [])][:2]
            for lv in LEVELS}


def make_job(seed: dict, sim: str, utter: str) -> dict:
    from cstpo.core.task_env import TaskEnv
    from cstpo.core.standard_user_sim import StandardUserSim

    for attempt in range(3):
        try:
            llm = LLMClient()
            if sim == "cogsim":
                env = TaskEnv(llm=llm, max_turns=30, debug=False)
                reset = env.reset(seed)
                r = env.step(copy.deepcopy(reset["checkpoint"]), utter)
                st = r["sim"].state
                base = {b["id"]: b["strength"] for b in (
                    reset["x"]["bdi"]["beliefs"] + reset["x"]["bdi"]["desires"]
                    + reset["x"]["bdi"]["intentions"])}
                post = {b.id: b.strength for b in (
                    list(st.beliefs) + list(st.desires)
                    + list(st.intentions))}
                state_delta = sum(abs(post.get(k, 0) - v)
                                  for k, v in base.items())
                return {"reply": r["user_reply"],
                        "state_delta": round(state_delta, 3)}
            usim = StandardUserSim(LLMClient(),
                                   variant=sim.replace("std_", ""))
            reset = usim.reset(seed)
            r = usim.step(copy.deepcopy(reset["checkpoint"]), utter)
            return {"reply": r["user_reply"]}
        except Exception:
            if attempt == 2:
                return {"reply": "", "error": "sim_fail"}
    return {"reply": "", "error": "sim_fail"}


def concession_score(llm: LLMClient, seed: dict, utter: str,
                     reply: str) -> int:
    from cstpo.core.task_env import prefix_turns

    item = seed["persona"].get("item_facts", {}) or {}
    asking = item.get("price") or seed["actor_task_view"].get("item", {}).get(
        "Price")
    ctx = "\n".join(f"{t['role']}: {t['text']}" for t in prefix_turns(seed))
    out = llm.chat_json(
        [{"role": "system", "content": _CONCESSION_SYSTEM},
         {"role": "user",
          "content": f"Listing asking price: {asking}\n\nConversation so "
                     f"far:\n{ctx}\n\nBuyer's utterance: {utter}\n\n"
                     f"Seller's reply: {reply}"}],
        max_tok=150)
    try:
        return max(0, min(3, int(out.get("concession", -1))))
    except (TypeError, ValueError):
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=120)
    args = ap.parse_args()
    load_api_key()
    files = sorted((ROOT / "data" / "seeds_draft" / "craigslistbargain")
                   .glob("*.json"))
    files = [f for f in files if f.name != "manifest.json"]
    import re as _re

    seeds = []
    for f in files:
        s = json.loads(f.read_text())
        persona_txt = " ".join(x.get("fact_en", "") for x in
                               s["persona"].get("facts", [])).lower()
        pre_txt = " ".join(t.get("text_en", "") for t in
                           s["context"].get("prefix", [])).lower()
        item_txt = (s["persona"].get("item_facts") or {}).get("title",
                                                              "").lower()
        all_txt = persona_txt + " " + pre_txt + " " + item_txt
        # 过滤：租赁/按月、前缀已成交或已议定价格——剂量语义错位
        if any(w in all_txt for w in ("rent", "month", "lease")):
            continue
        if _re.search(r"(shook|agreed on|we have a deal|sold|deal!)",
                      pre_txt):
            continue
        seeds.append(s)
        if len(seeds) >= args.n_seeds:
            break
    print(f"[cb-dose] 过滤后可用种子 {len(seeds)}/需 {args.n_seeds}",
          flush=True)
    print(f"[cb-dose] 种子 {len(seeds)}，workers {args.workers}", flush=True)
    # 1) 干预生成
    inter_map = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(gen_interventions, LLMClient(), s): s
                for s in seeds}
        for fut in as_completed(futs):
            s = futs[fut]
            inter_map[s["seed_id"]] = fut.result()
    # 2) 模拟续演
    arms = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for seed in seeds:
            for sim in SIMS:
                for lv in LEVELS:
                    for utter in inter_map[seed["seed_id"]][lv]:
                        futs[pool.submit(make_job, seed, sim, utter)] \
                            = (seed["seed_id"], sim, lv)
        done = 0
        for fut in as_completed(futs):
            sid, sim, lv = futs[fut]
            try:
                arms.setdefault(f"{sid}|{sim}|{lv}", []).append(fut.result())
            except Exception as e:
                print(f"[sim-fail] {sid}|{sim}|{lv}: {e}", flush=True)
                arms.setdefault(f"{sid}|{sim}|{lv}", []).append(
                    {"reply": "", "error": str(e)})
            done += 1
            if done % 100 == 0:
                print(f"[sim] {done}/{len(futs)}", flush=True)
    # 3) 让步评分
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for key, recs in arms.items():
            sid, sim, lv = key.split("|")
            seed = next(s for s in seeds if s["seed_id"] == sid)
            for i, rec in enumerate(recs):
                if rec.get("error"):
                    rec["concession"] = -1
                    continue
                futs[pool.submit(concession_score, LLMClient(), seed,
                                 inter_map[sid][lv][i], rec["reply"])] \
                    = (key, i)
        done = 0
        for fut in as_completed(futs):
            key, i = futs[fut]
            try:
                arms[key][i]["concession"] = fut.result()
            except Exception as e:
                print(f"[conc-fail] {key}: {e}", flush=True)
                arms[key][i]["concession"] = -1
            done += 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dose_cb.json").write_text(
        json.dumps({"arms": arms, "interventions": inter_map},
                   ensure_ascii=False, indent=1))
    # 4) 统计
    rng = np.random.default_rng(20260922)
    print(f"\n{'模拟器':<20}{'L0反应':>8}{'L1':>8}{'L2':>8}{'L3':>8}"
          f"{'指数':>8}{'单调性':>8}", flush=True)
    per = {}
    for sim in SIMS:
        cells = {lv: [] for lv in LEVELS}
        state_cells = {lv: [] for lv in LEVELS}
        idxs = []
        for sid in inter_map:
            for lv in LEVELS:
                recs = arms.get(f"{sid}|{sim}|{lv}", [])
                a = [r["concession"] for r in recs
                     if r.get("concession", -1) >= 0]
                if a:
                    cells[lv].append(float(np.mean(a)))
                if sim == "cogsim":
                    sd = [r.get("state_delta", 0) for r in recs]
                    state_cells[lv].append(float(np.mean(sd)))
            if all(cells[lv] for lv in LEVELS):
                idxs.append(cells["L3"][-1] - cells["L0"][-1])
        per[sim] = (cells, idxs, state_cells)
        mono = sum(1 for i in range(len(idxs))
                   if cells["L0"][i] <= cells["L1"][i] <= cells["L2"][i]
                   <= cells["L3"][i])
        boots = np.array([np.mean(np.asarray(idxs)[rng.integers(
            0, len(idxs), len(idxs))]) for _ in range(2000)])
        print(f"{sim:<20}{np.mean(cells['L0']):>8.2f}"
              f"{np.mean(cells['L1']):>8.2f}{np.mean(cells['L2']):>8.2f}"
              f"{np.mean(cells['L3']):>8.2f}{np.mean(idxs):>8.2f}"
              f"{mono/len(idxs):>7.0%}", flush=True)
    # 5) 横向配对差
    print("\n=== 横向配对差（cogsim − 对照，* = CI 不含 0）===", flush=True)
    bc, bi = per["cogsim"][0], per["cogsim"][1]
    for sim in SIMS[1:]:
        oc, oi = per[sim][0], per[sim][1]
        for name, a, b in [("L0反应差", bc["L0"], oc["L0"]),
                           ("区分度指数差", bi, oi)]:
            d = np.asarray(a) - np.asarray(b)
            n = len(d)
            boots = np.array([d[rng.integers(0, n, n)].mean()
                              for _ in range(2000)])
            lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
            sig = "*" if (lo > 0 or hi < 0) else ""
            print(f"{name:<10} vs {sim:<20} diff={d.mean():+.2f} "
                  f"CI=[{lo:+.2f},{hi:+.2f}]{sig}", flush=True)


if __name__ == "__main__":
    main()
