"""证据梯度剂量-反应探针：验证"不轻易被说服且不固化"。

同一用户同一时刻，agent 话语按证据强度四档（每档 2 措辞，flash 按种子
生成）：L0 空话（零信息重复劝说）、L1 泛泛信息、L2 针对该用户具体顾虑
的实质信息、L3 = L2 + 可行方案。四档模拟器各从同一快照分支续演一步。

测量（每臂）：回复文本 + 态度变化评分（0-3 单维 judge：上下文 + 该句 +
回复 → 用户态度向 agent 目标移动了多少）；Cog-Sim 额外记录 BDI 总
|delta|（状态版反应）。

主指标（每种子 × 模拟器）：
- L0 反应 = L0 两措辞态度分均值（越低越好 = 不轻易被说服）；
- 区分度指数 = L3 反应 − L0 反应（越高越好 = 证据敏感、不固化）。

横向对比：cogsim vs 三档对照，逐种子配对，bootstrap 95% CI。
预注册判据：①cogsim L0 不高于最强对照 + 0.2；②cogsim 区分度指数
显著高于最强对照。
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
    "You are writing what an agent says to a user in a dialogue task. "
    "Return exactly one JSON object with four keys L0, L1, L2, L3; each key "
    "maps to a list of exactly two short utterances (one or two sentences "
    "each). All eight utterances are spoken at the SAME moment to the SAME "
    "user, differing only in evidence strength:\n"
    "L0: empty repetition with zero new information (vague persuasion or "
    "generic comfort, no facts);\n"
    "L1: generic information anyone could say (general facts not tailored "
    "to this user);\n"
    "L2: specific, relevant information addressing THIS user's particular "
    "concerns or situation (tailored, substantive);\n"
    "L3: the L2-level specific information PLUS a concrete feasible next "
    "step the user could take.")

_ATTITUDE_SYSTEM = (
    "Read the user's situation and conversation so far, then the agent's "
    "latest utterance, then the user's reply. Judge how much the user's "
    "attitude or stance moved TOWARD the agent's goal in this reply. "
    "0 = no movement (or pushback); 1 = slight softening or mild interest; "
    "2 = noticeable movement (considers it, partial agreement); "
    "3 = clear movement (agrees, commits, or takes the proposed step). "
    "Score conservatively. Return exactly one JSON object: "
    '{"change": 0, "reason": "one short sentence"}.')


def gen_interventions(llm: LLMClient, task: str, seed: dict) -> dict:
    from cstpo.core.task_env import render_persona, prefix_turns

    user = (f"Task: {task}\nUser situation: {render_persona(seed)}\n"
            f"Conversation so far:\n"
            + "\n".join(f"{t['role']}: {t['text']}"
                        for t in prefix_turns(seed)))
    out = llm.chat_json(
        [{"role": "system", "content": _INTERVENTION_SYSTEM},
         {"role": "user", "content": user}], max_tok=600)
    return {lv: [str(x).strip() for x in (out.get(lv) or [])][:2]
            for lv in LEVELS}


def make_job(seed: dict, task: str, sim: str, lv: str, utter: str) -> dict:
    """一个臂 = 同快照 × 一句话。cogsim 走 TaskEnv，提示式走
    StandardUserSim。返回记录 dict。带 2 次重试（API 抖动容错）。"""
    from cstpo.core.task_env import TaskEnv
    from cstpo.core.standard_user_sim import StandardUserSim

    for attempt in range(3):
        try:
            return _make_job_once(seed, task, sim, utter)
        except Exception:
            if attempt == 2:
                return {"reply": "", "attitude": -1, "error": "sim_fail"}
    return {"reply": "", "attitude": -1, "error": "sim_fail"}


def _make_job_once(seed: dict, task: str, sim: str, utter: str) -> dict:
    from cstpo.core.task_env import TaskEnv
    from cstpo.core.standard_user_sim import StandardUserSim

    llm = LLMClient()
    if sim == "cogsim":
        env = TaskEnv(llm=llm, max_turns=30, debug=False)
        reset = env.reset(seed)
        r = env.step(copy.deepcopy(reset["checkpoint"]), utter)
        st = r["sim"].state
        # 状态版反应：BDI 总 |delta|（v1 口径；x 侧是 dict，sim 侧是对象）
        base_strengths = {b["id"]: b["strength"] for b in (
            reset["x"]["bdi"]["beliefs"] + reset["x"]["bdi"]["desires"]
            + reset["x"]["bdi"]["intentions"])}
        post_strengths = {b.id: b.strength for b in (
            list(st.beliefs) + list(st.desires) + list(st.intentions))}
        state_delta = sum(abs(post_strengths.get(k, 0) - v)
                          for k, v in base_strengths.items())
        return {"reply": r["user_reply"], "state_delta": round(state_delta, 3),
                "valence_delta": round(
                    st.emotion.valence - reset["x"]["emotion"]["valence"], 3)}
    usim = StandardUserSim(LLMClient(), variant=sim.replace("std_", ""))
    reset = usim.reset(seed)
    r = usim.step(copy.deepcopy(reset["checkpoint"]), utter)
    return {"reply": r["user_reply"]}


def attitude_score(llm: LLMClient, seed: dict, utter: str, reply: str) -> int:
    from cstpo.core.task_env import prefix_turns

    ctx = "\n".join(f"{t['role']}: {t['text']}" for t in prefix_turns(seed))
    out = llm.chat_json(
        [{"role": "system", "content": _ATTITUDE_SYSTEM},
         {"role": "user",
          "content": f"Conversation so far:\n{ctx}\n\nAgent's utterance: "
                     f"{utter}\n\nUser's reply: {reply}"}],
        max_tok=150)
    try:
        return max(0, min(3, int(out.get("change", -1))))
    except (TypeError, ValueError):
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--tasks", default="esconv,p4g,craigslistbargain")
    ap.add_argument("--workers", type=int, default=240)
    ap.add_argument("--resume", action="store_true",
                    help="增量模式：跳过已有记录，只补跑新种子并合并")
    args = ap.parse_args()
    load_api_key()
    seeds = {}
    for task in args.tasks.split(","):
        files = sorted((ROOT / "data" / "seeds_draft" / task).glob("*.json"))
        files = [f for f in files if f.name != "manifest.json"]
        seeds[task] = [json.loads(f.read_text()) for f in files[: args.n_seeds]]
    # 增量：读入已有产物，过滤已测种子
    old_arms, old_inter = {}, {}
    if args.resume and (OUT_DIR / "dose_probe.json").exists():
        old = json.loads((OUT_DIR / "dose_probe.json").read_text())
        old_arms, old_inter = old["arms"], old["interventions"]
        done_ids = {k.split("|")[0] + "|" + k.split("|")[1]
                    for k in old_arms}
        seeds = {t: [s for s in ss if f"{t}|{s['seed_id']}" not in done_ids]
                 for t, ss in seeds.items()}
        print(f"[resume] 已有 {len(old_arms)} 臂，待补种子："
              + ", ".join(f"{t}={len(ss)}" for t, ss in seeds.items()),
              flush=True)
    # 1) 生成干预（每种子 8 句）
    print("生成干预...", flush=True)
    inter_map = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for task, ss in seeds.items():
            for seed in ss:
                futs[pool.submit(gen_interventions, LLMClient(), task, seed)] \
                    = (task, seed)
        for fut in as_completed(futs):
            task, seed = futs[fut]
            inter_map[f"{task}|{seed['seed_id']}"] = fut.result()
    # 2) 模拟续演（4 模拟器 × 4 档 × 2 措辞）
    print("模拟续演...", flush=True)
    arms = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for task, ss in seeds.items():
            for seed in ss:
                inter = inter_map[f"{task}|{seed['seed_id']}"]
                for sim in SIMS:
                    for lv in LEVELS:
                        for utter in inter[lv]:
                            key = f"{task}|{seed['seed_id']}|{sim}|{lv}"
                            futs[pool.submit(make_job, seed, task, sim, lv,
                                             utter)] = key
        done = 0
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                arms.setdefault(key, []).append(fut.result())
            except Exception as e:
                print(f"[sim-fail] {key}: {type(e).__name__}: {e}",
                      flush=True)
                arms.setdefault(key, []).append(
                    {"reply": "", "attitude": -1, "error": str(e)})
            done += 1
            if done % 300 == 0:
                print(f"[sim] {done}/{len(futs)}", flush=True)
    # 3) 态度评分（每臂一次）
    print("态度评分...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for key, recs in arms.items():
            task, sid, sim, lv = key.split("|")
            seed = next(s for t, ss in seeds.items() if t == task
                        for s in ss if s["seed_id"] == sid)
            inter = inter_map[f"{task}|{sid}"]
            for i, rec in enumerate(recs):
                futs[pool.submit(attitude_score, LLMClient(), seed,
                                 inter[lv][i], rec["reply"])] = (key, i)
        done = 0
        for fut in as_completed(futs):
            key, i = futs[fut]
            try:
                arms[key][i]["attitude"] = fut.result()
            except Exception as e:
                print(f"[att-fail] {key}: {type(e).__name__}: {e}",
                      flush=True)
                arms[key][i]["attitude"] = -1
            done += 1
            if done % 300 == 0:
                print(f"[att] {done}/{len(futs)}", flush=True)
    if args.resume:
        arms.update(old_arms)
        inter_map.update(old_inter)
    (OUT_DIR / "dose_probe.json").write_text(
        json.dumps({"arms": arms, "interventions": inter_map},
                   ensure_ascii=False, indent=1))
    # 4) 每种子×模拟器：L0 反应、L3 反应、区分度指数
    print("\n=== 结果（均值 ± 种子聚类 bootstrap）===", flush=True)
    rng = np.random.default_rng(20260922)
    print(f"{'模拟器':<20}{'L0反应':>8}{'L3反应':>8}{'区分度指数':>10}"
          f"{'状态区分度(cogsim)':>14}")
    per = {}
    for sim in SIMS:
        l0s, l3s, idxs, state_idxs = [], [], [], []
        for key, recs in arms.items():
            task, sid, s, lv = key.split("|")
            if s != sim:
                continue
            a = [r["attitude"] for r in recs if r["attitude"] >= 0]
            if lv == "L0" and a:
                l0s.append((task, sid, float(np.mean(a))))
            if lv == "L3" and a:
                l3s.append((task, sid, float(np.mean(a))))
                if sim == "cogsim":
                    state_idxs.append(float(np.mean(
                        [r.get("state_delta", 0) for r in recs])))
        for (t1, s1, l0), (t2, s2, l3) in zip(sorted(l0s), sorted(l3s)):
            if (t1, s1) == (t2, s2):
                idxs.append(l3 - l0)
        per[sim] = (l0s, l3s, idxs, state_idxs)
        boots = np.array([np.mean(np.asarray(idxs)[rng.integers(
            0, len(idxs), len(idxs))]) for _ in range(2000)])
        print(f"{sim:<20}{np.mean([x[2] for x in l0s]):>8.2f}"
              f"{np.mean([x[2] for x in l3s]):>8.2f}"
              f"{np.mean(idxs):>10.2f}"
              f" CI=[{np.percentile(boots,2.5):+.2f},{np.percentile(boots,97.5):+.2f}]"
              f"{np.mean(state_idxs):>14.2f}")
    # 5) 横向配对差（cogsim − 对照）
    print("\n=== 横向配对差（cogsim − 对照，* = CI 不含 0）===")
    base_l0 = dict(((t, s), v) for t, s, v in per["cogsim"][0])
    base_idx = per["cogsim"][2]
    for sim in SIMS[1:]:
        opp_l0 = dict(((t, s), v) for t, s, v in per[sim][0])
        common = sorted(set(base_l0) & set(opp_l0))
        d_l0 = np.array([base_l0[k] - opp_l0[k] for k in common])
        d_idx = None
        # 区分度差：种子对齐后相减
        di = {}
        for (t1, s1, l0), (t2, s2, l3) in zip(sorted(per["cogsim"][0]),
                                              sorted(per["cogsim"][1])):
            if (t1, s1) == (t2, s2):
                di[(t1, s1)] = l3 - l0
        do = {}
        for (t1, s1, l0), (t2, s2, l3) in zip(sorted(per[sim][0]),
                                              sorted(per[sim][1])):
            if (t1, s1) == (t2, s2):
                do[(t1, s1)] = l3 - l0
        keys = sorted(set(di) & set(do))
        d_idx = np.array([di[k] - do[k] for k in keys])
        for name, d in [("L0反应差", d_l0), ("区分度指数差", d_idx)]:
            n = len(d)
            boots = np.array([d[rng.integers(0, n, n)].mean()
                              for _ in range(2000)])
            lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
            sig = "*" if (lo > 0 or hi < 0) else ""
            print(f"{name:<10} vs {sim:<20} diff={d.mean():+.2f} "
                  f"CI=[{lo:+.2f},{hi:+.2f}]{sig}")


if __name__ == "__main__":
    main()
