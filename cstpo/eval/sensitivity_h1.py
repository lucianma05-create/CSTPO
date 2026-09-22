"""H1 认知敏感度可测性实验（问题 01 §5 协议，648 单轮分支）。

设计：
- 父节点：3 任务 × 4 种子（dev 前 4）的 vanilla 轨迹，取第 2/4/6 轮
  快照 = 12 节点（早/中/后阶段 × 4 画像）；
- 每节点：3 策略（cooperative/pushy/neutral）× 2 措辞（固定提示变体）
  × 3 重复 = 18 个单轮分支；
- 测量：分支一步后 vs 父快照的 ΔBDI（节点内归一化）、ΔEmotion、
  结构变化（新增/删除）；同模式/跨模式分别报告；
- 分析：方差分量（策略/措辞/重复）、节点敏感度排序跨重复批次稳定性、
  策略标签置换检验。

失败保护：单分支异常重试 1 次仍失败记 invalid（不赋零）；逐条落盘
skip-existing 续跑。

成本：~5k 调用 / ~5.5M tokens。
用法：cd CSTPO && python -m cstpo.eval.sensitivity_h1 --workers 200
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient, StructuredCallError

from cstpo.core.agent import SYSTEM_BUILDERS, agent_turn
from cstpo.core.checkpoint import restore, snapshot
from cstpo.core.cost_ledger import RecordingLLM
from cstpo.core.stale_judger import check_stale, should_check
from cstpo.core.task_env import TaskEnv, prefix_turns

OUT = ROOT / "data" / "sensitivity_h1"
TASKS = ("esconv", "p4g", "craigslistbargain")
NODE_TURNS = (2, 4, 6)
STRATEGIES = ("direct_solution", "challenge_confront", "empathize_withdraw")
# 每策略 2 个固定措辞变体（实验专用；不动香草本体与 STYLE_PROFILES）。
# 三个策略是行为级差异：直接给方案 / 质疑对抗 / 纯共情不推进
PHRASINGS = {
    "direct_solution": (
        "Immediately give a concrete actionable solution or offer. Skip empathy "
        "and small talk entirely. State the action and the next step in one go.",
        "Cut straight to the point: one clear proposal with specifics, no "
        "preliminaries, no softeners."),
    "challenge_confront": (
        "Challenge the user's stated position directly: point out "
        "contradictions, demand justification, refuse to concede.",
        "Push back hard on what the user just said. Question their reasoning "
        "and press for a firm answer."),
    "empathize_withdraw": (
        "Only reflect feelings and express understanding. Do NOT propose "
        "anything, do NOT push any direction. Just listen.",
        "Stay purely supportive: acknowledge emotions and ask nothing that "
        "demands a decision."),
}
N_REPEATS = 3
SAFETY_CAP = 6  # 父轨迹只跑到 6 轮（节点取完即止）


def load_seed(task, i):
    seeds = sorted((ROOT / "data" / "seeds_draft" / task).glob("*.json"))
    seeds = [p for p in seeds if p.name != "manifest.json"]
    return json.loads(seeds[i].read_text())


def gen_candidate(task, seed, turns, strategy, phrasing):
    """策略+措辞变体 → 下一句 agent 话语（只读公开历史）。"""
    system = SYSTEM_BUILDERS[task](seed) + "\n" + phrasing
    msgs = [{"role": "system", "content": system}]
    for h in turns:
        msgs.append({"role": "assistant" if h["role"] == "assistant" else "user",
                     "content": h["text"]})
    out = LLMClient().chat(msgs, max_tok=150)
    return (out or "").strip() or "(keep talking)"


def gen_parent(task, seed) -> dict:
    """vanilla 轨迹 6 轮，返回 {snapshots: {turn: snapshot_dict}, turns}。"""
    llm = RecordingLLM()
    env = TaskEnv(llm=llm, max_turns=SAFETY_CAP)
    cp = env.reset(seed)["checkpoint"]
    turns = prefix_turns(seed)
    snaps = {}
    r = None
    n = 0
    while not (r and r["terminated"]):
        if should_check(n):
            j = check_stale(LLMClient(), turns, task)
            if j["stale"] and j["final_line"]:
                turns.append({"role": "user", "text": j["final_line"]})
                break
        utter = agent_turn(llm, task, "vanilla", turns, seed)
        r = env.step(cp, utter)
        cp = r["checkpoint"]
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": r["user_reply"]})
        n += 1
        if n in NODE_TURNS:
            snaps[n] = {"snapshot": snapshot(r["sim"]), "turns": list(turns)}
        if n >= SAFETY_CAP:
            break
    return {"seed_id": seed["seed_id"], "snaps": snaps}


def measure(pre_state, post_state) -> dict:
    """单步转移测量（节点内归一化）。"""
    pre_bdi = pre_state.bdi_dict()
    post_bdi = post_state.bdi_dict()
    pre_items = {it["id"]: it["strength"] for it in pre_bdi["beliefs"]
                 + pre_bdi["desires"] + pre_bdi["intentions"]}
    post_items = {it["id"]: it["strength"] for it in post_bdi["beliefs"]
                  + post_bdi["desires"] + post_bdi["intentions"]}
    deltas = {}
    for k, v in pre_items.items():
        if k in post_items:
            deltas[k] = post_items[k] - v
    new_ids = sorted(set(post_items) - set(pre_items))
    deleted_ids = sorted(set(pre_items) - set(post_items))
    total_pre = sum(pre_items.values()) or 1.0
    mag = sum(abs(d) for d in deltas.values()) / total_pre
    return {"mag_bdi": round(mag, 4),
            "n_new": len(new_ids), "n_deleted": len(deleted_ids),
            "d_valence": round(post_state.emotion.valence
                               - pre_state.emotion.valence, 4),
            "d_arousal": round(post_state.emotion.arousal
                               - pre_state.emotion.arousal, 4),
            "cat_change": int(post_state.emotion.category
                              != pre_state.emotion.category),
            "mode": None}


def run_branch(args) -> dict:
    task, seed, turn, strategy, phrasing_i, rep, llm = args
    did = f"{task}_{seed['seed_id']}_t{turn}_{strategy}_p{phrasing_i}_r{rep}"
    out_f = OUT / "branches" / f"{did}.json"
    if out_f.exists():
        return json.loads(out_f.read_text())
    phrasing = PHRASINGS[strategy][phrasing_i]
    try:
        with (OUT / "trajectories" /
              f"{task}_{seed['seed_id']}.pkl").open("rb") as f:
            parent = pickle.load(f)
        snap = parent["snaps"][turn]  # 键为 int（轮数）
        turns = snap["turns"]
        pre_state = snap["snapshot"]["state"]
        # restore 深拷贝快照 → 一步
        sim = restore(snap["snapshot"], llm)
        utter = gen_candidate(task, seed, turns, strategy, phrasing)
        pre_sim = snapshot(sim)
        reply = sim.simulate_turn(utter)
        m = measure(pre_sim["state"], sim.state)
        m["mode"] = sim.logs[-1].mode
        m["utter"] = utter[:200]
        m["reply"] = reply[:200]
        m["did"] = did
        m["invalid"] = False
    except (StructuredCallError, Exception) as e:  # 重试一次
        try:
            sim = restore(snap["snapshot"], llm)
            utter = gen_candidate(task, seed, turns, strategy, phrasing)
            pre_sim = snapshot(sim)
            reply = sim.simulate_turn(utter)
            m = measure(pre_sim["state"], sim.state)
            m["mode"] = sim.logs[-1].mode
            m["utter"] = utter[:200]
            m["reply"] = reply[:200]
            m["did"] = did
            m["invalid"] = False
        except Exception as e2:
            m = {"did": did, "invalid": True, "error": f"{type(e2).__name__}"}
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")
    return m


def analyze():
    """方差分量 + 排序稳定性 + 置换检验（简化版报告）。"""
    import numpy as np
    rows = {}
    for p in (OUT / "branches").glob("*.json"):
        d = json.loads(p.read_text())
        if d.get("invalid"):
            continue
        # did 格式: {task}_{seed_id}_t{turn}_{strategy}_p{pi}_r{rep}
        # seed_id 本身含下划线（如 cb_01），turn 在 t 前缀段
        parts = d["did"].split("_")
        task = parts[0]
        ti = next(i for i, x in enumerate(parts)
                  if x.startswith("t") and x[1:].isdigit())
        pi = next(i for i, x in enumerate(parts)
                  if x.startswith("p") and x[1:].isdigit())
        turn = parts[ti][1:]
        strategy = "_".join(parts[ti + 1:pi])  # 策略名可含下划线
        rows.setdefault((task, turn), []).append((strategy, d["mag_bdi"]))
    print(f"===== H1 分析（节点 = 任务×轮次；mag_bdi 节点内归一化） =====")
    n_invalid = len([p for p in (OUT / "branches").glob("*.json")
                     if json.loads(p.read_text()).get("invalid")])
    print(f"invalid 分支: {n_invalid}")
    for (task, turn), vals in sorted(rows.items()):
        by_strat = {}
        for s, v in vals:
            by_strat.setdefault(s, []).append(v)
        means = {s: np.mean(v) for s, v in by_strat.items()}
        overall = np.var([v for _, v in vals])
        between = np.var(list(means.values()))
        within = np.mean([np.var(v) for v in by_strat.values()])
        print(f"{task} t{turn}: 策略均值 { {s: round(m,3) for s,m in means.items()} }"
              f" | 组间方差 {between:.5f} | 组内方差 {within:.5f}"
              f" | 比 {between/max(within,1e-9):.2f}")
    (OUT / "summary.json").write_text(json.dumps(
        {"n_branches": len(rows), "n_invalid": n_invalid}, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=200)
    ap.add_argument("--phase", default="all",
                    choices=["parents", "branches", "analyze", "all"])
    args = ap.parse_args()

    seeds = {t: [load_seed(t, i) for i in range(4)] for t in TASKS}

    if args.phase in ("parents", "all"):
        (OUT / "trajectories").mkdir(parents=True, exist_ok=True)
        jobs = [(t, s) for t in TASKS for s in seeds[t]]
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futs = {ex.submit(gen_parent, t, s): (t, s) for t, s in jobs}
            for fut in as_completed(futs):
                t, s = futs[fut]
                res = fut.result()
                # 快照含 UserState 对象，用 pickle（本地实验数据）
                with (OUT / "trajectories" / f"{t}_{s['seed_id']}.pkl").open("wb") as f:
                    pickle.dump(res, f)
        print("父轨迹完成")

    if args.phase in ("branches", "all"):
        jobs = []
        for t in TASKS:
            for s in seeds[t]:
                with (OUT / "trajectories" / f"{t}_{s['seed_id']}.pkl").open("rb") as f:
                    turns_avail = set(pickle.load(f)["snaps"].keys())
                for turn in NODE_TURNS:
                    if turn not in turns_avail:
                        continue  # 轨迹提前终止的节点不生成分支
                    for strategy in STRATEGIES:
                        for pi in (0, 1):
                            for rep in range(N_REPEATS):
                                jobs.append((t, s, turn, strategy, pi, rep,
                                             RecordingLLM()))
        print(f"分支 {len(jobs)} 个（workers={args.workers}）...", flush=True)
        n_ok = n_bad = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_branch, j) for j in jobs]
            for k, fut in enumerate(as_completed(futs)):
                m = fut.result()
                if m.get("invalid"):
                    n_bad += 1
                else:
                    n_ok += 1
                if (k + 1) % 100 == 0:
                    print(f"  {k+1}/{len(jobs)}（invalid {n_bad}）", flush=True)
        print(f"分支完成: 有效 {n_ok} / invalid {n_bad}")

    if args.phase in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
