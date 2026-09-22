"""策略维度横向对比 v2：四档模拟器同场竞技。

同上下文（种子）× 3 策略 × 2 措辞，分别喂给四档用户模拟器：
cogsim（TaskEnv）+ std_persona / std_persona_resist / std_bdi
（StandardUserSim，与 diversity_probe 同口径）。

量化指标（全模拟器可比）：
- 回复极性（flash 三分类 worse/same/better → -1/0/+1，作情绪代理）；
- 回复长度；
- Cog-Sim 额外：情绪 valence/arousal delta、逐锚点 BDI delta。

方向性假设（同 v1，预注册；用极性代理对全模拟器统一检验）：
- esconv H1 修订：建议 > 提问 > 共情（极性）——v1 实测顺序；
- p4g H1：情感诉求 > 直接要钱；p4g H2：逻辑诉求 > 直接要钱；
- cb H1：报价立场变化 > 询问（仅 cogsim 状态可比，提示式用极性
  "报价 ≥ 询问" 近似：报价引发更强反应）；cb H2：询问极性 ≥ 报价。

输出：每档模拟器 × 每个假设的方向符合率 + 种子配对横向差（cogsim −
对照档）的 bootstrap CI。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient  # noqa: E402

from baseline.cogsim_eval.strategy_probe import WORDINGS  # noqa: E402
from baseline.runner import load_api_key  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "cogsim_eval"
SIMS = ["cogsim", "std_persona", "std_persona_resist", "std_bdi"]

# 极性代理版假设（对全模拟器）：(任务, 名称, 策略A, 策略B, 方向)
HYPOTHESES_POLARITY = [
    ("esconv", "H1_suggest_vs_reflect", "Providing Suggestions",
     "Reflection of feelings", "gt"),
    ("esconv", "H1b_suggest_vs_question", "Providing Suggestions",
     "Question", "gt"),
    ("p4g", "H1_emotion_vs_request", "emotion_appeal",
     "donate_request", "gt"),
    ("p4g", "H2_logic_vs_request", "logical_appeal",
     "donate_request", "gt"),
    ("craigslistbargain", "H1b_price_react_vs_inquire", "propose_price",
     "inquire", "le"),  # 报价引发更强（更负）反应 → 极性更低
    ("craigslistbargain", "H2_inform_vs_price", "inquire",
     "propose_price", "gt"),
]

POLARITY_SYSTEM = (
    "Read the user's reply in a dialogue. Classify the user's emotional "
    "tone as one of: worse (negative), same (neutral), better (positive). "
    "Return exactly one JSON object: {\"polarity\": \"worse\"| \"same\" | "
    "\"better\"}.")


def polarity(llm: LLMClient, reply: str) -> int:
    out = llm.chat_json(
        [{"role": "system", "content": POLARITY_SYSTEM},
         {"role": "user", "content": reply}], max_tok=50)
    return {"worse": -1, "same": 0, "better": 1}.get(out.get("polarity"), 0)


def run_sim(sim: str, seed: dict, wd: dict) -> dict:
    """同快照 × 3 策略 × 2 措辞。cogsim 走 TaskEnv；提示式走
    StandardUserSim（同一可见前缀口径）。"""
    from cstpo.core.task_env import TaskEnv, prefix_turns
    from cstpo.core.standard_user_sim import StandardUserSim

    arms = {}
    if sim == "cogsim":
        env = TaskEnv(llm=LLMClient(), max_turns=30, debug=False)
        reset = env.reset(seed)
        cp0 = reset["checkpoint"]
        for strat, ws in wd.items():
            for wi, utter in enumerate(ws):
                r = env.step(copy.deepcopy(cp0), utter)
                st = r["sim"].state
                arms[f"{strat}__{wi}"] = {
                    "reply": r["user_reply"],
                    "valence_delta": round(st.emotion.valence
                                           - reset["x"]["emotion"]["valence"], 3)}
    else:
        usim = StandardUserSim(LLMClient(), variant=sim.replace("std_", ""))
        reset = usim.reset(seed)
        cp0 = reset["checkpoint"]
        for strat, ws in wd.items():
            for wi, utter in enumerate(ws):
                r = usim.step(copy.deepcopy(cp0), utter)
                arms[f"{strat}__{wi}"] = {"reply": r["user_reply"]}
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
    load_api_key()
    records = {}
    for task, wd in WORDINGS.items():
        files = sorted((ROOT / "data" / "seeds_draft" / task).glob("*.json"))
        files = [f for f in files if f.name != "manifest.json"]
        for f in files[: args.n_seeds]:
            seed = json.loads(f.read_text())
            for sim in SIMS:
                arms = run_sim(sim, seed, wd)
                records[f"{task}|{seed['seed_id']}|{sim}"] = {
                    "task": task, "seed_id": seed["seed_id"], "sim": sim,
                    "arms": arms}
            print(f"[{task}] {seed['seed_id']} done", flush=True)
    # 极性打分（全模拟器可比指标）
    llm = LLMClient()
    for key, rec in records.items():
        for a in rec["arms"].values():
            a["polarity"] = polarity(llm, a["reply"])
            a["length"] = len(a["reply"].split())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "strategy_probe_v2.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1))

    def strat_mean(rec, strat, metric):
        vals = [a[metric] for k, a in rec["arms"].items()
                if k.startswith(strat + "__")]
        return float(np.mean(vals)) if vals else float("nan")

    rng = np.random.default_rng(20260921)
    print("\n=== 方向符合率（极性代理，n=种子数/档）===")
    results = {}
    for task, name, s_a, s_b, op in HYPOTHESES_POLARITY:
        row = []
        for sim in SIMS:
            hits = []
            for key, rec in records.items():
                if rec["task"] != task or rec["sim"] != sim:
                    continue
                a = strat_mean(rec, s_a, "polarity")
                b = strat_mean(rec, s_b, "polarity")
                hits.append(1 if (a > b if op == "gt" else a <= b) else 0)
            hits = np.asarray(hits)
            n = len(hits)
            boots = np.array([hits[rng.integers(0, n, n)].mean()
                              for _ in range(2000)])
            row.append((hits.mean(), np.percentile(boots, 2.5),
                        np.percentile(boots, 97.5), n))
            results[(name, sim)] = hits
        line = f"{name:<36}"
        for sim, (m, lo, hi, n) in zip(SIMS, row):
            line += f" {sim.split('_')[-1]:<8} {m:.0%} [{lo:.0%},{hi:.0%}]"
        print(line)

    print("\n=== 横向对比：cogsim 与对照档符合率差异（逐种子配对）===")
    for task, name, s_a, s_b, op in HYPOTHESES_POLARITY:
        base = results[(name, "cogsim")]
        for sim in SIMS[1:]:
            opp = results[(name, sim)]
            d = base - opp
            n = len(d)
            boots = np.array([d[rng.integers(0, n, n)].mean()
                              for _ in range(2000)])
            lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
            print(f"{name:<36} vs {sim:<18} "
                  f"cogsim={base.mean():.0%} opp={opp.mean():.0%} "
                  f"diff={d.mean():+.0%} CI=[{lo:+.0%},{hi:+.0%}]")


if __name__ == "__main__":
    main()
