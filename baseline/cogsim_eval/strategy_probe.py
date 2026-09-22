"""策略维度横向对比探针（同一快照 × 3 策略 × 2 措辞）。

目的：验证 Cog-Sim 用户对不同**策略类型**的反应有可测差异（方向性假设
预注册）——这是字段优势 A_strategy 的前置证据，也是 H1 失败后的修订版
（H1 教训：不测聚合幅度，测方向符合率；措辞取均值压噪；手工固定措辞）。

每任务 3 个 canonical 策略（对比度最大）× 2 条固定措辞，全部从同一
动作前快照分支续演一步。记录：逐锚点 BDI 强度 delta、情绪 delta、
回复文本。方向性假设（预注册）：

- esconv H1：valence(Reflection of feelings) > valence(Question)
  （共情比提问更安抚情绪）
- esconv H2：意图/愿望增量(Providing Suggestions) > 增量(Reflection)
  （建议比共情更促行动意图；种子无意图锚点时退化到 desires）
- p4g H1：valence(emotion_appeal) > valence(donate_request)
  （情感诉求比直接要钱更不伤情绪）
- p4g H2：valence(logical_appeal) > valence(donate_request)
- cb H1：立场 BDI |delta|(propose_price) > |delta|(inform)
  （报价直触立场，询问只收集信息）
- cb H2：valence(inform) >= valence(propose_price)
  （直接报价比信息询问更易让情绪走低）

主指标 = 每个假设的**方向符合率**（满足条件的种子占比），种子聚类
bootstrap 95% CI。
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

from baseline.runner import load_api_key  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "cogsim_eval"

# 固定措辞（手工，H1 同款做法）：每策略 2 条
WORDINGS = {
    "esconv": {
        "Reflection of feelings": [
            "It sounds like you're carrying a lot of pain right now, and that must feel overwhelming.",
            "I can hear how much this has been weighing on you — it's really hard."],
        "Providing Suggestions": [
            "One small step that might help is to go for a walk today and clear your head.",
            "Maybe you could try talking to a close friend about what happened — it often helps to share."],
        "Question": [
            "What's been the hardest part of this for you so far?",
            "Can you tell me more about what happened?"],
    },
    "p4g": {
        "emotion_appeal": [
            "Think of the children who go to bed hungry tonight — your gift could change that.",
            "A small donation from you could bring hope to a child who has none."],
        "logical_appeal": [
            "Save the Children publishes audited reports showing most donations go directly to programs for children.",
            "Every dollar you give is tracked, and most of it reaches children in need within weeks."],
        "donate_request": [
            "So, will you donate today?",
            "Please make a donation now — even a small amount helps."],
    },
    "craigslistbargain": {
        "propose_price": [
            "I'll pay five dollars under your asking price, cash today.",
            "Would you take ten dollars below asking? I can close the deal now."],
        "inquire": [
            "Could you tell me more about the condition and how long you've had it?",
            "Is there any flexibility on the price, and what's included?"],
        "inform": [
            "I've compared a few listings like yours, and similar ones usually go for a bit less.",
            "I'm a serious buyer and I can complete the payment today if we agree."],
    },
}

HYPOTHESES = [
    ("esconv", "H1_valence_reflection_vs_question", "valence",
     "Reflection of feelings", "Question", "gt"),
    ("esconv", "H2_intent_suggestion_vs_reflection", "intent",
     "Providing Suggestions", "Reflection of feelings", "gt"),
    ("p4g", "H1_valence_emotion_vs_request", "valence",
     "emotion_appeal", "donate_request", "gt"),
    ("p4g", "H2_valence_logic_vs_request", "valence",
     "logical_appeal", "donate_request", "gt"),
    ("craigslistbargain", "H1_stance_price_vs_inform", "stance",
     "propose_price", "inform", "gt"),
    ("craigslistbargain", "H2_valence_inform_vs_price", "valence",
     "inform", "propose_price", "ge"),
]


def bdi_deltas(seed: dict, wordings: dict) -> dict:
    """同快照分支：每策略每措辞续演一步，返回逐臂记录。"""
    from cstpo.core.task_env import TaskEnv

    env = TaskEnv(llm=LLMClient(), max_turns=30, debug=False)
    reset = env.reset(seed)
    cp0 = reset["checkpoint"]

    def strength_map(state, kind):
        items = {"belief": state.beliefs, "desire": state.desires,
                 "intention": state.intentions}[kind]
        return {b.id: b.strength for b in items}

    base = {k: strength_map(cp0["sim_snapshot"]["state"], k)
            for k in ("belief", "desire", "intention")}
    arms = {}
    for strat, ws in wordings.items():
        for wi, utter in enumerate(ws):
            cp = copy.deepcopy(cp0)
            r = env.step(cp, utter)
            st = r["sim"].state
            post = {k: strength_map(st, k)
                    for k in ("belief", "desire", "intention")}
            arms[f"{strat}__{wi}"] = {
                "utterance": utter, "reply": r["user_reply"],
                "emotion_before": (reset["x"]["emotion"]["valence"],
                                   reset["x"]["emotion"]["arousal"]),
                "emotion_after": (st.emotion.valence, st.emotion.arousal),
                "belief": {k: round(post["belief"].get(k, 0) - v, 3)
                           for k, v in base["belief"].items()},
                "desire": {k: round(post["desire"].get(k, 0) - v, 3)
                           for k, v in base["desire"].items()},
                "intention": {k: round(post["intention"].get(k, 0) - v, 3)
                              for k, v in base["intention"].items()},
            }
    return arms


def metric(seed_rec: dict, strat: str, kind: str) -> float:
    """策略两措辞取均值：valence=情绪价变化；intent=意图/愿望总|delta|；
    stance=信念+意图+愿望总|delta|。"""
    arms = {k: v for k, v in seed_rec.items() if k.startswith(strat + "__")}
    vals = []
    for a in arms.values():
        if kind == "valence":
            vals.append(a["emotion_after"][0] - a["emotion_before"][0])
        elif kind == "intent":
            d = {**a["intention"], **a["desire"]}
            vals.append(sum(abs(v) for v in d.values()))
        else:  # stance
            d = {**a["belief"], **a["desire"], **a["intention"]}
            vals.append(sum(abs(v) for v in d.values()))
    return float(np.mean(vals))


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
            arms = bdi_deltas(seed, wd)
            records[f"{task}|{seed['seed_id']}"] = {
                "task": task, "seed_id": seed["seed_id"], "arms": arms}
            print(f"[{task}] {seed['seed_id']} done", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "strategy_probe.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1))
    # 方向符合率（种子聚类 bootstrap）
    rng = np.random.default_rng(20260921)
    print("\n=== 方向性假设符合率（种子聚类 bootstrap 95% CI）===")
    for task, name, kind, s_a, s_b, op in HYPOTHESES:
        hits = []
        for key, rec in records.items():
            if rec["task"] != task:
                continue
            a, b = metric(rec["arms"], s_a, kind), metric(rec["arms"], s_b, kind)
            hits.append(1 if (a > b if op == "gt" else a >= b) else 0)
        hits = np.asarray(hits)
        n = len(hits)
        boots = np.array([hits[rng.integers(0, n, n)].mean()
                          for _ in range(2000)])
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        print(f"{name:<40} 符合率={hits.mean():.0%} "
              f"CI=[{lo:.0%},{hi:.0%}] (n={n})")


if __name__ == "__main__":
    main()
