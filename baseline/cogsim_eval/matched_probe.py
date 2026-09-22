"""Cog-Sim 条件响应性诊断（P2 匹配干预的自动化版）。

同一用户同一动作前快照，三支干预分别续演一步：
- generic：任务相关的泛泛话语，无新信息（预期：状态几乎不动）；
- evidence：针对该用户处境的具体有效信息（预期：相关信念/意图强度
  出现方向合理的更新，情绪不恶化）；
- pressure：施压/催促（预期：情绪向负，或至少不比 generic 更正向）。

每支从同一 checkpoint 分支（P0 快照基础设施），记录：
- 用户回复文本；
- 各 BDI 条目强度变化（含内容与 source_kind）；
- 情绪 valence/arousal 变化；
- 方向性检验：evidence 支的状态变化是否大于 generic 支且方向合理，
  pressure 支情绪是否不优于 generic 支。

干预话语由 flash 按种子生成（generic/evidence/pressure 各一句），
生成后人工抽检；仅诊断用途，不替代正式 P2 盲评。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient  # noqa: E402

from baseline.runner import load_api_key  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "cogsim_eval"
TASKS = ["esconv", "p4g", "craigslistbargain"]

_INTERVENTION_SYSTEM = (
    "You are designing three short agent utterances for a dialogue task, "
    "all spoken by the assistant to the SAME user at the SAME moment. "
    "Return exactly one JSON object with three keys: "
    '"generic" (a generic, empty statement with no new information, e.g. '
    'vague persuasion or generic comfort), '
    '"evidence" (a statement containing specific, relevant, helpful '
    'information or a concrete feasible plan tailored to THIS user), and '
    '"pressure" (a pushy statement that urges immediate action and ignores '
    'the user\'s feelings). Each value is one or two short sentences.')


def gen_interventions(llm, task: str, seed: dict) -> dict:
    from cstpo.core.task_env import render_persona, prefix_turns

    user = (f"Task: {task}\n"
            f"User situation: {render_persona(seed)}\n"
            f"Conversation so far:\n"
            + "\n".join(f"{t['role']}: {t['text']}" for t in prefix_turns(seed)))
    out = llm.chat_json(
        [{"role": "system", "content": _INTERVENTION_SYSTEM},
         {"role": "user", "content": user}],
        max_tok=400)
    return {"generic": (out.get("generic") or "").strip(),
            "evidence": (out.get("evidence") or "").strip(),
            "pressure": (out.get("pressure") or "").strip()}


def state_probe(seed: dict, interventions: dict) -> dict:
    """三支从同一快照各续演一步，返回每支的 BDI/情绪变化与回复。"""
    from cstpo.core.checkpoint import snapshot
    from cstpo.core.task_env import TaskEnv, prefix_turns

    env = TaskEnv(llm=LLMClient(), max_turns=30, debug=False)
    reset = env.reset(seed)
    cp0 = reset["checkpoint"]
    base_state = reset["x"]

    def bdi_strength(state) -> dict:
        return {b["id"]: b["strength"] for b in state.beliefs + state.desires
                + state.intentions}

    def arm(utter):
        cp = copy.deepcopy(cp0)
        r = env.step(cp, utter)
        post = r["turn_record"]
        st = r["sim"].state  # step 后的模拟器状态（直接读，不可 pickle）
        before = {"emotion": (base_state["emotion"]["valence"],
                              base_state["emotion"]["arousal"])}
        after = {"emotion": (st.emotion.valence, st.emotion.arousal)}
        return {"utterance": utter, "reply": r["user_reply"],
                "emotion_before": before["emotion"],
                "emotion_after": after["emotion"],
                "update_notes": post["update_notes"]}

    return {k: arm(v) for k, v in interventions.items()}


def bdi_delta_per_arm(seed: dict, interventions: dict) -> dict:
    """逐臂 BDI 强度变化（按条目 id 对齐）。"""
    from cstpo.core.task_env import TaskEnv

    env = TaskEnv(llm=LLMClient(), max_turns=30, debug=False)
    reset = env.reset(seed)
    cp0 = reset["checkpoint"]

    def strengths(state):
        return {b.id: b.strength
                for b in state.beliefs + state.desires + state.intentions}

    # 快照是纯数据 dict：state 字段即动作前的 UserState 深拷贝
    base = strengths(cp0["sim_snapshot"]["state"])
    out = {}
    for k, utter in interventions.items():
        cp = copy.deepcopy(cp0)
        r = env.step(cp, utter)
        post = strengths(r["sim"].state)
        out[k] = {bid: round(post.get(bid, 0.0) - base.get(bid, 0.0), 3)
                  for bid in post}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()
    load_api_key()
    seeds = {}
    for task in args.tasks.split(","):
        files = sorted((ROOT / "data" / "seeds_draft" / task).glob("*.json"))
        files = [f for f in files if f.name != "manifest.json"]
        seeds[task] = [json.loads(f.read_text()) for f in files[: args.n_seeds]]
    records = []
    llm = LLMClient()
    for task, ss in seeds.items():
        for seed in ss:
            inter = gen_interventions(llm, task, seed)
            arms = state_probe(seed, inter)
            deltas = bdi_delta_per_arm(seed, inter)
            records.append({"task": task, "seed_id": seed["seed_id"],
                            "interventions": inter, "arms": arms,
                            "bdi_deltas": deltas})
            print(f"[{task}] {seed['seed_id']} done", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "matched_probe.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1))
    print(f"写入 {OUT_DIR / 'matched_probe.json'}（{len(records)} 种子）")
    # 方向性摘要
    import numpy as np
    print("\n=== 方向性摘要（三臂均值）===")
    for metric in ["valence_delta", "arousal_delta", "bdi_total_abs"]:
        vals = {k: [] for k in ("generic", "evidence", "pressure")}
        for rec in records:
            for k, arm in rec["arms"].items():
                if metric == "valence_delta":
                    vals[k].append(arm["emotion_after"][0] - arm["emotion_before"][0])
                elif metric == "arousal_delta":
                    vals[k].append(arm["emotion_after"][1] - arm["emotion_before"][1])
                else:
                    vals[k].append(sum(abs(d) for d in rec["bdi_deltas"][k].values()))
        print(metric, {k: round(float(np.mean(v)), 3) for k, v in vals.items()})


if __name__ == "__main__":
    main()
