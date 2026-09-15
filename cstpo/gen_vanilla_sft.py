"""香草子集生成（SFT_SPEC 步骤 ③）：vanilla 画像对话 + flash 回合标签。

- 用 data/seeds_vanilla 的种子（与冻结种子源对话零重叠）跑 vanilla 画像
  对话（自由结束 + 15 轮 judger），产物 data/sft/vanilla_dialogues/<task>/。
- 对每条 agent 回合用 flash 补策略标签（label_maps 收敛类；补标规则 03§4：
  只读当时可见历史 + 当前 actor 话语，不读用户下一回复或终局）。
- 标签存 dialogue 文件 turns 附字段 label；无把握返回 <unlabeled>。

用法：cd CSTPO && python -m cstpo.gen_vanilla_sft --workers 120
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient, StructuredCallError

from cstpo.agent import agent_turn
from cstpo.cost_ledger import RecordingLLM
from cstpo.label_maps import (CB_CANONICAL, ESCONV_CANONICAL, P4G_CANONICAL)
from cstpo.stale_judger import check_stale, should_check
from cstpo.task_env import TaskEnv, prefix_turns

SEEDS = ROOT / "data" / "seeds_vanilla"
OUT = ROOT / "data" / "sft" / "vanilla_dialogues"
SAFETY_CAP = 30

LABEL_SETS = {"esconv": ESCONV_CANONICAL, "p4g": P4G_CANONICAL,
              "craigslistbargain": CB_CANONICAL}

LABELER_SYSTEM = """You annotate ONE strategy label for ONE assistant utterance in a
dialogue. You may ONLY read the conversation history UP TO AND INCLUDING the
assistant utterance being labeled (nothing after it). Choose the single most
appropriate label from the given set. If none fits or you are unsure, return
null. Return exactly one JSON object: {"label": "..."} or {"label": null}"""


def label_turn(args):
    task, history, utter, llm = args
    labels = LABEL_SETS[task]
    ctx = "\n".join(f"{m['role']}: {m['text']}" for m in history[-12:])
    try:
        out = llm.chat_json(
            [{"role": "system", "content": LABELER_SYSTEM},
             {"role": "user", "content": (
                 f"Task: {task}\nValid labels: {labels}\n\nConversation so far:\n"
                 f"{ctx}\n\nAssistant utterance to label:\n{utter}")}],
            max_tok=100)
    except StructuredCallError:
        return None
    lab = out.get("label")
    return lab if lab in labels else None


def run_dialogue(args):
    task, seed, llm, lllm = args
    env = TaskEnv(llm=llm, max_turns=SAFETY_CAP)
    cp = env.reset(seed)["checkpoint"]
    pre = prefix_turns(seed)
    turns = list(pre)
    r = None
    n = 0
    reason = None
    while not (r and r["terminated"]):
        if should_check(n):
            j = check_stale(LLMClient(), turns, task)
            if j["stale"] and j["final_line"]:
                turns.append({"role": "user", "text": j["final_line"]})
                reason = "judger_stale_end"
                break
        utter = agent_turn(llm, task, "vanilla", turns, seed)
        r = env.step(cp, utter)
        cp = r["checkpoint"]
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": r["user_reply"]})
        n += 1
    # 补标 agent 回合（只读可见历史）
    labeled = 0
    for i in range(len(pre), len(turns)):
        if turns[i]["role"] != "assistant":
            continue
        lab = label_turn((task, turns[:i + 1], turns[i]["text"], lllm))
        turns[i]["label"] = lab or "<unlabeled>"
        if lab:
            labeled += 1
    n_agent = len([t for t in turns[len(pre):] if t["role"] == "assistant"])
    return {"task": task, "seed_id": seed["seed_id"],
            "termination": reason or r["termination_reason"],
            "n_turns": n, "prefix_n": len(pre), "turns": turns,
            "labeled": labeled, "n_agent": n_agent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--n", type=int, default=0, help="每任务种子数（0=全部）")
    args = ap.parse_args()
    jobs = []
    for task in ("esconv", "p4g", "craigslistbargain"):
        seeds = [json.loads(p.read_text()) for p in sorted(
            (SEEDS / task).glob("*.json")) if p.name != "manifest.json"]
        if args.n:
            seeds = seeds[:args.n]
        done = {p.stem for p in (OUT / task).glob("*.json")} if (OUT / task).exists() else set()
        for seed in seeds:
            if seed["seed_id"] in done:
                continue
            jobs.append((task, seed))
    print(f"生成 {len(jobs)} 条 vanilla 对话（workers={args.workers}）", flush=True)
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_dialogue, (t, s, RecordingLLM(), LLMClient())): (t, s)
                for t, s in jobs}
        for k, fut in enumerate(as_completed(futs)):
            t, s = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [跳过] {t}/{s['seed_id']}: {type(e).__name__}", flush=True)
                continue
            d = OUT / t
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{s['seed_id']}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1) + "\n")
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  已落盘 {n_ok}/{len(jobs)}", flush=True)
    print(f"完成 {n_ok}/{len(jobs)} → {OUT}")


if __name__ == "__main__":
    main()
