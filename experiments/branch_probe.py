"""Bounded single-turn diagnostic; no RL, reward model, or simulator modifications.

python experiments/branch_probe.py --env-file /path/to/.env --out experiments/results/branch_probe
python experiments/branch_probe.py --summarize experiments/results/branch_probe
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from audit_cogsim import fingerprint
from evaluation.state_transition import rj_violations
from simulator.llm import LLMClient
from simulator.run_sim import SCENARIOS, build_state
from simulator.simulator import UserSimulator
from simulator.state.schema import MAX_ITEMS

# Handcrafted interventions, NOT canonical dataset labels or samples from a trained actor.
# Statements about donations/products are fictional premises in these synthetic demos.
STIMULI = {
    "donation": {
        "concrete_impact": [
            "这个项目会公开每笔钱的用途，五元也能为一个孩子提供一顿饭，小额捐赠同样有具体作用。",
            "项目账目可以查询，即使只捐五元，也能给一个孩子增加一顿饭，小额帮助也能落到实处。",
        ],
        "reduce_commitment": [
            "可以把捐赠限定在五元以内，只做这一次，不需要长期承诺，也不必影响自己的日常开销。",
            "这次可以只考虑五元的一次性捐赠，没有后续固定扣款，先保留好自己生活所需的预算。",
        ],
        "social_norm": [
            "不少和你一样的普通上班族会从五元开始参与公益，大家一起出一点，已经成了常见的做法。",
            "许多收入普通的人也会选择捐五元，和大家一起为公益出一份小小的力，这种参与很普遍。",
        ],
    },
    "bargain": {
        "firm_reservation": [
            "八十五元是我的最低价，八十元的报价我已经拒绝过，这次也不会按八十元出售。",
            "这辆车我最低只能收八十五元，之前有人出八十我没有答应，这个底价不会再降。",
        ],
        "quality_value": [
            "这辆车刹车和轮胎状态都很好，近期不用额外维修，八十五元包含的是这个车况的价值。",
            "轮胎和刹车都维护好了，买回去不用再花钱整修，所以八十五元与这辆车的质量相符。",
        ],
        "immediate_utility": [
            "按八十五元今天就能交车，你今晚可以骑走，明天开学就能直接用上，省去继续找车的时间。",
            "八十五元成交的话今天可以马上取车，明天上学不会耽误，也就不需要再花时间联系别的卖家。",
        ],
    },
    "support": {
        "small_action": [
            "可以先把任务缩小到明天给导师发一条进度消息，不要求一次解决论文，只完成这个小步骤。",
            "明天先发一条消息告诉导师当前的进展就好，把它当作独立的小任务，不必同时解决整篇论文。",
        ],
        "reframe_difficulty": [
            "目前进展不顺，只能说明眼前的问题还没找到解法，不能据此判断你完全没有解决论文的能力。",
            "现在卡住并不等于你没有能力完成论文，暂时缺少合适的办法和完全解决不了是两件不同的事。",
        ],
        "normalize_difficulty": [
            "研究过程中遇到停滞是许多研究生都会经历的阶段，这种压力和焦虑是可以理解的反应。",
            "不少研究生也会在论文阶段卡住并感到焦虑，你现在的压力属于遇到困难时常见的感受。",
        ],
    },
}


class RequestBudget:
    def __init__(self, limit):
        self.limit, self.used = limit, 0
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            if self.used >= self.limit:
                raise RuntimeError("request_budget_exhausted")
            self.used += 1


class MeteredLLM(LLMClient):
    def __init__(self, budget, model):
        super().__init__(model=model)
        sdk = self.client.with_options(timeout=45, max_retries=0)
        self.http_requests, self.http_errors, self.structured_errors = 0, 0, 0

        def create(**kwargs):
            budget.take()
            self.http_requests += 1
            try:
                return sdk.chat.completions.create(**kwargs)
            except Exception as exc:
                self.http_errors += 1
                raise RuntimeError(f"provider_request_failed:{type(exc).__name__}") from None

        # Keep the frozen chat/chat_json logic; count both initial and fallback requests.
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def chat_json(self, *args, **kwargs):
        try:
            return super().chat_json(*args, **kwargs)
        except Exception:
            self.structured_errors += 1
            raise


def gates(log):
    issues = rj_violations(SimpleNamespace(**copy.deepcopy(log)))
    before = {i["id"]: i for group in log["bdi_before"].values() for i in group}
    after = {i["id"]: i for group in log["bdi_after"].values() for i in group}
    # Parent-node alignment is deliberately conservative; new nodes are logged separately.
    for iid, original in before.items():
        current = after.get(iid)
        if current is None or any(current[k] != original[k] for k in ("type", "content", "polarity", "active")):
            issues.append(f"parent_anchor_changed:{iid}")
    for kind, maximum in MAX_ITEMS.items():
        if sum(i["active"] for i in after.values() if i["type"] == kind) > maximum:
            issues.append(f"active_capacity:{kind}")
    if log["mode"] in ("elicit", "social") and log["bdi_before"] != log["bdi_after"]:
        issues.append("noninfluence_bdi_changed")
    if any("[robustness]" in note for note in log["update_notes"]):
        issues.append("component_fallback_or_invalid_field")
    updates = log["proposed_bdi"].get("bdi_updates", [])
    ids = [u.get("id") for u in updates if isinstance(u, dict)]
    if len(ids) != len(set(ids)):
        issues.append("duplicate_update_proposal")
    return issues


def branch(job, budget, model, out):
    task, strategy, wording, repeat = job
    llm = MeteredLLM(budget, model)
    # For a non-root checkpoint also copy prev_gc and conversation_ended explicitly.
    state = build_state(copy.deepcopy(SCENARIOS[task]))
    sim = UserSimulator(state, llm=llm, route_mode="deterministic", debug=False)
    row = dict(task=task, strategy=strategy, wording=wording, repeat=repeat,
               parent_sha256=hashlib.sha256(json.dumps(asdict(state), sort_keys=True).encode()).hexdigest(),
               assistant_reply=STIMULI[task][strategy][wording], issues=[])
    try:
        sim.simulate_turn(row["assistant_reply"])
        row["log"] = asdict(sim.logs[-1])
        row["issues"] = gates(row["log"])
    except Exception as exc:
        row["issues"].append(f"branch_error:{type(exc).__name__}")
    if llm.http_errors or llm.structured_errors:
        row["issues"].append("provider_or_structured_call_failure")
    row["usage"] = {key: getattr(llm, key) for key in (
        "http_requests", "http_errors", "structured_errors", "calls", "prompt_tokens",
        "completion_tokens", "parse_errors", "repair_attempts", "repair_successes")}
    row["valid"] = not row["issues"]
    (out / f"{task}__{strategy}__{wording}__{repeat}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    return row


def vector(row, block):
    log = row["log"]
    if block == "emotion":
        return [(log["emotion_after"][key] - log["emotion_before"][key]) / scale / math.sqrt(2)
                for key, scale in (("valence", 2), ("arousal", 1))]
    b = {i["id"]: i for group in log["bdi_before"].values() for i in group}
    a = {i["id"]: i for group in log["bdi_after"].values() for i in group}
    return [(a[iid]["strength"] - b[iid]["strength"]) / 4 / math.sqrt(len(b)) for iid in sorted(b)]


def mean(xs):
    return [sum(column) / len(xs) for column in zip(*xs)]


def distance2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def components(rows, block):
    """Balanced nested descriptive ANOVA; negative estimates retained, not significance tests."""
    keys = sorted({row["strategy"] for row in rows})
    cells = {(k, w): [vector(r, block) for r in rows if r["strategy"] == k and r["wording"] == w]
             for k in keys for w in range(2)}
    if len(keys) != 3 or any(len(cell) != 2 for cell in cells.values()):
        return {"available": False, "reason": "requires all 3 x 2 x 2 valid observations"}
    means = {key: mean(value) for key, value in cells.items()}
    strategy_means = {k: mean([means[k, w] for w in range(2)]) for k in keys}
    grand = mean(list(strategy_means.values()))
    noise = sum(distance2(x, means[key]) for key, xs in cells.items() for x in xs) / 6
    ms_word = 2 * sum(distance2(means[k, w], strategy_means[k]) for k in keys for w in range(2)) / 3
    ms_strategy = 4 * sum(distance2(strategy_means[k], grand) for k in keys) / 2
    return dict(available=True, repeat_noise=noise, wording_component=(ms_word - noise) / 2,
                strategy_component=(ms_strategy - ms_word) / 4,
                raw_between_strategy_mean_square=ms_strategy,
                note="Fixed handcrafted strategy set, one parent; signed moment estimates, no population inference")


def summarize(out):
    manifest = json.loads((out / "manifest.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*__*.json"))]
    result = dict(completed=len(rows), planned=manifest["planned_branches"],
                  valid=sum(r["valid"] for r in rows),
                  usage={key: sum(r["usage"][key] for r in rows) for key in (
                      "http_requests", "http_errors", "prompt_tokens", "completion_tokens")}, tasks={})
    for task in manifest["tasks"]:
        subset = [r for r in rows if r["task"] == task]
        valid = [r for r in subset if r["valid"]]
        result["tasks"][task] = dict(
            completed=len(subset), valid=len(valid),
            modes={m: sum(r.get("log", {}).get("mode") == m for r in subset)
                   for m in ("influence", "elicit", "social")},
            bdi_anchors=components(valid, "bdi"), emotion=components(valid, "emotion"),
            influence_only_bdi=components([r for r in valid if r["log"]["mode"] == "influence"], "bdi"),
            branches=[dict(strategy=r["strategy"], wording=r["wording"], repeat=r["repeat"],
                           issues=r["issues"], mode=r.get("log", {}).get("mode"),
                           route=r.get("log", {}).get("route"), judgment=r.get("log", {}).get("judgment"))
                      for r in subset])
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def self_check():
    rows = []
    for k in range(3):
        for w in range(2):
            for r in range(2):
                rows.append(dict(strategy=str(k), wording=w, repeat=r, log=dict(
                    emotion_before=dict(valence=0, arousal=0),
                    emotion_after=dict(valence=float(k - 1), arousal=0))))
    result = components(rows, "emotion")
    assert result["repeat_noise"] == 0 and result["wording_component"] == 0
    assert math.isclose(result["strategy_component"], 0.125)
    assert not components(rows[:-1], "emotion")["available"]
    print("Variance sanity checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/branch_probe")
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=list(STIMULI), default=list(STIMULI))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-requests", type=int, default=300)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.summarize:
        result = summarize(args.summarize)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == "DEEPSEEK_API_KEY":
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Provide DEEPSEEK_API_KEY or --env-file; credentials are never saved")
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = [(task, strategy, wording, repeat) for task in args.tasks
            for strategy in STIMULI[task] for wording in range(2) for repeat in range(2)]
    random.Random(20260913).shuffle(jobs)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), model=args.model,
                    tasks=args.tasks, planned_branches=len(jobs), max_http_requests=args.max_requests,
                    workers=3, source_sha256=fingerprint(), stimuli=STIMULI, execution_order=jobs,
                    sampling="temperature=0; deterministic route; backend output can still vary",
                    limitations="Synthetic demo roots, handcrafted interventions, no terminal returns or human references")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    budget = RequestBudget(args.max_requests)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(branch, job, budget, args.model, args.out) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            print(f"{row['task']}/{row['strategy']}/{row['wording']}/{row['repeat']} "
                  f"valid={row['valid']} requests={row['usage']['http_requests']}", flush=True)
    result = summarize(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
