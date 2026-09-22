"""critic 消融：v4 提取（A，无过滤）vs v4+信念分类 critic（B）。

同源控制：A/B 使用同一批种子与同一次提取结果（A 直接取 data/seeds_draft 现状），
差异仅在 B 多一步 critic。评测双口径：
1. flash auto_review 的 belief_issues 计数（与提取同模型，可能共偏）；
2. deepseek-v4-pro 独立判定（不同模型）：对每条剩余 belief 判"是否主观命题态度"，
   报告每条件的不合格率与合格产出量（合格率才是关键：只砍数量不算有用）。

用法：cd CSTPO && python -m cstpo.seedgen.ablate_critic --judge-subset 10
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient, StructuredCallError

from cstpo.seedgen import auto_review as ar_mod

SEEDS = ROOT / "data" / "seeds_draft"
ABL = ROOT / "data" / "ablation_critic"
TASKS = ("esconv", "p4g", "craigslistbargain")

CRITIC_SYSTEM = """You are a strict classifier for extracted BDI belief items.
Classify each belief into exactly one category:
- "attitude": a SUBJECTIVE propositional attitude/judgment the user holds
  (e.g. "small donations have limited impact", "I cannot trust people anymore").
- "emotion": a transient emotional expression ("I feel so negative", "I am nervous").
- "fact": an objective event or context fact (breakup, listed price, counterparty offer).
- "intention": an action tendency ("I will work to live up to it").
- "other": anything else.
Only "attitude" items may stay as beliefs. Return exactly one JSON object:
{"items": [{"index": 0, "category": "attitude", "reason": "brief"}], ...}"""

CRITIC_USER = """Prefix context (for reference):
{prefix}

Beliefs to classify:
{beliefs}

Return {{"items": [{{"index": 0, "category": "attitude|emotion|fact|intention|other",
"reason": "<=15 words"}}]}}"""

JUDGE_SYSTEM = """You are an independent evaluator of dialogue-simulation seed quality.
For each belief item below, judge whether it is a SUBJECTIVE propositional
attitude/judgment the user holds (a valid belief for a cognitive model).
NOT valid: transient emotions ("I feel so negative"), objective events/facts
(breakup, listed prices, counterparty offers), action tendencies (intentions),
or transient states ("I'm fine today").
Return exactly one JSON object:
{"verdicts": [{"index": 0, "valid": true|false, "reason": "<=15 words"}]}"""

JUDGE_USER = """Beliefs:
{beliefs}

Return {{"verdicts": [{{"index": 0, "valid": true, "reason": "..."}}]}}"""


def load_seeds(task: str) -> list[tuple[Path, dict]]:
    out = []
    for p in sorted((SEEDS / task).glob("*.json")):
        if p.name == "manifest.json":
            continue
        out.append((p, json.loads(p.read_text())))
    return out


def critic_one(args, keep_categories: set) -> tuple:
    """按 keep_categories 过滤（A 无 critic；B 只留 attitude；
    C 只删 emotion/fact，即留 attitude+intention+other）。"""
    seed, llm = args
    beliefs = seed["initial_bdi"]["beliefs"]
    if not beliefs:
        return seed, []
    prefix = "\n".join(f"[{m['original_index']}] {m['role']}: {m['text_en'][:100]}"
                       for m in seed["context"]["prefix"][-12:])
    try:
        out = llm.chat_json(
            [{"role": "system", "content": CRITIC_SYSTEM},
             {"role": "user", "content": CRITIC_USER.format(
                 prefix=prefix,
                 beliefs=json.dumps([{"index": i, "content": b["content_en"]}
                                     for i, b in enumerate(beliefs)],
                                    ensure_ascii=False, indent=1))}],
            max_tok=1200)
    except StructuredCallError:
        return seed, [{"index": i, "category": "error", "reason": "critic fallback"}
                      for i in range(len(beliefs))]
    items = out.get("items", [])
    keep, labels, downgraded = [], [], []
    for i, b in enumerate(beliefs):
        cat = next((x.get("category") for x in items
                    if int(x.get("index", -1)) == i), "other")
        labels.append({"index": i, "content": b["content_en"][:60],
                       "category": cat, "reason": ""})
        if cat in keep_categories:
            keep.append(b)
        else:
            downgraded.append({"content": b["content_en"][:60], "category": cat})
    seed["initial_bdi"]["beliefs"] = keep
    seed["review"]["custom"]["critic_labels"] = labels
    seed["review"]["custom"]["critic_dropped"] = downgraded
    return seed, labels


def auto_review_seeds(seed_list, llm_pool):
    with ThreadPoolExecutor(max_workers=len(llm_pool)) as ex:
        return list(ex.map(ar_mod.review_one, list(zip(seed_list, llm_pool))))


def judge_one(args):
    """v4-pro 独立判定：返回 (valid_count, total, verdicts)。"""
    seed, llm = args
    beliefs = seed["initial_bdi"]["beliefs"]
    if not beliefs:
        return 0, 0, []
    try:
        out = llm.chat_json(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": JUDGE_USER.format(
                 beliefs=json.dumps([{"index": i, "content": b["content_en"]}
                                     for i, b in enumerate(beliefs)],
                                    ensure_ascii=False, indent=1))}],
            max_tok=1200)
    except StructuredCallError:
        return 0, len(beliefs), []
    verdicts = out.get("verdicts", [])
    n_valid = sum(1 for v in verdicts if v.get("valid") is True)
    return n_valid, len(beliefs), verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--judge-subset", type=int, default=10,
                    help="每任务独立判定（v4-pro）的种子数")
    args = ap.parse_args()

    CONDITIONS = [("A", None), ("B", {"attitude"}),
                  ("C", {"attitude", "intention", "other"})]
    by_cond: dict[str, list] = {"A": [], "B": [], "C": []}
    for task in TASKS:
        for name, _ in CONDITIONS:
            (ABL / name / task).mkdir(parents=True, exist_ok=True)
        seeds = load_seeds(task)
        for p, s in seeds:
            (ABL / "A" / task / p.name).write_text(
                json.dumps(s, ensure_ascii=False, indent=2) + "\n")
        by_cond["A"].extend([s for _, s in seeds])
        for name, keep in CONDITIONS[1:]:
            cond_seeds = [copy.deepcopy(s) for _, s in seeds]
            llms = [LLMClient() for _ in cond_seeds]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(critic_one, list(zip(cond_seeds, llms)),
                                      [keep] * len(cond_seeds)))
            cond_seeds = [r[0] for r in results]
            for (p, _), s in zip(seeds, cond_seeds):
                (ABL / name / task / p.name).write_text(
                    json.dumps(s, ensure_ascii=False, indent=2) + "\n")
            by_cond[name].extend(cond_seeds)
            print(f"[{task}] {name} 写入完成 | critic 调用 {sum(c.calls for c in llms)} 次")

    # 口径 1：flash auto_review
    print("\n=== 口径 1：flash auto_review（同模型，共偏需打折） ===")
    for name, seeds in by_cond.items():
        llms = [LLMClient() for _ in seeds]
        auto_review_seeds(seeds, llms)
        n = sum(len((s["review"]["custom"].get("auto_review", {})
                     .get("llm_findings", {}) or {}).get("belief_issues", []))
                for s in seeds)
        tot = sum(len(s["initial_bdi"]["beliefs"]) for s in seeds)
        print(f"{name}: belief_issues={n} | 剩余 beliefs={tot} | 每信念问题率 {n/max(tot,1):.2f}")

    # 口径 2：v4-pro 独立判定（子集：每任务每隔 n/subset 取一个）
    print("\n=== 口径 2：deepseek-v4-pro 独立判定（子集） ===")
    for name, seeds in by_cond.items():
        subset = []
        for task in TASKS:
            task_seeds = [s for s in seeds if s["provenance"]["dataset"] == task]
            step = max(1, len(task_seeds) // args.judge_subset)
            subset.extend(task_seeds[::step][:args.judge_subset])
        judges = [LLMClient(model="deepseek-v4-pro") for _ in subset]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(judge_one, list(zip(subset, judges))))
        n_valid = sum(r[0] for r in results)
        n_tot = sum(r[1] for r in results)
        print(f"{name}: 判定 beliefs={n_tot} | 合格={n_valid} | "
              f"不合格率 {(n_tot-n_valid)/max(n_tot,1):.1%} | "
              f"合格产出/种子 {n_valid/len(subset):.2f}")


if __name__ == "__main__":
    main()
