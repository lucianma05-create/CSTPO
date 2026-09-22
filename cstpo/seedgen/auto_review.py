"""种子草稿自动化初审：按用户已审阅意见提炼的规则检查剩余种子。

规则来源（2026-09-14 用户人工审阅 20 个种子的批注）：
R1 语义分工：belief 必须是主观命题性态度；纯客观事件（出轨/分手/标价/对方出价）、
   瞬时状态（"我今天还好"）不应作为 belief；对事实的纯重述性评价需标记。
R2 价格信息归属：要价/标价事实 → persona.item_facts/context；价格态度/底线 → belief/desire。
R3 初始 intention：CB 中把前缀真实报价写成初始 intention 属开放问题，标记待讨论。
R4 截断与覆盖：前缀过短、BDI 全空、persona 无有效信息的种子标记。
R5 人称：BDI content 应为用户第一人称；第三人称表述标记。
R6 证据一致性：BDI 项的 evidence 必须真实支撑内容，不得越界。

结果写入 seed["review"]["custom"]["auto_review"]（程序化检查 + LLM 检查），
不动 review_notes/checklist（人工区）。已人工审阅（有 review_date）的种子跳过。

用法：cd CSTPO && python -m cstpo.seedgen.auto_review --workers 30
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient, StructuredCallError

SEEDS = ROOT / "data" / "seeds_draft"
TASKS = ("esconv", "p4g", "craigslistbargain")

REVIEW_SYSTEM = """You are an automated first-pass reviewer for dialogue-simulation seed drafts.
Check the seed against the following rules distilled from human review feedback,
and return exactly one JSON object with findings (empty arrays when clean):

R1 (semantic division): a Belief must be a SUBJECTIVE propositional attitude the user
holds (e.g. "small donations have limited impact"). These must NOT be beliefs:
- pure objective events (infidelity, breakup, accepting an invitation, family relations);
- facts directly readable from context (listed price, the buyer's offer amount);
- transient states ("I'm fine today").
Objective facts belong in persona/context; an evaluative restatement of a bare fact
(e.g. "the buyer's offer of 2500 is too low" whose only evidence is the offer itself)
should be flagged as "restatement-like belief".

R2 (price attribution): asking/listing price facts -> persona.item_facts/context;
price attitudes/limits ("unwilling to sell below 600") -> belief/desire.

R3 (initial intention): in craigslist bargaining, writing the user's real offer from the
prefix as an initial Intention is an OPEN design question - flag it, do not modify.

R5 (person): BDI content must be first-person user statements. Third-person phrasing
("the user has set ...", "he/she ...") must be flagged.

R6 (evidence): every BDI item's evidence must genuinely support its content; evidence
citing messages beyond the prefix or not supporting the claim must be flagged.

Output schema:
{{
  "belief_issues": [{{"where": "belief[0]", "issue": "...", "suggestion": "..."}}],
  "persona_issues": [{{"where": "persona[1]", "issue": "...", "suggestion": "..."}}],
  "intention_issues": [{{"where": "intention[0]", "issue": "...", "suggestion": "..."}}],
  "cutoff_issues": [{{"issue": "...", "suggestion": "..."}}],
  "person_issues": [{{"where": "belief[1]", "issue": "...", "suggestion": "..."}}],
  "evidence_issues": [{{"where": "desire[0]", "issue": "...", "suggestion": "..."}}]
}}"""


def review_payload(seed: dict) -> str:
    bdi = seed["initial_bdi"]
    prefix = [f"[{p['original_index']}] {p['role']}: {p['text_en']}"
              for p in seed["context"]["prefix"]]
    obj = {
        "task": seed["task"]["task_id"],
        "persona_facts": [{"i": i, "category": f.get("category"),
                           "fact": f.get("fact_en"), "evidence": f.get("evidence")}
                          for i, f in enumerate(seed["persona"]["facts"])],
        "beliefs": [{"i": i, "content": b.get("content_en"),
                     "evidence": b.get("evidence")}
                    for i, b in enumerate(bdi["beliefs"])],
        "desires": [{"i": i, "content": d.get("content_en"),
                     "evidence": d.get("evidence"), "source_kind": d.get("source_kind")}
                    for i, d in enumerate(bdi["desires"])],
        "intentions": [{"i": i, "content": it.get("content_en"),
                        "evidence": it.get("evidence"), "source_kind": it.get("source_kind")}
                       for i, it in enumerate(bdi["intentions"])],
        "prefix_messages": prefix,
        "cutoff_rule": seed["context"]["cutoff"].get("rule", ""),
    }
    return json.dumps(obj, ensure_ascii=False, indent=1)


def programmatic_checks(seed: dict) -> dict:
    """确定性检查（无需 LLM）：前缀长度、BDI 覆盖、人称、证据越界。"""
    out = {"prefix_too_short": None, "bdi_empty": [], "third_person": [],
           "evidence_out_of_range": []}
    n = len(seed["context"]["prefix"])
    if n <= 1:
        out["prefix_too_short"] = f"前缀仅 {n} 条消息，信息量可能不足"
    for kind in ("beliefs", "desires", "intentions"):
        items = seed["initial_bdi"][kind]
        if not items:
            out["bdi_empty"].append(kind)
        for i, it in enumerate(items):
            if re.match(r"^(the user|he\b|she\b|they\b)", it["content_en"], re.I):
                out["third_person"].append(f"{kind}[{i}]")
            ev = str(it.get("evidence", ""))
            m = re.findall(r"prefix msg (\d+)", ev)
            for idx in m:
                if int(idx) >= len(seed["context"]["prefix"]):
                    out["evidence_out_of_range"].append(
                        f"{kind}[{i}] 引用 prefix msg {idx} 超出前缀范围")
    return out


def review_one(args) -> dict:
    seed, llm = args
    prog = programmatic_checks(seed)
    try:
        llm_out = llm.chat_json(
            [{"role": "system", "content": REVIEW_SYSTEM},
             {"role": "user", "content": review_payload(seed)}], max_tok=1500)
    except StructuredCallError:
        llm_out = {"_fallback": "review"}
    result = {
        "reviewer": "auto-llm (deepseek-flash)",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rules": ["R1 语义分工(客观事实不入信念)", "R2 价格信息归属",
                  "R3 初始intention开放问题", "R4 截断与覆盖",
                  "R5 第一人称", "R6 证据一致性"],
        "programmatic": prog,
        "llm_findings": llm_out,
    }
    seed["review"].setdefault("custom", {})["auto_review"] = result
    return seed, len(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--force", action="store_true",
                    help="已人工审阅的种子也重跑")
    args = ap.parse_args()
    targets = []
    for task in TASKS:
        for p in sorted((SEEDS / task).glob("*.json")):
            if p.name == "manifest.json":
                continue
            seed = json.loads(p.read_text())
            if not args.force and seed["review"].get("review_date"):
                continue
            targets.append((task, p, seed))
    print(f"待初审种子: {len(targets)} 个（跳过已人工审阅的）")

    llms = [LLMClient() for _ in targets]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(review_one, list(zip(
            [s for _, _, s in targets], llms))))
    total_findings = 0
    for (task, p, _), (seed, size) in zip(targets, results):
        p.write_text(json.dumps(seed, ensure_ascii=False, indent=2) + "\n")
        prog = seed["review"]["custom"]["auto_review"]["programmatic"]
        n_llm = sum(len(v) for k, v in
                    seed["review"]["custom"]["auto_review"]["llm_findings"].items()
                    if isinstance(v, list))
        total_findings += n_llm + sum(len(v) for v in prog.values() if isinstance(v, list))
    print(f"完成，写入 review.custom.auto_review")
    print(f"LLM 总调用 {sum(c.calls for c in llms)} 次 | "
          f"{sum(c.prompt_tokens + c.completion_tokens for c in llms)} tokens")


if __name__ == "__main__":
    main()
