"""种子草稿生成：程序化 envelope（seed_adapter）+ LLM 语义提取 + 中文渲染。

用法：
    cd CSTPO && python -m cstpo.generate_seeds --task esconv --n 30
    （三任务全跑：--task all）

产出：
    data/seeds_draft/<task>/<seed_id>.json    种子草稿（含 review.review_notes 自定义字段）
    data/seeds_draft/review_zh/<task>.md      中文审阅版（人工预审用）
    data/seeds_draft/<task>/manifest.json     生成设置与 cutoff 统计

模型 deepseek-flash（用户已授权 API 调用）；提取与翻译均为草稿产物，
供 02§10 人工预审，不构成正式种子。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient, StructuredCallError

from cstpo.seed_adapter import (CHECKLIST, DEFAULT_PROFILE, EMOTION_MAP_V1,
                                sample_seeds)

OUT = ROOT / "data" / "seeds_draft"

EXTRACT_SYSTEM = """You are an initialization annotator converting raw dialogue data into
a draft user cognitive state for a simulator. Strict rules (revised after human
pre-review round 1):

1. Use ONLY the provided allowed material (situation, questionnaire, prefix dialogue
   up to the cutoff). Never use anything after the cutoff or any outcome labels.
   State-at-cutoff: the extracted BDI is the user's cognitive state AT THE CUTOFF —
   AFTER all prefix interactions shown. It must reflect any attitude changes
   evidenced within the prefix (if the user softened or hardened a stance during
   the prefix, extract the softened/hardened stance, NOT the hypothetical
   pre-conversation state). The simulated conversation continues FROM this state.
2. Persona facts are relatively STABLE background facts, constraints, experiences,
   stable values/preferences, or evidenced communication style. Do NOT write current
   beliefs/intentions/emotions into persona. Do not invent age/gender/income/job.
   Do NOT dump raw questionnaire numeric codes (e.g. "income level 2") into persona
   facts — only qualitative, task-relevant renderings with field evidence.
3. BDI items MUST be first-person statements ("I ...", "My ..."). Third-person
   phrasing is rejected.
   - Belief: a SUBJECTIVE propositional attitude the user holds (need not be
     objectively true). NOT a belief: objective events (breakup, infidelity,
     accepting an invitation, family relations), facts directly stated in the
     prefix (listed prices, the counterparty's offer amounts), or transient
     states ("I'm fine today"). An evaluative statement whose only support is
     the bare fact it restates must be OMITTED.
     REV v4 hard exclusions:
     (a) transient emotional expressions ("I feel so negative", "I am a little
         nervous") are NOT beliefs — report them in emotion_candidates instead;
     (b) pure narrative/causal facts are NOT beliefs — persona or omit;
     (c) action tendencies ("I will work to live up to it") are NOT beliefs —
         they belong to intentions.
   - Keep at most 4 beliefs per seed; if more candidates exist, keep the most
     central ones (core=true first, then highest strength).
   - Desire: an outcome the user wants to achieve or avoid (polarity: approach/avoid).
   - Intention: a concrete action the user is inclined/committed to take; may be empty.
   - strength ONLY from {1.0, 2.0, 3.0}: 1.0 weak/hesitant, 2.0 plain statement,
     3.0 strong/emphatic/repeated. Never output other decimals. If a fact is worth
     writing but no strength evidence exists, use 2.0.
   - core: true for items central to the user's situation; false for peripheral details.
   - evidence: cite the specific prefix message index AND a short quote fragment
     (e.g. "prefix msg 2: '...'"), or "situation: <fragment>". A bare "situation"
     with no fragment is insufficient.
   - Do not create duplicate items. Leave lists EMPTY when there is no evidence.
4. Output exactly one JSON object."""

EXTRACT_ESCONV_USER = """Task: emotional support conversation (user=seeker, actor=supporter).

Situation: {situation}
Problem type: {problem_type}
Experience type: {experience_type}

Prefix dialogue (user's own words up to cutoff; indices are 0-based;
the LAST message marks the cutoff — extract the state as it stands at that moment):
{prefix}

Return:
{{
  "persona_facts": [{{"category": "background|constraints|experience|values|style",
                      "fact_en": "...", "evidence": "..."}}],
  "beliefs": [{{"content_en": "first-person proposition", "strength": 2.0,
                "core": true, "evidence": "..."}}],
  "desires": [{{"content_en": "...", "strength": 2.0, "core": true,
                "polarity": "approach|avoid", "source_kind": "observed|inferred",
                "evidence": "..."}}],
  "intentions": [{{"content_en": "...", "strength": 2.0, "core": true,
                   "polarity": "approach|avoid", "source_kind": "observed|inferred",
                   "evidence": "..."}}]
}}
Note: a belief like "I cannot solve this at all" may be a symptom of distress —
write it as a belief only if the user genuinely holds it as a proposition, and mark
core=true. Objective events in the situation (breakup, infidelity, job facts) belong
to persona facts, NOT beliefs — a belief must be the user's subjective attitude or
judgment about such facts (e.g. "I will never trust anyone again"). If the user has
not expressed any intention, leave intentions empty.
TASK CONSTRAINT (Q1 ruling): intentions must NOT contain coping/relief actions that
realize the task goal — seeing a therapist, diverting concentration, taking time for
oneself, seeking help, etc. The whole point of this conversation is to move the user
toward such actions; a seed that already contains them has its task value broken.
Counterfactuals ("if I could change things, I would...") and tentative musings
("maybe I should...") are NOT intentions — only present-tense commitments count."""

EXTRACT_P4G_USER = """Task: persuade-donation (user=persuadee, actor=persuader).

Persuadee questionnaire (task-pre measurements, user-side only):
{scales}

Prefix dialogue (up to cutoff; indices 0-based; the LAST message marks the
cutoff — extract the state as it stands at that moment):
{prefix}

Return:
{{
  "persona_facts": [{{"category": "background|constraints|experience|values|style",
                      "fact_en": "...", "evidence": "..."}}],
  "beliefs": [{{"content_en": "...", "strength": 2.0, "core": true,
                "evidence": "..."}}],
  "desires": [{{"content_en": "...", "strength": 2.0, "core": true,
                "polarity": "approach|avoid", "source_kind": "observed|inferred",
                "evidence": "..."}}],
  "intentions": [{{"content_en": "...", "strength": 2.0, "core": true,
                   "polarity": "approach|avoid", "source_kind": "observed|inferred",
                   "evidence": "..."}}],
  "emotion_candidates": [{{"category": "anxiety|sadness|frustration|anger|interest|satisfaction|neutral",
                           "evidence": "prefix msg N", "is_latest": true}}]
}}
Rules: high care/agreeable scores do NOT equal trusting a charity or planning to
donate. Do not convert questionnaire numbers into beliefs about the current charity.
Do NOT dump numeric questionnaire codes into persona facts — only qualitative,
task-relevant renderings (e.g. "values helping others", evidence: benevolence.x).
Only prefix evidence supports B/D/I about the current conversation. Leave empty
when no evidence.
TASK CONSTRAINT (Q1 ruling): intentions must NOT contain donation commitments or
specific donation amounts ("I'll donate 5 dollars") — a donation commitment is the
terminal task goal. Even if the prefix shows such a statement, do NOT write it as
an intention."""

EXTRACT_CB_USER = """Task: craigslist bargaining (user=seller, actor=buyer).

Item the seller is selling (seller-visible):
{item_facts}

Prefix dialogue and events up to cutoff (indices 0-based; events are structured
actions, not utterances; the LAST message marks the cutoff — extract the state as
it stands at that moment):
{prefix}

Return:
{{
  "persona_facts": [{{"category": "background|constraints|experience|values|style",
                      "fact_en": "...", "evidence": "..."}}],
  "beliefs": [{{"content_en": "...", "strength": 2.0, "core": true,
                "evidence": "..."}}],
  "desires": [{{"content_en": "...", "strength": 2.0, "core": true,
                "polarity": "approach|avoid", "source_kind": "observed|inferred|controlled_default",
                "evidence": "..."}}],
  "intentions": [{{"content_en": "...", "strength": 2.0, "core": true,
                   "polarity": "approach|avoid", "source_kind": "observed|inferred|controlled_default",
                   "evidence": "..."}}],
  "emotion_candidates": [{{"category": "anxiety|sadness|frustration|anger|interest|satisfaction|neutral",
                           "evidence": "prefix msg N", "is_latest": true}}]
}}
Rules: the desire "sell the item at a favorable price, as close to my target as
possible" comes from the task instruction — mark source_kind=controlled_default,
strength 2.0. A listed price or a counterparty offer is a CONTEXT FACT, NOT a
belief — do not write "the price is $3000" or "the buyer's offer is $2500" as
beliefs; only genuine subjective attitudes (e.g. "my price is fair given the
condition") with evidence beyond the bare fact. Price facts belong to persona
item_facts, which are filled programmatically — do not duplicate them.
Intentions only from real offers/stated next steps in the prefix (observed).
Do not invent urgency or financial need.
TASK CONSTRAINT (Q1 ruling): intentions must NOT contain deal closure (accepting
a deal / finalizing the sale) — that is the terminal task goal. Price offers in the
prefix are process moves and may be recorded as intentions."""

TRANSLATE_USER = """Translate the following seed content into natural Simplified
Chinese for a human reviewer. Keep structure and technical terms (BDI 类型、强度、
core、极性、消息索引) unchanged; translate only the prose (situation, facts, BDI
contents, dialogue texts, task descriptions). Return exactly one JSON object with
the same keys:
{payload}"""


def extract(llm: LLMClient, task: str, seed: dict) -> dict:
    prefix_lines = []
    for p in seed["context"]["prefix"]:
        prefix_lines.append(f"[{p['original_index']}] {p['role']}: {p['text_en']}")
    prefix_txt = "\n".join(prefix_lines) or "(empty)"
    if task == "esconv":
        user = EXTRACT_ESCONV_USER.format(
            situation=seed["persona"].get("situation_en") or "(none)",
            problem_type=seed["persona"].get("problem_type") or "(none)",
            experience_type=seed["persona"].get("experience_type") or "(none)",
            prefix=prefix_txt)
    elif task == "p4g":
        scales = json.dumps(seed["persona"]["measurements"], indent=1)
        user = EXTRACT_P4G_USER.format(scales=scales, prefix=prefix_txt)
    else:
        user = EXTRACT_CB_USER.format(
            item_facts=json.dumps(seed["persona"]["item_facts"], indent=1),
            prefix=prefix_txt + "\n" +
                   json.dumps(seed["context"].get("task_events_in_prefix", []),
                              indent=1))
    try:
        return llm.chat_json([{"role": "system", "content": EXTRACT_SYSTEM},
                              {"role": "user", "content": user}], max_tok=2000)
    except StructuredCallError:
        return {"persona_facts": [], "beliefs": [], "desires": [],
                "intentions": [], "_fallback": "extract"}


def translate(llm: LLMClient, payload: dict) -> dict:
    try:
        return llm.chat_json([{"role": "user", "content":
                               TRANSLATE_USER.format(payload=json.dumps(payload,
                                                                        ensure_ascii=False))}],
                             max_tok=2000)
    except StructuredCallError:
        return {"_fallback": "translate"}


def sanitize(seed: dict) -> list[str]:
    """v2 程序化清洗：content 归一化去重、第一人称标志、证据越界标志。

    结果写入 review.custom.extraction_flags（供人工审阅参考），不静默改写内容。
    """
    flags = []
    bdi = seed["initial_bdi"]
    for kind in ("beliefs", "desires", "intentions"):
        seen, out = set(), []
        for it in bdi[kind]:
            key = re.sub(r"\s+", " ", str(it.get("content_en", "")).lower().strip())
            if key in seen:
                flags.append(f"dedup {kind}: {it.get('content_en', '')[:50]}")
                continue
            seen.add(key)
            out.append(it)
        bdi[kind] = out
        for i, it in enumerate(bdi[kind]):
            c = str(it.get("content_en", "")).strip()
            if not re.match(r"^(i\b|i'm\b|my\b|we\b)", c, re.I):
                flags.append(f"first_person {kind}[{i}]: {c[:50]}")
            ev = str(it.get("evidence", ""))
            for idx in re.findall(r"prefix msg (\d+)", ev):
                if int(idx) >= len(seed["context"]["prefix"]):
                    flags.append(f"evidence_oob {kind}[{i}]: {ev[:50]}")
    seed["review"].setdefault("custom", {})
    seed["review"]["custom"]["extraction_flags"] = flags
    return flags


BELIEF_CAP = 4  # v4：belief 软上限（02 §6 容量上限 4 的程序化执行）
INTENTION_CAP = 3  # 02 §6：活跃 I 至多 3

# Q1 裁定（2026-09-15）：初始 intention 不得包含任务终局目标的实现动作
GOAL_INTENTION_PATTERNS = {
    "esconv": [r"therapist", r"counsel", r"psychiatrist", r"medication",
               r"divert", r"distract", r"take (some )?time", r"time for myself",
               r"break from", r"seek(ing)? help", r"talk to (someone|a professional)"],
    "p4g": [r"donat"],
    "craigslistbargain": [r"accept the offer", r"finaliz", r"close the deal"],
}


# 非承诺表达（任务中立）：intention 只收当下承诺
NONCOMMIT_INTENTION_PATTERNS = [
    r"\bif i (could|had|were|can)\b",      # 反事实："if I could change things"
    r"\bmaybe i (should|will|can)\b",      # 犹豫："maybe I should..."
    r"\bi (think )?i should\b",            # 试探："I think I should..."
    r"\bi would (have|rather)\b",          # 反事实/偏好："I would have..."
]


def goal_intention_filter(seed: dict) -> None:
    """Q1 任务约束：终局目标实现动作从 intentions 中剔除并记 flag；
    非承诺表达（反事实/犹豫/试探）同样剔除——intention 只收当下承诺。

    ESConv：缓解/求助行动（看咨询师、转移注意力、给自己时间…）；
    P4G：捐赠承诺与金额；CB：成交/结束交易的承诺。"""
    patterns = GOAL_INTENTION_PATTERNS[seed["task"]["task_id"]] + \
        NONCOMMIT_INTENTION_PATTERNS
    kept, dropped = [], []
    for it in seed["initial_bdi"]["intentions"]:
        if any(re.search(p, it["content_en"], re.I) for p in patterns):
            dropped.append(it)
        else:
            kept.append(it)
    seed["initial_bdi"]["intentions"] = kept
    if dropped:
        flags = seed["review"].setdefault("custom", {}).setdefault("extraction_flags", [])
        for d in dropped:
            flags.append(f"goal_intention_drop: {d.get('content_en', '')[:60]}")


def task_completed_risk(seed: dict) -> None:
    """前缀已含任务终局目标实现证据的种子标记（供预审决定排除或保留）。"""
    task = seed["task"]["task_id"]
    risks = []
    if task == "p4g":
        for m in seed["context"]["prefix"]:
            if m["role"] == "persuadee" and re.search(r"donat|\$\d+|\d+ dollars", m["text_en"], re.I):
                risks.append(f"prefix[{m['original_index']}] 含捐赠承诺")
    elif task == "esconv":
        pats = GOAL_INTENTION_PATTERNS["esconv"]
        for m in seed["context"]["prefix"]:
            if m["role"] == "seeker" and any(re.search(p, m["text_en"], re.I) for p in pats):
                risks.append(f"prefix[{m['original_index']}] 含缓解行动表述")
    else:
        for e in seed["context"].get("task_events_in_prefix", []):
            if e["action"] == "accept":
                risks.append(f"prefix 事件[{e['index']}] accept（已成交）")
    seed["context"]["task_completed_risk"] = risks


def critic_route(seed: dict, llm: LLMClient) -> None:
    """critic+分流（2026-09-14 消融结论落地）：attitude→beliefs；
    intention→intentions（容量 3，超限丢弃并记 flag）；emotion→prefix_candidates；
    fact/other→丢弃并记 critic_dropped。"""
    from cstpo.ablate_critic import CRITIC_SYSTEM, CRITIC_USER
    beliefs = seed["initial_bdi"]["beliefs"]
    if not beliefs:
        return
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
        seed["review"].setdefault("custom", {})
        seed["review"]["custom"]["critic_labels"] = {"fallback": True}
        return
    items = out.get("items", [])
    keep, labels, routed_i, routed_e, dropped = [], [], [], [], []
    for i, b in enumerate(beliefs):
        cat = next((x.get("category") for x in items
                    if int(x.get("index", -1)) == i), "other")
        labels.append({"index": i, "content": b["content_en"][:60],
                       "category": cat})
        if cat == "attitude":
            keep.append(b)
        elif cat == "intention":
            routed_i.append(b)
        elif cat == "emotion":
            routed_e.append({"content": b["content_en"][:60],
                             "evidence": b.get("evidence", ""),
                             "source": "critic-routed"})
        else:
            dropped.append({"content": b["content_en"][:60], "category": cat})
    seed["initial_bdi"]["beliefs"] = keep
    ints = sorted(seed["initial_bdi"]["intentions"] + routed_i,
                  key=lambda x: (x.get("core", True), x.get("strength", 2.0)),
                  reverse=True)
    overflow = ints[INTENTION_CAP:]
    seed["initial_bdi"]["intentions"] = ints[:INTENTION_CAP]
    seed["initial_emotion"].setdefault("prefix_candidates", [])
    seed["initial_emotion"]["prefix_candidates"] += routed_e
    seed["review"].setdefault("custom", {})
    seed["review"]["custom"]["critic_labels"] = labels
    seed["review"]["custom"]["critic_dropped"] = dropped
    flags = seed["review"]["custom"].setdefault("extraction_flags", [])
    for it in overflow:
        flags.append(f"intention_cap_drop: {it.get('content_en', '')[:50]}")


def fill_seed(llm: LLMClient, task: str, seed: dict) -> dict:
    ex = extract(llm, task, seed)
    seed["persona"]["facts"] = ex.get("persona_facts", [])
    bdi = seed["initial_bdi"]
    for kind in ("beliefs", "desires", "intentions"):
        for item in ex.get(kind, []):
            item.setdefault("strength", 2.0)
            item.setdefault("core", True)
            item.setdefault("source_kind", "observed")
        bdi[kind] = ex.get(kind, [])
    # v4：情绪候选（前缀中表达的瞬时情绪 → 初始情绪参考）
    emo = ex.get("emotion_candidates", [])
    if isinstance(emo, list):
        seed["initial_emotion"]["prefix_candidates"] = emo
    # v4.1：critic 分类分流（消融结论：只留 attitude 作 belief，其余归位/丢弃）
    critic_route(seed, llm)
    # v4.2：Q1 任务约束（终局目标动作不入初始 intention）+ 任务完成风险标记
    goal_intention_filter(seed)
    task_completed_risk(seed)
    # v4：belief 软上限（core 优先，再按强度），超出部分丢弃并记 flag
    if len(bdi["beliefs"]) > BELIEF_CAP:
        dropped = sorted(bdi["beliefs"], key=lambda b: (b.get("core", True),
                          b.get("strength", 2.0)), reverse=True)[BELIEF_CAP:]
        bdi["beliefs"] = sorted(bdi["beliefs"], key=lambda b: (b.get("core", True),
                                b.get("strength", 2.0)), reverse=True)[:BELIEF_CAP]
        seed["review"].setdefault("custom", {})
        seed["review"]["custom"].setdefault("extraction_flags", [])
        for d in dropped:
            seed["review"]["custom"]["extraction_flags"].append(
                f"belief_cap_drop: {d.get('content_en', '')[:50]}")
    sanitize(seed)
    # 中文渲染：persona facts + BDI + 前缀 + situation
    payload = {
        "situation": seed["persona"].get("situation_en"),
        "task_description": seed["task"]["description"],
        "persona_facts": [f["fact_en"] for f in seed["persona"]["facts"]],
        "beliefs": [i["content_en"] for i in bdi["beliefs"]],
        "desires": [i["content_en"] for i in bdi["desires"]],
        "intentions": [i["content_en"] for i in bdi["intentions"]],
        "prefix": [{"index": p["original_index"], "role": p["role"],
                    "text": p["text_en"]} for p in seed["context"]["prefix"]],
    }
    zh = translate(llm, payload)
    seed["_zh"] = zh if "_fallback" not in zh else {}
    if "_zh" in seed["_zh"] and isinstance(seed["_zh"].get("situation"), str):
        seed["persona"]["situation_zh"] = seed["_zh"]["situation"]
    return seed


def review_md(task: str, seeds: list[dict], path: Path) -> None:
    name = {"esconv": "ESConv 情感支持", "p4g": "P4G 劝说捐赠",
            "craigslistbargain": "CraigslistBargain 讨价还价"}[task]
    lines = [f"# {name} 种子草稿人工预审（{len(seeds)} 个）", "",
             "> 生成：seed_adapter v1 草案（02§12 白名单 / §6 档位 / §7 cutoff）；"
             "B/D/I 由 deepseek-flash 草拟，强度仅 {1,2,3}，**均待人工复核**。"
             "`review_notes` 可写任意批注（写入 JSON 的 review 区）。", ""]
    for s in seeds:
        zh = s.get("_zh", {})
        lines += [f"## 种子 {s['seed_id']}", "",
                  f"- 来源：`{s['provenance']['source_id']}` | 分组：`{s['provenance']['group_id']}`"
                  f" | cutoff 规则：{s['context']['cutoff'].get('rule','')}"
                  f" | low_info：{s['context'].get('low_info', False)}",
                  f"- 任务：{zh.get('task_description', s['task']['description'])}",
                  f"- 认知参数：η_R={s['cognitive_profile']['eta_R']}, τ_A={s['cognitive_profile']['tau_A']},"
                  f" τ_R={s['cognitive_profile']['tau_R']}（controlled_default）",
                  f"- 初始情绪：{s['initial_emotion']['category']}"
                  f"（原始 {s['initial_emotion']['raw'].get('emotion_type','—')}"
                  f" {s['initial_emotion']['raw'].get('initial_emotion_intensity','')}）",
                  ""]
        if task == "esconv" and s["persona"].get("situation_en"):
            lines += ["### 情境", f"- 中文：{s['persona'].get('situation_zh', '（翻译失败）')}",
                      f"- 原文：{s['persona']['situation_en']}", ""]
        facts = zh.get("persona_facts") or [f.get("fact_en") for f in s["persona"]["facts"]]
        if facts:
            lines += ["### Persona 事实（草案）"]
            for i, f in enumerate(s["persona"]["facts"]):
                lines.append(f"- {facts[i] if i < len(facts) else f['fact_en']}"
                             f"　｜证据：{f.get('evidence','')}")
            lines.append("")
        for kind, label in (("beliefs", "信念 B"), ("desires", "愿望 D"),
                            ("intentions", "意图 I")):
            items = s["initial_bdi"][kind]
            if items:
                lines += [f"### {label}（草案，{len(items)} 项）", ""]
                z = zh.get(kind) or []
                for i, it in enumerate(items):
                    pol = f"/{it.get('polarity','')}" if it.get("polarity") and kind != "beliefs" else ""
                    lines.append(f"- **{it.get('strength')}** (core={it.get('core')}{pol}) "
                                 f"{z[i] if i < len(z) else it['content_en']}"
                                 f"　｜{it.get('content_en')}　｜证据：{it.get('evidence','')}")
                lines.append("")
        if s["context"]["prefix"]:
            lines += ["### 前缀对话（截止点前，中英对照）", ""]
            zprefix = {p["index"]: p.get("text") for p in zh.get("prefix", [])}
            for p in s["context"]["prefix"]:
                lines.append(f"- [{p['original_index']}] **{p['role']}**："
                             f"{zprefix.get(p['original_index'], p['text_en'])}"
                             f"　（{p['text_en'][:80]}）")
            lines.append("")
        lines += ["### 人工预审（02§10 九项检查 + 自定义批注）", "",
                  "> 在种子 JSON 的 `review.review_notes` 写文字批注，或 `review.custom` 写结构化信息；"
                  "`review.checklist` 逐项填 true/false。", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fill_one(args) -> tuple:
    """worker：每线程独立 LLMClient（usage 计数不共享），单种子异常不中断整体。"""
    task, s = args
    llm = LLMClient()
    try:
        fill_seed(llm, task, s)
        error = None
    except Exception as exc:  # 网络/限流等：标记失败，后续重跑该种子
        s["_gen_error"] = f"{type(exc).__name__}: {exc}"
        error = s["_gen_error"]
    return s, llm, error


def preserve_reviews(task: str, seeds: list[dict]) -> dict:
    """按 seed_id 保留已有人工审阅（reviewer/notes/checklist/custom 中的人工键），
    auto_review 因提取变更视为过期、不保留。"""
    saved = {}
    tdir = OUT / task
    if tdir.exists():
        for p in tdir.glob("*.json"):
            if p.name == "manifest.json":
                continue
            try:
                old = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            r = dict(old.get("review") or {})
            custom = dict(r.get("custom") or {})
            custom.pop("auto_review", None)
            custom.pop("extraction_flags", None)
            r["custom"] = custom
            saved[old.get("seed_id")] = r
    for s in seeds:
        if s["seed_id"] in saved:
            s["review"] = saved[s["seed_id"]]
    return saved


def generate(task: str, n: int, seed_rng: int = 20260914, workers: int = 1) -> dict:
    seeds = sample_seeds(task, n, seed_rng)
    saved_reviews = preserve_reviews(task, seeds)
    clients = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_fill_one, [(task, s) for s in seeds]))
        seeds = [r[0] for r in results]
        clients = [r[1] for r in results]
        errors = [r[2] for r in results if r[2]]
    else:
        errors = []
        for s in seeds:
            _, llm, err = _fill_one((task, s))
            clients.append(llm)
            if err:
                errors.append(err)
    # 汇总 usage（各 worker 独立计数）
    calls = sum(c.calls for c in clients)
    prompt = sum(c.prompt_tokens for c in clients)
    completion = sum(c.completion_tokens for c in clients)
    tdir = OUT / task
    tdir.mkdir(parents=True, exist_ok=True)
    review_md(task, seeds, OUT / "review_zh" / f"{task}.md")  # 先出审阅（用 _zh）
    for s in seeds:
        # _zh 为审阅辅助字段（中文渲染），保留在种子 JSON 中供前端/人工预审使用；
        # 模拟器侧消费时忽略该字段
        (tdir / f"{s['seed_id']}.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2) + "\n")
    low = sum(1 for s in seeds if s["context"].get("low_info"))
    manifest = {"task": task, "n": n, "seed_rng": seed_rng,
                "workers": workers,
                "generator": "seed_adapter v4.2 frozen 2026-09-15",
                "model": "deepseek-flash", "low_info": low,
                "gen_errors": errors,
                "cutoff_stats": {s["context"]["cutoff"]["rule"]:
                                 sum(1 for x in seeds
                                     if x["context"]["cutoff"]["rule"] == s["context"]["cutoff"]["rule"])
                                 for s in seeds},
                "llm_usage": f"LLM 调用 {calls} 次 | prompt {prompt} | completion {completion} | 总计 {prompt + completion}"}
    (tdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def _translate_payload(payload: dict) -> dict:
    llm = LLMClient()
    return translate(llm, payload)


def regenerate_review(task: str, workers: int = 30) -> None:
    """从已生成的 JSON 重跑翻译并重建中文审阅 MD（不重跑提取）。"""
    tdir = OUT / task
    seeds = [json.loads(p.read_text()) for p in sorted(tdir.glob("*.json"))
             if p.name != "manifest.json"]
    payloads = []
    for s in seeds:
        bdi = s["initial_bdi"]
        payloads.append({
            "situation": s["persona"].get("situation_en"),
            "task_description": s["task"]["description"],
            "persona_facts": [f["fact_en"] for f in s["persona"]["facts"]],
            "beliefs": [i["content_en"] for i in bdi["beliefs"]],
            "desires": [i["content_en"] for i in bdi["desires"]],
            "intentions": [i["content_en"] for i in bdi["intentions"]],
            "prefix": [{"index": p["original_index"], "role": p["role"],
                        "text": p["text_en"]} for p in s["context"]["prefix"]],
        })
    with ThreadPoolExecutor(max_workers=workers) as ex:
        zhs = list(ex.map(_translate_payload, payloads))
    fallback = 0
    for s, zh in zip(seeds, zhs):
        if "_fallback" in zh:
            fallback += 1
            s["_zh"] = {}
        else:
            s["_zh"] = zh
        # 持久化中文渲染到种子 JSON（审阅辅助字段）
        (tdir / f"{s['seed_id']}.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2) + "\n")
    review_md(task, seeds, OUT / "review_zh" / f"{task}.md")
    print(f"[{task}] 审阅版重建完成，翻译失败 {fallback}/{len(seeds)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["esconv", "p4g",
                     "craigslistbargain", "all"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--workers", type=int, default=1,
                    help="并发 worker 数（每 worker 独立 LLMClient）")
    ap.add_argument("--regen-review", action="store_true",
                    help="只重跑翻译并重建中文审阅 MD（不重跑提取）")
    args = ap.parse_args()
    tasks = ["esconv", "p4g", "craigslistbargain"] if args.task == "all" else [args.task]
    if args.regen_review:
        for t in tasks:
            regenerate_review(t, workers=args.workers)
        return
    for t in tasks:
        m = generate(t, args.n, args.seed, workers=args.workers)
        errs = len(m["gen_errors"])
        print(f"[{t}] {m['n']} 种子 | workers={m['workers']} | low_info {m['low_info']}"
              f" | 生成失败 {errs} | {m['llm_usage']}")


if __name__ == "__main__":
    main()
