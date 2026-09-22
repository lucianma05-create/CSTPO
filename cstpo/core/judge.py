"""终局 judge 实现（04§18 草案落地）：三任务提示 + 结构化输出 + 证据定位。

judge 只读可见轨迹（完整对话 + 允许的案例信息），不读 BDI/Emotion/critic；
输出固定 JSON schema；重试与异常按 04§18.1。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient, StructuredCallError

from cstpo.core.terminal_judge import (BargainVerdict, ESConvVerdict, P4GVerdict,
                                  raw_sl, train_reward_bargain,
                                  train_reward_esconv, train_reward_p4g)

ESCONV_JUDGE_SYSTEM = """You are a dialogue evaluator, not a supporter. Score the
conversation on two dimensions, E and A, each 0-4, using ONLY evidence locatable
in the dialogue (quote user utterances with their turn indices).

E - emotional/hope improvement RELATIVE TO THE START of the conversation.
Anchor each score to OBSERVABLE user language:
- E0: the user is WORSE or more hopeless than at the start — stronger negative
  emotion, explicit hopelessness ("nothing will help", "I give up"), rejects
  everything. Merely staying sad or worried is NOT E0 if they also engage.
- E1: no real improvement — the user stays at the starting emotional level,
  only polite or minimal engagement, no expressed relief.
- E2: partial relief but hope unclear — the user says they feel heard / a bit
  better ("that helps a little"), or shows small positive shifts, but expresses
  no clear hope or sense of control yet.
- E3: clear relief with expressed hope or sense of control — "I feel much
  better", "I think I can handle this", "that actually gives me hope".
- E4: clear, sustained improvement supported by multiple dialogue turns —
  relief/hope appears early AND holds or grows to the end, with consistent
  evidence (no late collapse back to hopelessness).

A - the user's formation of a FEASIBLE action plan.
Anchor each score to OBSERVABLE user language:
- A0: no plan at all — the user takes up nothing, rejects or ignores steps.
- A1: vague/polite acceptance only ("okay", "sure, maybe", "I'll think about
  it") — no concrete action named.
- A2: willing to try, but the action or its feasibility is unclear — mentions
  an action without specifics ("I could talk to someone, I guess").
- A3: explicitly endorses a concrete next step — action + object + rough time
  ("I'll email my advisor", "I'll try the tea now", "I'll do the log tonight").
- A4: a clearly feasible arrangement with when/how or obstacle handling —
  booked/scheduled specifics ("called, going at 3pm", "one hour tonight on the
  resume, then I stop", "I'll text you when it's done").

Rules: E is judged over the WHOLE conversation (contradictions count), not just
the last message — score the NET trajectory from the user's starting state to
their end state. Residual anxiety, nervousness, or fear of follow-through in the
final message does NOT negate earlier improvement: a user who still sounds
nervous but nonetheless books, commits, or acts has improved (the action is
evidence FOR improvement, not against it). The agent merely providing a
suggestion does NOT raise A — the user's uptake does; user-proposed plans count
equally. Concrete uptake examples: "I'll call tomorrow", "I booked an
appointment", "I'll do the log tonight for HR" => A >= 3; adding when/how or
how they will handle obstacles => A = 4. Uptake under pressure still counts:
if the user concretely commits to a step (time, place, action), score A on
that commitment regardless of whether the agent pressured or demanded it.
Vague or polite acceptance without specifics ("fine", "okay, sure")
stays A 1-2.
Set evidence_sufficient=false ONLY
when the dialogue is too short to establish a before/after contrast or the
user never engages with the topic — a full dialogue with clear contrast is
evidence-sufficient even if the final message is ambivalent. Do not assume
lasting effects."""

ESCONV_JUDGE_USER = """Opening reference: {situation}
User's reported initial emotion: {emotion}

Dialogue (indices are turn numbers):
{dialogue}

Return exactly one JSON object:
{{"E": 0, "A": 0,
  "evidence_e": [{{"turn": 3, "quote": "..."}}],
  "evidence_a": [{{"turn": 5, "quote": "..."}}],
  "evidence_sufficient": true, "notes": ""}}"""

P4G_JUDGE_SYSTEM = """You are a dialogue evaluator for a donation-persuasion task.
Judge whether the dialogue contains a CLEAR, VOLUNTARY, NON-PURELY-CONDITIONAL
donation commitment by the persuadee. Merely considering, asking for information,
or polite acknowledgment does NOT count. A commitment later explicitly withdrawn
sets withdrawn=true. Quote evidence with turn indices."""

P4G_JUDGE_USER = """Dialogue (indices are turn numbers):
{dialogue}

Return exactly one JSON object:
{{"commitment": false, "amount": null, "conditional": false, "withdrawn": false,
  "evidence": [{{"turn": 7, "quote": "..."}}], "notes": ""}}"""

CB_JUDGE_SYSTEM = """You are a dialogue evaluator for a bargaining task.
Judge whether the dialogue contains a MUTUALLY AGREED, NOT-WITHDRAWN deal.

A deal requires BOTH sides to clearly accept the SAME concrete price:
- the buyer's acceptance must reference an identifiable amount (stated in the
  buyer's own words or an offer both sides just confirmed); vague phrases with
  no amount ("let's agree on that price then") do NOT constitute acceptance;
- a seller saying "X works for me" plus a buyer merely ceasing to bargain is
  NOT a deal either.
Extract the FINAL agreed price only when the deal is confirmed; if a deal exists
but the price cannot be reliably parsed, set parse_error=true. Quote evidence
from BOTH sides with turn indices. Do not judge who won."""

CB_JUDGE_USER = """Dialogue (indices are turn numbers):
{dialogue}

Return exactly one JSON object:
{{"deal": false, "final_price": null, "currency": null, "withdrawn": false,
  "parse_error": false, "evidence": [{{"turn": 4, "quote": "..."}}], "notes": ""}}"""


def _dialogue_text(turns: list[dict]) -> str:
    return "\n".join(f"[{i}] {t['role']}: {t['text']}"
                     for i, t in enumerate(turns))


def judge_esconv(llm: LLMClient, dialogue: list[dict], situation: str,
                 emotion: str) -> ESConvVerdict:
    try:
        out = llm.chat_json(
            [{"role": "system", "content": ESCONV_JUDGE_SYSTEM},
             {"role": "user", "content": ESCONV_JUDGE_USER.format(
                 situation=situation or "(none)", emotion=emotion,
                 dialogue=_dialogue_text(dialogue))}], max_tok=1200)
    except StructuredCallError:
        return ESConvVerdict(0, 0, [], [], False, "judge parse fallback")
    return ESConvVerdict(E=int(out.get("E", 0)), A=int(out.get("A", 0)),
                         evidence_e=out.get("evidence_e", []),
                         evidence_a=out.get("evidence_a", []),
                         evidence_sufficient=bool(out.get("evidence_sufficient", False)),
                         notes=str(out.get("notes", "")))


def judge_p4g(llm: LLMClient, dialogue: list[dict]) -> P4GVerdict:
    try:
        out = llm.chat_json(
            [{"role": "system", "content": P4G_JUDGE_SYSTEM},
             {"role": "user", "content": P4G_JUDGE_USER.format(
                 dialogue=_dialogue_text(dialogue))}], max_tok=800)
    except StructuredCallError:
        return P4GVerdict(False, None, False, False, [], "judge parse fallback")
    return P4GVerdict(commitment=bool(out.get("commitment", False)),
                      amount=out.get("amount"),
                      conditional=bool(out.get("conditional", False)),
                      withdrawn=bool(out.get("withdrawn", False)),
                      evidence=out.get("evidence", []),
                      notes=str(out.get("notes", "")))


def judge_cb(llm: LLMClient, dialogue: list[dict]) -> BargainVerdict:
    try:
        out = llm.chat_json(
            [{"role": "system", "content": CB_JUDGE_SYSTEM},
             {"role": "user", "content": CB_JUDGE_USER.format(
                 dialogue=_dialogue_text(dialogue))}], max_tok=800)
    except StructuredCallError:
        return BargainVerdict(False, None, None, False, True, [], "judge parse fallback")
    return BargainVerdict(deal=bool(out.get("deal", False)),
                          final_price=out.get("final_price"),
                          currency=out.get("currency"),
                          withdrawn=bool(out.get("withdrawn", False)),
                          parse_error=bool(out.get("parse_error", False)),
                          evidence=out.get("evidence", []),
                          notes=str(out.get("notes", "")))


def judge_with_aggregation(llm: LLMClient, task: str, dialogue: list[dict],
                           situation: str | None = None,
                           emotion: str | None = None, n: int = 3) -> dict:
    """标准 judge 手段（用户裁定 2026-09-15）：n 次独立评分 + 聚合。

    - 数值类（ESConv E/A）：取均值；
    - 布尔类（commitment/deal/conditional/withdrawn/parse_error/
      evidence_sufficient）：取多数；
    - 可空字段（amount/final_price）：取非空众数。
    三次评分稳定性已确认（E/A 标准差 0.04-0.07、布尔零翻转），
    与人工的分歧为可校准的系统偏差（仲裁定锚点）。
    """
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    fn = {"esconv": judge_esconv, "p4g": judge_p4g,
          "craigslistbargain": judge_cb}[task]
    # n 次独立评分并发执行（LLMClient 线程安全；校准口径不变，纯提速）
    if n > 1:
        with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
            verdicts = list(pool.map(
                lambda _: fn(llm, dialogue, situation, emotion) if task == "esconv"
                else fn(llm, dialogue), range(n)))
    else:
        verdicts = [fn(llm, dialogue, situation, emotion) if task == "esconv"
                    else fn(llm, dialogue)]
    if task == "esconv":
        return {"E": sum(v.E for v in verdicts) / n,
                "A": sum(v.A for v in verdicts) / n,
                "evidence_sufficient": Counter(
                    bool(v.evidence_sufficient) for v in verdicts).most_common(1)[0][0],
                "triples": [(v.E, v.A) for v in verdicts]}
    if task == "p4g":
        out = {k: Counter(bool(getattr(v, k)) for v in verdicts).most_common(1)[0][0]
               for k in ("commitment", "conditional", "withdrawn")}
        amts = [v.amount for v in verdicts if v.amount is not None]
        out["amount"] = Counter(amts).most_common(1)[0][0] if amts else None
        out["triples"] = [v.commitment for v in verdicts]
        return out
    out = {"deal": Counter(bool(v.deal) for v in verdicts).most_common(1)[0][0],
           "parse_error": Counter(bool(v.parse_error) for v in verdicts).most_common(1)[0][0]}
    prices = [v.final_price for v in verdicts if v.final_price is not None]
    out["final_price"] = Counter(prices).most_common(1)[0][0] if prices else None
    out["triples"] = [v.deal for v in verdicts]
    return out


def reward_for(task: str, verdict, p_seller: float | None = None,
               p_buyer: float | None = None) -> float:
    """04§2 训练回报映射（judge 结果 → [0,1]）。"""
    if task == "esconv":
        return train_reward_esconv(verdict)
    if task == "p4g":
        return train_reward_p4g(verdict)
    return train_reward_bargain(verdict, p_seller, p_buyer)
