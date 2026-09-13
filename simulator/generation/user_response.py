"""自然语言生成（文档 §37、§38）。

三模式统一两步生成：先由 Engine（Influence）或合并调用（Elicit / Social）产出
appraisal + emotion_proposal + reaction_plan（Elicit 另记录 revealed_items），
程序算完 A/E 后，由统一 NLG 调用基于 (C_{t+1}, E_{t+1}) + reaction_plan
单独生成 utterance（认知-评价-语言分离；消除合并调用中文本只能看到 E_t 的滞后）。
"""
from __future__ import annotations

from simulator.affect.emotion_engine import CATEGORY_LIST, DESIRE_ASSESSMENT_DEF
from simulator.llm import StructuredCallError
from simulator.routing.route_features import INTERACTION_PRESSURE_DEF

# NLG 失败时的安全话语（按 mode，与 persona 语言一致，不含任何新承诺）
SAFE_UTTERANCES = {
    "influence": "嗯，我再想想吧。",
    "elicit": "这个不太好说。",
    "social": "嗯，行。",
}

NLG_SYSTEM = """You are simulating one user in a multi-turn conversation.
Generate the user's next message based on the updated internal state.
The message must be consistent with the user's beliefs, desires,
intentions, current emotion, persona, and conversation history."""

NLG_USER = """Persona:
{persona}

Updated internal state (JSON):
{state_json}

Current emotion (updated this turn):
{emotion}

Recent conversation:
{history}

Assistant's latest reply:
{assistant_reply}

Dialogue mode: {mode}
{mode_grounding}
Reaction plan for this reply:
{reaction_plan}
{reveal_block}
Rules:
- Do not explain the user's full internal state.
- A real user usually reveals only part of what they think.
- Do not mention: belief, desire, intention, appraisal, processing route,
  acceptance or rejection labels, cognitive transition.
- The reaction plan is a semantic direction, NOT text to copy or translate.
- Sound like a real user in this situation: concise, no templated openers
  (e.g. avoid starting every reply with "I understand"), no restating the
  whole situation, no echoing the assistant's words back.
- Keep the user's existing conversational style and level of detail.
- Do not become more cooperative simply because the assistant has a goal.

Output ONLY the user's next utterance text, no quotes, no JSON."""


NLG_MODE_GROUNDING = {
    "influence": ("Mode rule: express the updated cognitive-affective reaction — "
                  "opinions, goals, or behavioral tendencies may shift; do not "
                  "restate the full state."),
    "elicit": ("Mode rule: reveal selected EXISTING states only (state revelation "
               "is NOT state change — revealing a thought creates no new cognition "
               "and no new commitment)."),
    "social": ("Mode rule: affective/social response only. Do not create a new "
               "commitment, promise, or action intention that is absent from the "
               "current state."),
}


def generate_utterance(llm, state, reaction_plan: str | None, emotion=None,
                       assistant_reply: str | None = None,
                       revealed: list | None = None,
                       mode: str = "influence") -> str:
    """生成用户下一轮话语（文档 §37、§38），三模式统一的两步生成入口。

    - emotion：本轮更新后的 Emotion（E_{t+1}），保证话语基调与最终状态一致
      （不传则用 state.emotion）。
    - assistant_reply：agent 本轮回复原文，话语须回应它（不传则仅靠 plan）。
    - revealed：Elicit 的 revealed_items（将被揭示的已有节点 id），话语自然
      揭示其中部分内容。
    - mode：q_t 显式传入（Mode-Conditioned Response Generation）——三种模式
      各有最小 grounding 约束（NLG_MODE_GROUNDING），Social 在最终语言层
      兜底防承诺超发。
    """
    emo = emotion if emotion is not None else state.emotion
    reveal_block = ""
    if revealed:
        reveal_block = ("\nState revelation (Elicit): naturally reveal part of the "
                        f"content of these existing nodes: {revealed}\n")
    msgs = [
        {"role": "system", "content": NLG_SYSTEM},
        {"role": "user", "content": NLG_USER.format(
            persona=state.persona,
            state_json=str(state.bdi_dict()),
            emotion=f"valence={emo.valence:.2f}, arousal={emo.arousal:.2f}, category={emo.category}",
            history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-8:]),
            assistant_reply=assistant_reply or "(not available)",
            mode=mode,
            mode_grounding=NLG_MODE_GROUNDING.get(mode, ""),
            reaction_plan=reaction_plan or "(react naturally)",
            reveal_block=reveal_block,
        )},
    ]
    for attempt in range(2):
        try:
            text = llm.chat(msgs, max_tok=300, json_mode=False)
        except Exception as e:
            print(f"  [nlg] 生成调用失败（第 {attempt + 1} 次）: {e}")
            text = ""
        text = text.strip().strip('"').strip("“”").strip()
        if text:
            return text
    # 两次失败：返回与 mode 一致的安全话语，不产生新承诺（Validation 1.1）
    print(f"  [nlg] 生成失败，返回安全话语（mode={mode}）")
    return SAFE_UTTERANCES.get(mode, "嗯。")


RESPOND_USER = """You are simulating the user in a multi-turn conversation.

Persona:
{persona}

Current BDI state (JSON):
{state_json}

Current emotion:
{emotion}

Recent conversation:
{history}

Assistant's latest reply:
{reply}

Dialogue mode: {mode}

Rules:
- In this mode the user's BDI does NOT change.
{mode_rules}- Do NOT generate the utterance here — a separate NLG step will produce the reply
  from your appraisal, emotion proposal, and reaction plan.
- "reaction_plan": one short sentence (5-15 words) describing what the user will
  SAY in this turn, so the separate NLG step can follow it.{plan_rules}
- The future utterance must not mention belief/desire/intention/appraisal/route/judgment terms.
- Do not become more cooperative simply because the assistant has a goal.

Return exactly one JSON object:
{{
  "appraisal": {{"goal_congruence": 0.0, "coping_potential": 0.0, "future_expectancy": 0.0}},
  "desire_assessment": [{{"id": "D1", "relevance": 0.8, "gc": -0.3}}],
  "interaction_pressure": "low | medium | high",
  "emotion_proposal": {{"category": "neutral"}},
  "reaction_plan": "brief plan for the next utterance"{revealed_field}
}}

Appraisal values are in [-1, 1], and must reflect THIS turn's situation relative
to the user's desires — do not copy the example numbers.
(emotion_proposal only takes a category — the program derives valence and arousal.)
{desire_assessment_def}
The program may recompute goal_congruence from desire_assessment; still fill in
appraisal.goal_congruence with your best estimate as a fallback.
"interaction_pressure": {pressure_def}
category must be one of: {category_list}."""


def respond_without_bdi_change(llm, state, reply: str, mode: str) -> dict:
    """Elicit / Social 分支：BDI 冻结，产出 appraisal + emotion 提案 + 反应计划。

    reply 必须传入：该分支不重算认知，但 appraisal 与反应计划仍需以
    agent 本轮实际说了什么为依据（否则生成会答非所问）。
    话语不在此生成——统一由两步 NLG 基于 (C_{t+1}, E_{t+1}) 产出。
    Elicit 额外要求 revealed_items（仅 id）；Social 的反应计划限定为
    情感/社会性回应（防口头承诺超发，审计 §8.2）。
    """
    if mode == "elicit":
        mode_rules = ("- Elicit: decide which existing B/D/I ids the user would reveal in this\n"
                      "  turn's answer, and list them in \"revealed_items\" (ids only — the\n"
                      "  program already knows their content). Revealing a thought is NOT a\n"
                      "  cognitive change.\n")
        plan_rules = " (which question to answer, which existing nodes to partially reveal)"
        revealed_field = ',\n  "revealed_items": ["D2"]'
    else:
        mode_rules = "- Social: keep it light; only mild emotion change is allowed.\n"
        plan_rules = (" (an affective / social reaction only — NO new commitment, NO\n"
                      "  behavioral promise, NO change of intention)")
        revealed_field = ""
    msgs = [
        {"role": "user", "content": RESPOND_USER.format(
            persona=state.persona,
            state_json=str(state.bdi_dict()),
            emotion=f"valence={state.emotion.valence:.2f}, arousal={state.emotion.arousal:.2f}, category={state.emotion.category}",
            history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-6:]),
            reply=reply,
            mode=mode,
            mode_rules=mode_rules,
            plan_rules=plan_rules,
            revealed_field=revealed_field,
            desire_assessment_def=DESIRE_ASSESSMENT_DEF,
            pressure_def=INTERACTION_PRESSURE_DEF,
            category_list=CATEGORY_LIST,
        )},
    ]
    try:
        out = llm.chat_json(msgs, max_tok=800)
    except StructuredCallError:
        # 保守 fallback：C 冻结、无揭示、中性 appraisal、与 mode 相符的最小计划
        plan = ("answer briefly, revealing only what is asked, no new state"
                if mode == "elicit" else "stay light, social/affective only, no commitment")
        out = {"appraisal": {"goal_congruence": 0.0, "coping_potential": 0.0,
                             "future_expectancy": 0.0},
               "desire_assessment": [], "interaction_pressure": "medium",
               "emotion_proposal": {"category": "neutral"},
               "reaction_plan": plan, "revealed_items": [],
               "_fallback": "srr_err"}
    out.setdefault("reaction_plan", None)
    out.setdefault("revealed_items", [])
    return out
