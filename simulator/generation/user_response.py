"""自然语言生成（文档 §37、§38）。

三模式统一两步生成：先由 Engine（Influence）或合并调用（Elicit / Social）产出
appraisal + emotion_proposal + reaction_plan（Elicit 另记录 revealed_items），
程序算完 A/E 后，由统一 NLG 调用基于 (C_{t+1}, E_{t+1}) + reaction_plan
单独生成 utterance（认知-评价-语言分离；消除合并调用中文本只能看到 E_t 的滞后）。
"""
from __future__ import annotations

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

Reaction plan for this reply:
{reaction_plan}
{reveal_block}
Rules:
- Do not explain the user's full internal state.
- A real user usually reveals only part of what they think.
- Do not mention: belief, desire, intention, appraisal, processing route,
  acceptance or rejection labels, cognitive transition.
- Follow the reaction plan naturally.
- Keep the user's existing conversational style and level of detail.
- Do not become more cooperative simply because the assistant has a goal.

Output ONLY the user's next utterance text, no quotes, no JSON."""


def generate_utterance(llm, state, reaction_plan: str | None, emotion=None,
                       assistant_reply: str | None = None,
                       revealed: list | None = None) -> str:
    """生成用户下一轮话语（文档 §37、§38），三模式统一的两步生成入口。

    - emotion：本轮更新后的 Emotion（E_{t+1}），保证话语基调与最终状态一致
      （不传则用 state.emotion）。
    - assistant_reply：agent 本轮回复原文，话语须回应它（不传则仅靠 plan）。
    - revealed：Elicit 的 revealed_items（将被揭示的已有节点 id），话语自然
      揭示其中部分内容。
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
            reaction_plan=reaction_plan or "(react naturally)",
            reveal_block=reveal_block,
        )},
    ]
    text = llm.chat(msgs, max_tok=300, json_mode=False)
    return text.strip().strip('"').strip("“”").strip()


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
- In this mode the user's BDI does NOT change (state revelation only).
- In Elicit mode: decide which existing B/D/I ids the user would reveal in this
  turn's answer, and list them in "revealed_items".
- In Social mode: keep it light; only mild emotion change is allowed.
- Do NOT generate the utterance here — a separate NLG step will produce the reply
  from your appraisal, emotion proposal, and reaction plan.
- "reaction_plan": describe what the user will SAY in this turn, so the separate
  NLG step can follow it (Elicit: which question to answer, which existing nodes
  to partially reveal; Social: stay light, no substantive state).
- The future utterance must not mention belief/desire/intention/appraisal/route/judgment terms.
- Do not become more cooperative simply because the assistant has a goal.

Return exactly one JSON object:
{{
  "appraisal": {{"goal_congruence": 0.0, "coping_potential": 0.0, "future_expectancy": 0.0}},
  "desire_assessment": [
    {{"id": "D1", "relevance": 0.8, "gc": -0.3}}
  ],
  "interaction_pressure": "low | medium | high",
  "emotion_proposal": {{"category": "neutral"}},
  "reaction_plan": "brief plan for the next utterance",
  "revealed_items": ["D2"]
}}

Appraisal values are in [-1, 1].
(emotion_proposal only takes a category — the program derives valence and arousal.)
Appraisal values must reflect THIS turn's situation relative to the user's desires —
do not copy the example numbers.
"desire_assessment": assess each ACTIVE desire against THIS turn's situation
(the assistant's latest reply and its consequences). Only include desires with
relevance > 0. "relevance" in [0, 1] = how directly the situation concerns that
desire. "gc" in [-1, 1] = whether the situation promotes (+) or hinders (-) that
desire from the user's perspective. This is about the situation's favorability,
NOT about whether the desire strength should change.
The program may recompute goal_congruence from desire_assessment; still fill in
appraisal.goal_congruence with your best estimate as a fallback.
"interaction_pressure": how much the assistant's latest reply pushes the user to
respond or decide immediately. Judge ONLY the demand the reply places on the user's
immediate response — do not confuse it with persuasiveness, topic importance, or
how the user might feel. high: presses for an answer or commitment right now
(deadline, ultimatum, confrontation, repeated push, a question that cannot
reasonably be deferred). medium: invites a response or nudges toward a decision
but leaves room to defer (ordinary question or mild suggestion, no deadline).
low: low-stakes and open-ended (chitchat, plain information, casual remarks);
the user could stay silent or reply at leisure without consequence.
category must be one of: neutral, sadness, anxiety, frustration, interest, hope, relief, satisfaction, anger."""


def respond_without_bdi_change(llm, state, reply: str, mode: str) -> dict:
    """Elicit / Social 分支：BDI 冻结，产出 appraisal + emotion 提案 + 反应计划。

    reply 必须传入：该分支不重算认知，但 appraisal 与反应计划仍需以
    agent 本轮实际说了什么为依据（否则生成会答非所问）。
    话语不在此生成——统一由两步 NLG 基于 (C_{t+1}, E_{t+1}) 产出。
    """
    msgs = [
        {"role": "user", "content": RESPOND_USER.format(
            persona=state.persona,
            state_json=str(state.bdi_dict()),
            emotion=f"valence={state.emotion.valence:.2f}, arousal={state.emotion.arousal:.2f}, category={state.emotion.category}",
            history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-8:]),
            reply=reply,
            mode=mode,
        )},
    ]
    out = llm.chat_json(msgs, max_tok=800)
    out.setdefault("reaction_plan", None)
    out.setdefault("revealed_items", [])
    return out
