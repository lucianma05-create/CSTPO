"""Cognitive Engine（文档 §28、§29、§30）：主 LLM 调用，提出候选 BDI 更新 +
appraisal + emotion_proposal + reaction_plan。

输入显式包含 Route 与 Judgment（文档 §28），LLM 输出只是"提案"，
最终状态由 Deterministic Updater 施加约束后得到（文档 §31）。
"""
from __future__ import annotations

ENGINE_SYSTEM = """You are simulating one user in a multi-turn conversation.

Your task is not to help the assistant succeed.
Your task is to simulate how this specific user's cognition and emotion
would realistically change after the assistant's latest reply.

The user's cognitive state contains:

Beliefs:
what the user currently thinks is true.

Desires:
what outcomes the user wants to achieve or avoid.

Intentions:
concrete actions the user is currently inclined or committed to take.

Emotion:
the user's short-term affective state.

Important rules:

1. Preserve continuity with the current cognitive state.
2. Do not change unrelated beliefs, desires, or intentions.
3. Do not make large cognitive changes unless the provided cognitive
   transition constraints explicitly permit them.
4. A strong intention change should be supported by an existing or updated
   belief or desire.
5. Do not optimize for the assistant's task objective.
6. The user's cognition is task-neutral.
7. The final user utterance should sound natural and should reveal only
   part of the internal state.
8. Never explicitly mention BDI, processing route, judgment category,
   appraisal variables, or cognitive transition labels in the utterance.
9. Follow the provided cognitive transition contract strictly."""

ENGINE_USER = """Persona:
{persona}

Cognitive habit card (fixed across the whole conversation):
{habit_card}

Current BDI state (JSON):
{state_json}

Current emotion:
{emotion}

Recent conversation:
{history}

Assistant's latest reply:
{reply}

Dialogue mode: Influence
Processing route: {route}
Judgment: {judgment}
Cognitive transition contract for this turn:
{contract}

Propose the user's cognitive update as exactly one JSON object:
{{
  "bdi_updates": [
    {{"operation": "update", "id": "B1", "new_strength": 2.4, "reason": "brief reason"}},
    {{"operation": "update", "id": "D2", "new_strength": 3.5, "reason": "the deal's feasibility became clearer, strengthening this desire"}},
    {{"operation": "update", "id": "I1", "new_strength": 2.3, "reason": "the intention's supporting belief weakened"}}
  ],
  "new_items": [
    {{"type": "belief", "content": "user first-person statement", "strength": 1.5,
      "core": true, "cue": false, "conflicts_with": ["B1"], "reason": "brief reason"}},
    {{"type": "desire", "content": "user first-person statement", "strength": 1.2,
      "core": true, "polarity": "avoid", "reason": "brief reason"}}
  ],
  "reaction_plan": "Acknowledge the evidence but remain cautious because spending concerns remain.",
  "appraisal": {{"goal_congruence": 0.0, "coping_potential": 0.0, "future_expectancy": 0.0}},
  "desire_assessment": [
    {{"id": "D1", "relevance": 0.8, "gc": -0.3}},
    {{"id": "D2", "relevance": 0.5, "gc": 0.4}}
  ],
  "emotion_proposal": {{"category": "neutral"}}
}}

Notes:
- "operation" is always "update" for existing items; new items go to "new_items".
- Beliefs, desires AND intentions may all be updated — output the ones that actually
  change this turn, within the contract limits. A desire changes when the reply alters
  its importance, feasibility or relevance. An intention changes when its supporting
  beliefs or desires change; if a supporting belief clearly moved, do not leave the
  intention untouched. The example above only illustrates the format.
- strength is in [0, 4]. Appraisal values are in [-1, 1].
  (emotion_proposal only takes a category — the program derives valence and arousal.)
- Appraisal values must reflect THIS turn's situation relative to the user's desires —
  do not copy the example numbers. goal_congruence: how favorable the situation is to the
  user's important desires now; coping_potential: whether the user feels able to act;
  future_expectancy: whether an acceptable outcome seems achievable.
- "desire_assessment": assess each ACTIVE desire against THIS turn's situation
  (the assistant's latest reply and its consequences). Only include desires with
  relevance > 0. "relevance" in [0, 1] = how directly the situation concerns that
  desire. "gc" in [-1, 1] = whether the situation promotes (+) or hinders (-) that
  desire from the user's perspective. This is about the situation's favorability,
  NOT about whether the desire strength should change.
- The program may recompute goal_congruence from desire_assessment; still fill in
  appraisal.goal_congruence with your best estimate as a fallback.
- category must be one of: neutral, sadness, anxiety, frustration, interest, hope, relief, satisfaction, anger.
- "cue": true ONLY for a peripheral-cue belief about trust, popularity, authority,
  familiarity, or social norm. Issue-relevant beliefs about the matter under
  discussion are NOT cue beliefs — mark them "cue": false even if the reply
  mentions popularity or authority.
- "core": true for beliefs/desires/intentions central to the user's current
  situation; use "core": false only for clearly peripheral details.
- "polarity" applies to desires and intentions only: "approach" = wants to achieve / to do it,
  "avoid" = wants to avoid / not to do it. Omit it for beliefs.
- "conflicts_with" applies to new beliefs: list the ids of existing beliefs that the new belief
  directly contradicts (e.g. "the seller may accept 80" vs "the seller will not go below 85").
  The program will weaken the conflicting belief accordingly, so only mark real contradictions.
- Only output the JSON object."""


def propose_cognitive_update(llm, state, reply: str, contract: str,
                             route: str, judgment: str) -> dict:
    msgs = [
        {"role": "system", "content": ENGINE_SYSTEM},
        {"role": "user", "content": ENGINE_USER.format(
            persona=state.persona,
            habit_card=state.habit_card,
            state_json=str(state.bdi_dict()),
            emotion=f"valence={state.emotion.valence:.2f}, arousal={state.emotion.arousal:.2f}, category={state.emotion.category}",
            history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-8:]),
            reply=reply,
            route=route,
            judgment=judgment,
            contract=contract,
        )},
    ]
    return llm.chat_json(msgs, max_tok=1200)
