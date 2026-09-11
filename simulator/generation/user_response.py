"""自然语言生成（文档 §18、§19、§20、§21）。

Influence：Cognitive Engine 先给 reaction_plan，再单独生成 utterance（认知与语言分离）。
Elicit / Social：BDI 不变，一次轻量调用完成 appraisal + emotion_proposal +
reaction_plan + utterance（并记录 Elicit 的 revealed_items，文档 §20）。
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

Current emotion:
{emotion}

Recent conversation:
{history}

Reaction plan for this reply:
{reaction_plan}

Rules:
- Do not explain the user's full internal state.
- A real user usually reveals only part of what they think.
- Do not mention: belief, desire, intention, appraisal, processing route,
  acceptance or rejection labels, cognitive transition.
- Follow the reaction plan naturally.
- Keep the user's existing conversational style and level of detail.
- Do not become more cooperative simply because the assistant has a goal.

Output ONLY the user's next utterance text, no quotes, no JSON."""


def generate_utterance(llm, state, reaction_plan: str | None) -> str:
    msgs = [
        {"role": "system", "content": NLG_SYSTEM},
        {"role": "user", "content": NLG_USER.format(
            persona=state.persona,
            state_json=str(state.bdi_dict()),
            emotion=f"valence={state.emotion.valence:.2f}, arousal={state.emotion.arousal:.2f}, category={state.emotion.category}",
            history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-8:]),
            reaction_plan=reaction_plan or "(react naturally)",
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

Dialogue mode: {mode}

Rules:
- In this mode the user's BDI does NOT change (state revelation only).
- In Elicit mode: answer the assistant's question naturally, revealing part of the
  existing state. List which existing B/D/I ids were revealed in "revealed_items".
- In Social mode: keep it light; only mild emotion change is allowed.
- The utterance must not mention belief/desire/intention/appraisal/route/judgment terms.
- Do not become more cooperative simply because the assistant has a goal.

Return exactly one JSON object:
{{
  "appraisal": {{"goal_congruence": 0.0, "coping_potential": 0.0, "future_expectancy": 0.0}},
  "emotion_proposal": {{"valence": 0.0, "arousal": 0.4, "category": "neutral"}},
  "reaction_plan": "brief plan",
  "utterance": "the user's next message",
  "revealed_items": ["D2"]
}}

Appraisal values and valence are in [-1, 1]; arousal in [0, 1].
Appraisal values must reflect THIS turn's situation relative to the user's desires —
do not copy the example numbers.
category must be one of: neutral, sadness, anxiety, frustration, interest, hope, relief, satisfaction, anger."""


def respond_without_bdi_change(llm, state, mode: str) -> dict:
    """Elicit / Social 分支：BDI 冻结，只更新情绪并生成回复。"""
    msgs = [
        {"role": "user", "content": RESPOND_USER.format(
            persona=state.persona,
            state_json=str(state.bdi_dict()),
            emotion=f"valence={state.emotion.valence:.2f}, arousal={state.emotion.arousal:.2f}, category={state.emotion.category}",
            history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-8:]),
            mode=mode,
        )},
    ]
    out = llm.chat_json(msgs, max_tok=800)
    out.setdefault("reaction_plan", None)
    out.setdefault("revealed_items", [])
    out.setdefault("utterance", "")
    return out
