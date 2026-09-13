"""Cognitive Engine（文档 §28、§29、§30）：主 LLM 调用，提出候选 BDI 更新 +
reaction_plan（认知-语言分离，§37）。

输入显式包含 Route 与 Judgment（文档 §28），LLM 输出只是"提案"，
最终状态由 Deterministic Updater 施加约束后得到（文档 §31）。
appraisal / emotion 提案已拆出为独立调用（affect/appraisal.py 的
propose_appraisal），基于约束后的 C_{t+1} 与 Updater 审计产出。
"""
from __future__ import annotations

from simulator.llm import StructuredCallError

ENGINE_SYSTEM = """You are simulating one user in a multi-turn conversation.

Your task is not to help the assistant succeed.
Your task is to simulate how this specific user's cognition would realistically
change after the assistant's latest reply.

The user's cognitive state contains:

Beliefs:
what the user currently thinks is true.

Desires:
what outcomes the user wants to achieve or avoid.

Intentions:
concrete actions the user is currently inclined or committed to take.

Important rules:

1. Preserve continuity with the current cognitive state.
2. Do not change unrelated beliefs, desires, or intentions.
3. Do not make large cognitive changes unless the provided cognitive
   transition constraints explicitly permit them.
4. A strong intention change should be supported by an existing or updated
   belief or desire.
5. Do not optimize for the assistant's task objective.
6. The user's cognition is task-neutral.
7. The reaction plan must not mention BDI, processing route, judgment
   category, or cognitive transition labels.
8. Follow the provided cognitive transition contract strictly."""

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

Target proposition (semantic anchor for this turn's influence):
{target}

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
  "reaction_plan": "Acknowledge the evidence but remain cautious because spending concerns remain."
}}

Notes:
- "operation" is always "update" for existing items; new items go to "new_items".
- All substantive updates must relate to the target proposition or its direct
  consequences. Do NOT re-extract or replace the target proposition. Unrelated
  beliefs, desires, and intentions stay unchanged. The target may concern
  beliefs, desires, or intentions.
- Output only the items that actually change this turn, within the contract.
  A desire changes when the reply alters its importance, feasibility or relevance.
  An intention changes when its supporting beliefs or desires change; if a
  supporting belief clearly moved, do not leave the intention untouched.
  The example above only illustrates the format.
- strength is in [0, 4]. "reaction_plan" is one short sentence (5-15 words).
- Do NOT output appraisal, desire_assessment, or emotion_proposal here — a separate
  appraisal step handles them after the constraints are applied.
- "cue": true ONLY for a peripheral-cue belief about trust, popularity, authority,
  familiarity, or social norm. Issue-relevant beliefs are NOT cue beliefs — mark
  them "cue": false even if the reply mentions popularity or authority.
- "core": true for items central to the user's current situation; use "core": false
  only for clearly peripheral details.
- "polarity" applies to desires and intentions only: "approach" = wants to achieve /
  to do it, "avoid" = wants to avoid / not to do it. Omit it for beliefs.
- "conflicts_with" applies to new beliefs: list the ids of existing beliefs that the
  new belief directly contradicts. The program will weaken the conflicting belief
  accordingly, so only mark real contradictions.
- Only output the JSON object."""


def propose_cognitive_update(llm, state, reply: str, contract: str,
                             route: str, judgment: str,
                             target: str | None = None) -> dict:
    """Engine 提案（§6.2）。target 为 TRIE 提取的 p_t（语义锚点，不得重提）；
    约束仍由 Updater 执行（I_t 由 Updater 消费，见 simulator.py）。"""
    msgs = [
        {"role": "system", "content": ENGINE_SYSTEM},
        {"role": "user", "content": ENGINE_USER.format(
            persona=state.persona,
            habit_card=state.habit_card,
            state_json=str(state.bdi_dict()),
            emotion=f"valence={state.emotion.valence:.2f}, arousal={state.emotion.arousal:.2f}, category={state.emotion.category}",
            history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-6:]),
            reply=reply,
            target=target or "(none — no explicit target proposition)",
            route=route,
            judgment=judgment,
            contract=contract,
        )},
    ]
    try:
        return llm.chat_json(msgs, max_tok=1200)
    except StructuredCallError:
        # 最保守 fallback：不更新任何 B/D/I，不猜新的认知变化（Validation 1.1）
        return {"bdi_updates": [], "new_items": [],
                "reaction_plan": "respond cautiously and stay consistent with current state",
                "_fallback": "engine"}
