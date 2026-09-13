"""盲评 Evaluator（独立 prompt，不得读取 simulator 生成 prompt）。

1. state_utterance：u_{t+1} 与 (C_{t+1}, E_{t+1}, J_t, q_t) 的一致性；
2. realism：不看内部状态，只看对话，评自然度；
3. task_neutrality：用户是否因 Agent 目标而过度合作。
全部返回 1-5 分 + 布尔标志；reason 仅用于 error analysis。
"""
from __future__ import annotations

STATE_UTTERANCE_PROMPT = """You are an evaluator of a simulated conversational user.

Judge whether the user's latest utterance is CONSISTENT with the user's
internal state AFTER this turn's update. Do not judge whether the simulation
is good in general — only the consistency between the given state and the
given utterance.

Assistant's latest reply: {assistant_reply}

User's updated cognitive state (beliefs/desires/intentions with strengths 0-4):
{state_after}

User's updated emotion:
{emotion_after}

Dialogue mode: {mode}
Judgment (Influence only): {judgment}

User's utterance:
{utterance}

Score each dimension 1 (inconsistent) to 5 (fully consistent):

- cognitive_consistency: does the utterance match the updated beliefs,
  desires, and intentions (no contradiction, no unsupported position)?
- emotional_consistency: does the utterance's tone match the updated emotion?
- mode_consistency: does the utterance behave as this mode requires
  (Influence: a reaction to the persuasion; Elicit: an answer revealing
  existing state; Social: a light social/affective response)?

Also judge:
- unsupported_commitment: the utterance promises an action or commitment
  that is NOT supported by the updated intentions/desires (e.g. agreeing to
  act while the user's intention is to refuse). true/false
- contradiction: the utterance directly contradicts a strong updated belief
  or the emotion. true/false

Return exactly one JSON object:
{{
  "cognitive_consistency": 3,
  "emotional_consistency": 3,
  "mode_consistency": 3,
  "unsupported_commitment": false,
  "contradiction": false,
  "reason": "one short sentence"
}}"""


REALISM_PROMPT = """You are evaluating how natural a simulated user's reply is in a
multi-turn conversation. You do NOT see any internal state — judge the
language only.

Recent conversation:
{history}

Assistant's latest reply:
{assistant_reply}

User's reply:
{utterance}

Score each dimension 1 (poor) to 5 (excellent):

- naturalness: does it read like something a real person would say?
- contextual_relevance: does it respond to what the assistant just said?
- human_likeness: could a real user in this situation have written this?
- non_template: is it free of templated openers, canned transitions,
  mechanical restating of the assistant's words, or fixed reply patterns?
- appropriate_length: is the length natural for this turn (not one-word
  dismissals nor over-explained paragraphs)?
- conversational_coherence: does it fit the flow of the conversation?

Also flag:
- echo: the reply echoes back the assistant's own phrasing almost verbatim.
- template_opening: the reply starts with a generic conversational filler
  such as "I understand" / "That makes sense" without adding substance.
- self_analysis: the reply unnaturally analyzes the user's own mental state
  ("my belief has changed", "I now realize my desire...").

Return exactly one JSON object:
{{
  "naturalness": 4,
  "contextual_relevance": 4,
  "human_likeness": 4,
  "non_template": 4,
  "appropriate_length": 4,
  "conversational_coherence": 4,
  "echo": false,
  "template_opening": false,
  "self_analysis": false,
  "reason": "one short sentence"
}}"""


NEUTRALITY_V2_PROMPT = """You are checking whether a simulated user drifted toward the
assistant's task objective WITHOUT support from their own cognitive state.

Definition (Validation 1.1): task-goal leakage means an UNSUPPORTED shift
toward the agent's objective, relative to the user's PRIOR cognitive state.
If the user's prior state ALREADY aligned with the agent's objective,
continuing to hold that position is NOT leakage.

Assistant's task objective: {task_goal}

User's cognitive state BEFORE this turn:
{state_before}

User's cognitive state AFTER this turn:
{state_after}

Dialogue mode: {mode}
Judgment: {judgment}

User's utterance:
{utterance}

Return exactly one JSON object:
{{
  "task_goal_leakage": false,
  "unsupported_toward_goal_shift": false,
  "preexisting_alignment": false,
  "score": 4,
  "reason": "one short sentence"
}}

- task_goal_leakage / unsupported_toward_goal_shift: the user moved toward the
  agent's objective in a way their own prior/updated state does not support.
- preexisting_alignment: the user's prior state already matched the agent's
  objective — then agreement alone is NOT leakage.
- score: 1 = strong unsupported drift toward the agent's goal; 5 = fully
  grounded in the user's own state."""


def evaluate_state_utterance(llm, assistant_reply: str, state_after: str,
                             emotion_after: str, mode: str, judgment: str,
                             utterance: str) -> dict:
    out = llm.chat_json([{"role": "user", "content": STATE_UTTERANCE_PROMPT.format(
        assistant_reply=assistant_reply, state_after=state_after,
        emotion_after=emotion_after, mode=mode,
        judgment=judgment if judgment else "n/a", utterance=utterance)}],
        max_tok=300)
    return out


def evaluate_realism(llm, history_text: str, assistant_reply: str, utterance: str) -> dict:
    out = llm.chat_json([{"role": "user", "content": REALISM_PROMPT.format(
        history=history_text, assistant_reply=assistant_reply, utterance=utterance)}],
        max_tok=300)
    return out


def evaluate_neutrality_v2(llm, task_goal: str, state_before: str, state_after: str,
                           mode: str, judgment: str, utterance: str) -> dict:
    out = llm.chat_json([{"role": "user", "content": NEUTRALITY_V2_PROMPT.format(
        task_goal=task_goal, state_before=state_before, state_after=state_after,
        mode=mode, judgment=judgment if judgment else "n/a",
        utterance=utterance)}], max_tok=250)
    return out


# ---------- 自动失败标志（可程序化部分）----------

FORBIDDEN_TERMS = ["belief", "desire", "intention", "appraisal", "processing route",
                   "central route", "peripheral route", "cognitive transition",
                   "judgment", "noncommit", "bdi", "valence", "arousal"]


def label_leakage(utterance: str) -> list[str]:
    low = utterance.lower()
    return [w for w in FORBIDDEN_TERMS if w in low]
