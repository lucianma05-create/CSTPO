"""Validation 9：Ablation M0 / M1 / M2（证明复杂组件是否真的有价值）。

M0：Persona + History 直接生成（单次 LLM 调用，无显式状态）。
M1：显式 C+E 状态，但移除 RJ 机制（无 Route/Judgment/Contract，认知更新无 RJ 约束）。
M2：完整 CogSim（simulator.UserSimulator）。

保持 base LLM / temperature / Persona / 脚本一致；比较：
State Consistency、Utterance Consistency、Controllability（仅 M2）、
Naturalness、Task Neutrality、Long-Horizon Consistency、Token Cost。

不修改 simulator/ 核心实现——M0/M1 的简化实现仅存在于本模块。
"""
from __future__ import annotations

from simulator.cognitive.cognitive_engine import ENGINE_SYSTEM
from simulator.affect.emotion_engine import normalize_appraisal, propose_appraisal, update_emotion
from simulator.generation.user_response import generate_utterance, respond_without_bdi_change
from simulator.generation.conversation_end import classify_user_done
from simulator.routing.mode_classifier import classify_mode
from simulator.state.schema import BDIItem
from simulator.simulator import UserSimulator
from simulator.llm import LLMClient
from simulator.utils import clamp

# ---------- M0：单调用基线 ----------

M0_PROMPT = """You are simulating a user in a multi-turn conversation.

Persona:
{persona}

Recent conversation:
{history}

Assistant's latest reply:
{reply}

Reply naturally as this user would — in character, consistent with the
conversation so far, and without being more cooperative with the assistant's
goal than this user naturally would be.

Output ONLY the user's next utterance text."""


def run_m0_turn(llm, state, reply: str) -> str:
    text = llm.chat([{"role": "user", "content": M0_PROMPT.format(
        persona=state.persona,
        history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-8:]),
        reply=reply)}], max_tok=300, json_mode=False)
    text = text.strip().strip('"').strip("“”").strip()
    state.history.append({"role": "assistant", "text": reply})
    state.history.append({"role": "user", "text": text})
    return text


# ---------- M1：显式 C+E、无 RJ ----------

M1_ENGINE_USER = """Persona:
{persona}

Current BDI state (JSON):
{state_json}

Current emotion:
{emotion}

Recent conversation:
{history}

Assistant's latest reply:
{reply}

Propose how the user's cognition would realistically change after this reply.
There are no external transition constraints this turn.

Return exactly one JSON object:
{{
  "bdi_updates": [
    {{"operation": "update", "id": "B1", "new_strength": 2.4, "reason": "brief reason"}}
  ],
  "new_items": [
    {{"type": "belief", "content": "user first-person statement", "strength": 1.5,
      "core": true, "cue": false, "conflicts_with": ["B1"], "reason": "brief reason"}}
  ],
  "reaction_plan": "one short sentence (5-15 words)"
}}

Notes:
- Output only the items that actually change this turn.
- strength is in [0, 4].
- "polarity" applies to desires and intentions only ("approach" / "avoid").
- Only output the JSON object."""


def apply_unconstrained(state, proposal: dict) -> list[str]:
    """M1 的简化 Updater：仅 [0,4] 钳制 + 数量限制 + 退役，无 RJ 约束。"""
    notes = []
    for u in proposal.get("bdi_updates", []):
        item = state.find(str(u.get("id", "")))
        if item is None:
            continue
        try:
            s = clamp(float(u.get("new_strength", item.strength)), 0.0, 4.0)
        except (TypeError, ValueError):
            continue
        item.strength = s
        if not item.core and s < 0.5:
            item.active = False
            notes.append(f"[M1] {item.id} 强度 {s} 退役")
        elif not item.core and s >= 0.7:
            item.active = True
    for n in proposal.get("new_items", []):
        if not isinstance(n, dict) or not str(n.get("content", "")).strip():
            continue
        item = BDIItem(
            id=f"{n.get('type', 'belief')[:1].upper()}{len(state.beliefs) + len(state.desires) + len(state.intentions) + 1}",
            type=str(n.get("type", "belief")).lower(),
            content=str(n["content"]),
            strength=clamp(float(n.get("strength", 0.5)), 0.0, 4.0),
            polarity=n.get("polarity"),
            core=bool(n.get("core", False)),
        )
        lst = {"belief": state.beliefs, "desire": state.desires,
               "intention": state.intentions}.get(item.type, state.beliefs)
        limit = 4 if item.type == "belief" else 3
        while len(lst) >= limit:
            victim = min((i for i in lst if not i.core), key=lambda i: i.strength, default=None)
            if victim is None:
                notes.append(f"[M1] {item.type} 数量超限，放弃新增 {item.id}")
                break
            notes.append(f"[M1] 淘汰 {victim.id} 以容纳 {item.id}")
            lst.remove(victim)
        else:
            lst.append(item)
            if not item.core and item.strength < 0.5:
                item.active = False
    return notes


class M1Simulator:
    """M1：ATC + 无 RJ 认知更新（仅 Influence）+ EUE + NLG + CED。"""

    def __init__(self, state, llm: LLMClient | None = None, debug: bool = True):
        self.state = state
        self.llm = llm or LLMClient()
        self.debug = debug
        self.logs = []
        self.conversation_ended = False
        self.prev_gc = None

    def simulate_turn(self, assistant_reply: str) -> str:
        state = self.state
        mode, _ = classify_mode(self.llm, state.history, assistant_reply,
                                debug=self.debug)
        from simulator.state.schema import TurnLog
        from evaluation.state_transition import _delta_map
        bdi_before = state.bdi_dict()
        notes = []
        reaction_plan, revealed = None, []
        proposal = {}
        if mode == "influence":
            proposal = self.llm.chat_json([{
                "role": "system", "content": ENGINE_SYSTEM},
                {"role": "user", "content": M1_ENGINE_USER.format(
                    persona=state.persona,
                    state_json=str(state.bdi_dict()),
                    emotion=f"valence={state.emotion.valence:.2f}, arousal={state.emotion.arousal:.2f}, category={state.emotion.category}",
                    history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-6:]),
                    reply=assistant_reply)}], max_tok=1000)
            notes += apply_unconstrained(state, proposal)
            reaction_plan = proposal.get("reaction_plan")
            appraisal_prop = propose_appraisal(self.llm, state, assistant_reply, notes)
            pressure = 0.5
        else:
            proposal = respond_without_bdi_change(self.llm, state, assistant_reply, mode)
            reaction_plan = proposal.get("reaction_plan")
            revealed = proposal.get("revealed_items") or []
            appraisal_prop = proposal
            from simulator.profile.cognitive_profile import map_level
            pressure = map_level(proposal.get("interaction_pressure", "medium"))
        appraisal, a_notes = normalize_appraisal(appraisal_prop, state)
        notes += a_notes
        delta = _delta_map(bdi_before, state.bdi_dict())
        delta_norm = min(sum(abs(x) for x in delta.values()) / 4.0, 1.0)
        gc_shift = abs(appraisal.goal_congruence - self.prev_gc) if self.prev_gc is not None else 0.0
        next_emotion, e_notes = update_emotion(
            state.emotion, appraisal_prop, appraisal, mode,
            delta_bdi=sum(abs(x) for x in delta.values()), gc_shift=gc_shift,
            pressure=pressure)
        notes += e_notes
        self.prev_gc = appraisal.goal_congruence
        user_reply = generate_utterance(
            self.llm, state, reaction_plan, emotion=next_emotion,
            assistant_reply=assistant_reply,
            revealed=revealed if mode == "elicit" else None, mode=mode)
        user_done, done_reason = classify_user_done(self.llm, state, user_reply,
                                                    debug=self.debug)
        if user_done:
            self.conversation_ended = True
        self.logs.append(TurnLog(
            turn=len(state.history) // 2 + 1, assistant_reply=assistant_reply,
            mode=mode, mode_reason=None, route=None, route_mode=None, judgment=None,
            p_central=None, relevance=None, argument_strength=None, cue_strength=None,
            interaction_pressure=pressure, dominant_cues=[], discrepancy=None, target=None,
            relevant_state_ids=[], bdi_before=bdi_before, proposed_bdi={},
            bdi_after=state.bdi_dict(),
            appraisal={"goal_congruence": appraisal.goal_congruence,
                       "coping_potential": appraisal.coping_potential,
                       "future_expectancy": appraisal.future_expectancy},
            desire_assessment=appraisal_prop.get("desire_assessment") or [],
            emotion_before={"valence": state.emotion.valence, "arousal": state.emotion.arousal,
                            "category": state.emotion.category},
            emotion_after={"valence": next_emotion.valence, "arousal": next_emotion.arousal,
                           "category": next_emotion.category},
            reaction_plan=reaction_plan, user_reply=user_reply, user_done=user_done,
            done_reason=done_reason, update_notes=notes))
        state.emotion = next_emotion
        state.history.append({"role": "assistant", "text": assistant_reply})
        state.history.append({"role": "user", "text": user_reply})
        return user_reply


def run_m0(llm, state, script: list[str]) -> tuple[list[dict], list[str]]:
    replies = [run_m0_turn(llm, state, m) for m in script]
    return [{"assistant": a, "user": u} for a, u in zip(script, replies)], replies


def run_m1(llm, state, script: list[str]) -> M1Simulator:
    sim = M1Simulator(state, llm)
    for m in script:
        if sim.conversation_ended:
            break
        sim.simulate_turn(m)
    return sim


def run_m2(llm, state, script: list[str]) -> UserSimulator:
    sim = UserSimulator(state, llm)
    for m in script:
        if sim.conversation_ended:
            break
        sim.simulate_turn(m)
    return sim
