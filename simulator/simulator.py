"""Simulator 主循环（文档 §24 伪代码的落地实现）。

每轮：a_t -> Mode -> (RJ -> ΔBDI | Reveal | No Change) -> Appraisal -> Emotion -> u_{t+1}
"""
from __future__ import annotations

import json
from pathlib import Path

from simulator.affect.appraisal import normalize_appraisal
from simulator.affect.emotion import update_emotion
from simulator.cognitive.cognitive_engine import propose_cognitive_update
from simulator.cognitive.judgment_controller import control_judgment
from simulator.cognitive.rj_contract import build_rj_contract
from simulator.cognitive.route_controller import control_route
from simulator.generation.conversation_end import classify_user_done
from simulator.generation.user_response import generate_utterance, respond_without_bdi_change
from simulator.llm import LLMClient
from simulator.routing.mode_classifier import classify_mode
from simulator.state.schema import TurnLog, UserState
from simulator.state.updater import constrained_apply


class UserSimulator:
    def __init__(self, state: UserState, llm: LLMClient | None = None):
        self.state = state
        self.llm = llm or LLMClient()
        self.logs: list[TurnLog] = []
        self.conversation_ended = False

    def simulate_turn(self, assistant_reply: str) -> str:
        """文档 §24：单轮模拟，返回模拟用户的下一轮话语。

        若用户本轮表达了结束对话的意图（任务中立信号，非任务成败），
        self.conversation_ended 置为 True，驱动方应停止继续追问。
        """
        user = self.state
        turn = len(user.history) // 2 + 1

        # 1) 模式分类
        mode = classify_mode(self.llm, user.history, assistant_reply)
        bdi_before = user.bdi_dict()

        route = judgment = target = d_t = p_C = support_q = None
        reaction_plan: str | None = None
        appraisal_prop: dict = {}
        notes: list[str] = []

        if mode == "influence":
            # 2) Route（文档 §9）：语义特征 + eta_R
            r_info = control_route(self.llm, user.profile, user, assistant_reply)
            route, p_C = r_info["route"], r_info["p_central"]

            # 3) Judgment（文档 §10）：语义距离 + tau 阈值
            j_info = control_judgment(self.llm, user.profile, user, assistant_reply)
            target, d_t, support_q = j_info["target"], j_info["discrepancy"], j_info["support_quality"]
            judgment = j_info["judgment"]

            # 4) RJ Contract（文档 §11）
            contract = build_rj_contract(route, judgment)

            # 5) Cognitive Engine 提出候选更新（文档 §12、§13）
            proposal = propose_cognitive_update(self.llm, user, assistant_reply, contract)

            # 6) Deterministic Updater 施加约束（文档 §14）
            proposed_bdi = {
                "bdi_updates": proposal.get("bdi_updates", []),
                "new_items": proposal.get("new_items", []),
            }
            notes += constrained_apply(
                user, proposal, route, judgment, j_info["related_belief_ids"]
            )
            reaction_plan = proposal.get("reaction_plan")
            appraisal_prop = proposal
        else:
            # Elicit：State Revelation；Social：Affective Interaction，BDI 均不变
            proposal = respond_without_bdi_change(self.llm, user, mode)
            proposed_bdi = {}
            appraisal_prop = proposal
            reaction_plan = proposal.get("reaction_plan")
            if mode == "elicit":
                revealed = proposal.get("revealed_items", []) or []
                if revealed:
                    notes.append(f"Elicit 揭示已有节点: {revealed}")

        # 7) Appraisal（文档 §15）：只做归一化钳制
        appraisal, a_notes = normalize_appraisal(appraisal_prop)
        notes += a_notes

        # 8) Emotion 更新（文档 §16、§17）：Appraisal 定目标 + 惯性平滑
        next_emotion, e_notes = update_emotion(user.emotion, appraisal_prop, appraisal, mode)
        notes += e_notes

        # 9) 自然语言生成（文档 §18、§19）
        if mode == "influence":
            user_reply = generate_utterance(self.llm, user, reaction_plan)
        else:
            user_reply = proposal.get("utterance", "").strip()
            if not user_reply:
                user_reply = generate_utterance(self.llm, user, reaction_plan)

        # 10) 对话结束信号（任务中立）：用户是否想结束本次对话
        user_done, done_reason = classify_user_done(self.llm, user, user_reply)
        if user_done:
            self.conversation_ended = True

        # 记录 + 推进历史
        self.logs.append(TurnLog(
            turn=turn,
            assistant_reply=assistant_reply,
            mode=mode,
            route=route,
            judgment=judgment,
            p_central=round(p_C, 3) if p_C is not None else None,
            discrepancy=d_t,
            target=target,
            support_quality=support_q,
            bdi_before=bdi_before,
            proposed_bdi=proposed_bdi,
            bdi_after=user.bdi_dict(),
            appraisal={"goal_congruence": appraisal.goal_congruence,
                       "coping_potential": appraisal.coping_potential,
                       "future_expectancy": appraisal.future_expectancy},
            emotion_before={"valence": user.emotion.valence, "arousal": user.emotion.arousal,
                            "category": user.emotion.category},
            emotion_after={"valence": next_emotion.valence, "arousal": next_emotion.arousal,
                           "category": next_emotion.category},
            reaction_plan=reaction_plan,
            user_reply=user_reply,
            user_done=user_done,
            done_reason=done_reason,
            update_notes=notes,
        ))
        user.emotion = next_emotion
        user.history.append({"role": "assistant", "text": assistant_reply})
        user.history.append({"role": "user", "text": user_reply})
        return user_reply

    def dump_logs(self, path: str | Path) -> None:
        Path(path).write_text(
            "\n".join(json.dumps(l.__dict__, ensure_ascii=False, default=str) for l in self.logs),
            encoding="utf-8",
        )

