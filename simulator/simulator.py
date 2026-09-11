"""Simulator 主循环（改进文档 0911 §39 伪代码的落地实现）。

每轮 Influence：Route 特征提取(§14) -> Route 决策(§16、§17) ->
Discrepancy(§19) -> Judgment(§20) -> Contract(§22-27) ->
Engine 提案(§28-30) -> Deterministic Updater(§31) ->
Appraisal(§32) -> Emotion(§33、§34) -> NLG(§37、§38)。
Elicit / Social：BDI 冻结，只更新情绪并生成回复（§35、§36）。
"""
from __future__ import annotations

import json
from pathlib import Path

from simulator.affect.appraisal import normalize_appraisal
from simulator.affect.emotion import measure_bdi_change, update_emotion
from simulator.cognitive.cognitive_engine import propose_cognitive_update
from simulator.cognitive.rj_contract import build_rj_contract
from simulator.generation.conversation_end import classify_user_done
from simulator.generation.user_response import generate_utterance, respond_without_bdi_change
from simulator.llm import LLMClient
from simulator.profile.cognitive_profile import map_level
from simulator.routing.discrepancy import estimate_discrepancy
from simulator.routing.judgment_controller import control_judgment
from simulator.routing.mode_classifier import classify_mode
from simulator.routing.route_controller import decide_route
from simulator.routing.route_features import extract_route_features
from simulator.state.schema import TurnLog, UserState
from simulator.state.updater import constrained_apply


class UserSimulator:
    def __init__(self, state: UserState, llm: LLMClient | None = None,
                 route_mode: str = "deterministic"):
        self.state = state
        self.llm = llm or LLMClient()
        # Route 采样模式（文档 §17、§17.1）：bernoulli 保留随机性，
        # deterministic（p_C>=0.5 取 Central）用于可复现 benchmark。
        self.route_mode = route_mode
        self.logs: list[TurnLog] = []
        self.conversation_ended = False
        self.prev_gc: float | None = None   # §34 的 |ΔGC| 需要上一轮 GC（首轮无基线）

    def simulate_turn(self, assistant_reply: str) -> str:
        """文档 §39：单轮模拟，返回模拟用户的下一轮话语。

        若用户本轮表达了结束对话的意图（任务中立信号，非任务成败），
        self.conversation_ended 置为 True，驱动方应停止继续追问。
        """
        user = self.state
        turn = len(user.history) // 2 + 1

        # 1) 模式分类（§8、§9）
        mode, mode_reason = classify_mode(self.llm, user.history, assistant_reply)
        bdi_before = user.bdi_dict()

        route = judgment = target = d_t = p_C = None
        feats: dict = {}
        relevant_ids: list[str] = []
        pressure = 0.5   # 交互压力（§34）：Influence 由 route features 标注，其余由合并调用标注
        reaction_plan: str | None = None
        appraisal_prop: dict = {}
        notes: list[str] = []

        if mode == "influence":
            # 2) Route 特征提取（§13、§14）：三特征 + 交互压力 + target_proposition，一次调用
            feats = extract_route_features(self.llm, user, assistant_reply)
            pressure = feats.get("interaction_pressure", 0.5)

            # 3) Route 决策（§16、§17）：程序公式 + 采样模式
            route, p_C = decide_route(
                user.profile.eta_R,
                feats["relevance"], feats["argument_strength"], feats["cue_strength"],
                mode=self.route_mode,
            )

            # 4) Cognitive Discrepancy（§18、§19）：语义立场距离 + 相关节点
            d_info = estimate_discrepancy(self.llm, user, feats["target_proposition"])
            target, d_t = d_info["target"], d_info["stance_distance"]
            relevant_ids = d_info["relevant_state_ids"]

            # 5) Judgment（§20）：程序侧阈值映射
            judgment = control_judgment(user.profile, d_t)

            # 6) RJ Contract（§22-27）
            contract = build_rj_contract(route, judgment)

            # 7) Cognitive Engine 提出候选更新（§28、§29、§30）
            proposal = propose_cognitive_update(
                self.llm, user, assistant_reply, contract, route, judgment
            )

            # 8) Deterministic Updater 施加约束（§31）
            proposed_bdi = {
                "bdi_updates": proposal.get("bdi_updates", []),
                "new_items": proposal.get("new_items", []),
            }
            belief_ids = [i for i in relevant_ids
                          if (it := user.find(i)) is not None and it.type == "belief"]
            notes += constrained_apply(user, proposal, route, judgment, belief_ids)
            reaction_plan = proposal.get("reaction_plan")
            appraisal_prop = proposal
        else:
            # Elicit：State Revelation（§35）；Social：Affective Interaction（§36），BDI 均不变
            proposal = respond_without_bdi_change(self.llm, user, assistant_reply, mode)
            proposed_bdi = {}
            appraisal_prop = proposal
            reaction_plan = proposal.get("reaction_plan")
            pressure = map_level(proposal.get("interaction_pressure", "medium"))
            if mode == "elicit":
                revealed = proposal.get("revealed_items", []) or []
                if revealed:
                    notes.append(f"Elicit 揭示已有节点: {revealed}")

        # 9) Appraisal（§32）：钳制提议值；GC 用 desire_assessment + 更新后 BDI
        #    按 §32.1 公式重算（user 此刻已是 Updater 约束后的状态）
        appraisal, a_notes = normalize_appraisal(appraisal_prop, user)
        notes += a_notes
        # 审计提示：CP/FE 仍由 LLM 在 Updater 约束前提议，
        # 若本回合存在被约束的更新，其数值基于的局势与实际状态变化可能不一致
        if mode == "influence" and any(k in n for n in notes for k in ("截断", "禁止", "拒绝")):
            notes.append("审计提示：CP/FE 基于 Updater 约束前的提案，本回合存在被约束的更新")

        # 10) Emotion 更新（§33、§34）：Appraisal 定 valence 目标，§34 公式定 arousal 目标
        delta_bdi = measure_bdi_change(bdi_before, user.bdi_dict())
        gc_shift = abs(appraisal.goal_congruence - self.prev_gc) if self.prev_gc is not None else 0.0
        next_emotion, e_notes = update_emotion(
            user.emotion, appraisal_prop, appraisal, mode,
            delta_bdi=delta_bdi, gc_shift=gc_shift, pressure=pressure,
        )
        notes += e_notes
        self.prev_gc = appraisal.goal_congruence

        # 11) 自然语言生成（§37、§38）
        if mode == "influence":
            user_reply = generate_utterance(self.llm, user, reaction_plan)
        else:
            user_reply = proposal.get("utterance", "").strip()
            if not user_reply:
                user_reply = generate_utterance(self.llm, user, reaction_plan)

        # 12) 对话结束信号（任务中立）：用户是否想结束本次对话
        user_done, done_reason = classify_user_done(self.llm, user, user_reply)
        if user_done:
            self.conversation_ended = True

        # 记录 + 推进历史
        self.logs.append(TurnLog(
            turn=turn,
            assistant_reply=assistant_reply,
            mode=mode,
            mode_reason=mode_reason,
            route=route,
            route_mode=self.route_mode if mode == "influence" else None,
            judgment=judgment,
            p_central=round(p_C, 3) if p_C is not None else None,
            relevance=feats.get("relevance"),
            argument_strength=feats.get("argument_strength"),
            cue_strength=feats.get("cue_strength"),
            interaction_pressure=pressure,
            dominant_cues=feats.get("dominant_cues", []),
            discrepancy=d_t,
            target=target,
            relevant_state_ids=relevant_ids,
            bdi_before=bdi_before,
            proposed_bdi=proposed_bdi,
            bdi_after=user.bdi_dict(),
            appraisal={"goal_congruence": appraisal.goal_congruence,
                       "coping_potential": appraisal.coping_potential,
                       "future_expectancy": appraisal.future_expectancy},
            desire_assessment=appraisal_prop.get("desire_assessment") or [],
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
