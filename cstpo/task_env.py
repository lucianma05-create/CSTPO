"""TaskEnv：03§3 合同的环境骨架（冻结种子 → 模拟器状态 → step 循环）。

- reset(seed)：编译 UserState（prefix_reset：前缀直接作为 history），返回
  {o, x, checkpoint}——o 为 Actor 可见观察（history + actor_task_view），
  x 为训练侧状态（o + user_task_view + persona + BDI + emotion + prev_gc +
  任务事件 + 剩余轮数）。
- step(checkpoint, u)：调用 v1.0.1-fix simulate_turn，返回用户回复、新
  checkpoint、终止原因（user_ended / time_limit / continue）与逐组件成本
  （LLMClient 计数增量）。

首版约定：
- 情绪 valence/arousal 用按类别默认表（controlled_default，02 §12 校准待办）；
- persona 文本由 facts 确定性渲染（LLM 渲染留待后续优化）；
- 前缀同角色消息保留为连续 history 条目（模拟器不合并）。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient
from simulator.simulator import UserSimulator
from simulator.state.schema import BDIItem, Emotion, UserState

from cstpo.checkpoint import snapshot

# category → (valence, arousal)：controlled_default，02 §12 校准前占位
EMOTION_DEFAULTS = {
    "neutral": (0.0, 0.3), "sadness": (-0.5, 0.3), "anxiety": (-0.5, 0.6),
    "frustration": (-0.4, 0.5), "anger": (-0.5, 0.7), "interest": (0.3, 0.6),
    "hope": (0.5, 0.5), "relief": (0.5, 0.2), "satisfaction": (0.6, 0.3),
}

# 种子角色 → 模拟器 history 角色（assistant=Agent 发言，user=模拟用户发言）
USER_ROLES = {"seeker", "persuadee", "seller"}


# 实验前缀轮数：只取截止前缀的前 2 回合（4 条消息）作为可见历史起点。
# 全量前缀（ESConv/P4G 9 回合）已接近真实对话尾声，续写空间不足；
# CB 前缀本为 1 回合，不受影响。注：persona/BDI/情绪仍来自全量前缀提取
# （种子 v4.2），可见历史与认知初始化存在"超前"偏差，正式实验需种子 v5。
PREFIX_ROUNDS = 2


def prefix_turns(seed: dict, n_rounds: int | None = None) -> list[dict]:
    """种子截止前缀 → [{role:'assistant'|'user', text}]（与 compile_state 同一
    角色映射）。默认取前 PREFIX_ROUNDS 回合（实验口径）；n_rounds 显式
    指定时取前 n_rounds 回合。Agent 可见历史与提示式模拟器历史都应从该前缀继续。"""
    turns = []
    for m in seed["context"].get("prefix", []):
        role = "user" if m["role"] in USER_ROLES else "assistant"
        turns.append({"role": role, "text": m["text_en"]})
    if n_rounds is None:
        n_rounds = PREFIX_ROUNDS
    return turns[:2 * n_rounds]


def render_persona(seed: dict) -> str:
    """v1 确定性渲染：任务说明 + persona 事实 +（CB）物品事实。"""
    parts = [seed["task"]["description"]]
    for f in seed["persona"]["facts"]:
        parts.append(f"- {f.get('fact_en', '')} (evidence: {f.get('evidence', '')})")
    if seed["task"]["task_id"] == "craigslistbargain":
        item = seed["persona"].get("item_facts", {})
        parts.append(f"- Item: {item.get('title')} | category {item.get('category')} "
                     f"| listed ${item.get('price')}")
    return "\n".join(parts)


def compile_state(seed: dict) -> UserState:
    """种子 → UserState（v4.2 冻结种子专用编译路径）。"""
    state = UserState(persona=render_persona(seed))
    bdi = seed["initial_bdi"]
    for i, b in enumerate(bdi["beliefs"], 1):
        state.beliefs.append(BDIItem(f"B{i}", "belief", b["content_en"],
                                     b.get("strength", 2.0), core=b.get("core", True)))
    for i, d in enumerate(bdi["desires"], 1):
        state.desires.append(BDIItem(f"D{i}", "desire", d["content_en"],
                                     d.get("strength", 2.0), core=d.get("core", True),
                                     polarity=d.get("polarity", "approach")))
    for i, it in enumerate(bdi["intentions"], 1):
        state.intentions.append(BDIItem(f"I{i}", "intention", it["content_en"],
                                        it.get("strength", 2.0),
                                        core=it.get("core", True),
                                        polarity=it.get("polarity", "approach")))
    cat = seed["initial_emotion"]["category"]
    v, a = EMOTION_DEFAULTS.get(cat, EMOTION_DEFAULTS["neutral"])
    state.emotion = Emotion(valence=v, arousal=a, category=cat)
    for m in seed["context"]["prefix"][:2 * PREFIX_ROUNDS]:
        role = "user" if m["role"] in USER_ROLES else "assistant"
        state.history.append({"role": role, "text": m["text_en"]})
    return state


class TaskEnv:
    def __init__(self, llm: LLMClient | None = None, max_turns: int = 8,
                 debug: bool = True, route_mode: str = "deterministic"):
        self.llm = llm
        self.max_turns = max_turns
        self.debug = debug
        self.route_mode = route_mode

    def reset(self, seed: dict) -> dict:
        """03§3 TaskEnv.reset：Seed → o、x、Checkpoint。"""
        self.seed = seed
        state = compile_state(seed)
        sim = UserSimulator(state, llm=self.llm, route_mode=self.route_mode,
                            debug=self.debug)
        o = {"history": copy.deepcopy(state.history),
             "actor_task_view": seed.get("actor_task_view", {})}
        x = {"history": copy.deepcopy(state.history),
             "actor_task_view": seed.get("actor_task_view", {}),
             "user_task_view": seed.get("user_task_view", {}),
             "persona": state.persona,
             "profile": {"eta_R": state.profile.eta_R, "tau_A": state.profile.tau_A,
                         "tau_R": state.profile.tau_R},
             "bdi": state.bdi_dict(),
             "emotion": {"valence": state.emotion.valence,
                         "arousal": state.emotion.arousal,
                         "category": state.emotion.category},
             "prev_gc": None,
             "task_events": seed["context"].get("task_events_in_prefix", []),
             "remaining_turns": self.max_turns,
             }
        checkpoint = {"sim_snapshot": snapshot(sim), "turns_used": 0,
                      "seed_id": seed["seed_id"]}
        return {"o": o, "x": x, "checkpoint": checkpoint, "sim": sim}

    def step(self, checkpoint: dict, utterance: str) -> dict:
        """03§3 TaskEnv.step：Checkpoint、u → 用户回复、新 Checkpoint、
        终止原因、成本。"""
        from cstpo.checkpoint import restore
        sim = restore(checkpoint["sim_snapshot"], llm=self.llm)
        before = (sim.llm.calls, sim.llm.prompt_tokens, sim.llm.completion_tokens)
        user_reply = sim.simulate_turn(utterance)
        after = (sim.llm.calls, sim.llm.prompt_tokens, sim.llm.completion_tokens)
        costs = {"calls": after[0] - before[0],
                 "prompt_tokens": after[1] - before[1],
                 "completion_tokens": after[2] - before[2]}
        turns = checkpoint["turns_used"] + 1
        if sim.conversation_ended:
            terminated, reason = True, "user_ended"
        elif turns >= self.max_turns:
            terminated, reason = True, "time_limit"
        else:
            terminated, reason = False, "continue"
        new_checkpoint = {"sim_snapshot": snapshot(sim), "turns_used": turns,
                          "seed_id": checkpoint["seed_id"]}
        turn_record = {"seed_id": checkpoint["seed_id"], "turn": turns,
                       "assistant_reply": utterance, "user_reply": user_reply,
                       "terminated": terminated, "reason": reason,
                       "costs": costs, "mode": sim.logs[-1].mode,
                       "update_notes": sim.logs[-1].update_notes}
        return {"user_reply": user_reply, "checkpoint": new_checkpoint,
                "terminated": terminated, "termination_reason": reason,
                "costs": costs, "turn_record": turn_record, "sim": sim}
