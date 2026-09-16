"""verl RL 集成的环境适配层（03§3 合同 + SFT_SPEC 生成配置）。

职责（不依赖 vllm 引擎细节，纯 Python）：
1. prompt 数据集构建：种子 → verl 数据格式
   {"prompt": [{"role": "system", ...}, {"role": "user", ...}, ...],
    "seed_id": ...}——prompt 以 user 消息结尾（agent 先发言口径）；
2. 自定义 reward：verl RewardManager 的 compute_score 适配——
   对 rollout 生成的完整对话跑 TaskEnv + judge_with_aggregation，
   返回终局回报（04§2：esconv G=(E+A)/8、p4g commitment 1/0、
   cb clip(raw_SL,0,1)）与 TurnRecord 成本；
3. 生成配置常量：与 ab_validate 同一套（贪心 + suppress + rp + ngram +
   首段截断），verl 侧采样配置必须引用本文件常量。

verl 接入约定（写代码时核对 verl 0.9 API）：
- 数据流：verl 的 custom dataset 提供 prompts → rollout 引擎生成 →
  compute_score(data_source, solution_str) 返回 reward；
- 多轮：首版最小 smoke 先做"单包 rollout 完成后整段 judge"（不逐轮
  奖励），与 03§3 的终局监督口径一致；
- 标签 trie 采样（两字段输出）在 RL smoke 第二阶段接入（trie_qwen）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from cstpo.agent import SYSTEM_BUILDERS, agent_turn
from cstpo.judge import judge_with_aggregation
from cstpo.task_env import TaskEnv, prefix_turns

# 生成配置（SFT_SPEC 冻结；verl 采样参数必须与此一致）
GEN_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 128,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "suppress_tokens": ("<|url|>", "<|code|>", "<|audio|>", "<|video|>",
                        "<|quote|>", "<think>", "</think>"),
    "enable_thinking": False,
}

SAFETY_CAP = 30


def build_prompt(seed: dict) -> dict:
    """种子 → verl prompt（messages 以 user 结尾；agent 从下一条开始生成）。"""
    system = SYSTEM_BUILDERS[seed["task"]["task_id"]](seed)
    msgs = [{"role": "system", "content": system}]
    for t in prefix_turns(seed):
        msgs.append({"role": "assistant" if t["role"] == "assistant" else "user",
                     "content": t["text"]})
    return {"prompt": msgs, "seed_id": seed["seed_id"],
            "task": seed["task"]["task_id"]}


def run_episode(seed: dict, generated_turns: list[str]) -> dict:
    """rollout 生成的 agent 话语序列 → 环境完整对话 → 终局回报与成本。

    generated_turns：verl 生成的每轮 agent 话语（不含前缀）。
    环境侧：TaskEnv + Cog-Sim（15 轮 judger 兜底，与训练数据口径一致）。
    """
    from cstpo.stale_judger import check_stale, should_check
    from cstpo.cost_ledger import RecordingLLM
    from simulator.llm import LLMClient

    task = seed["task"]["task_id"]
    llm = RecordingLLM()
    env = TaskEnv(llm=llm, max_turns=SAFETY_CAP)
    cp = env.reset(seed)["checkpoint"]
    turns = prefix_turns(seed)
    reason = None
    for i, utter in enumerate(generated_turns):
        if should_check(i):
            j = check_stale(LLMClient(), turns, task)
            if j["stale"] and j["final_line"]:
                turns.append({"role": "user", "text": j["final_line"]})
                reason = "judger_stale_end"
                break
        utter = (utter or "").strip() or "(keep talking)"
        r = env.step(cp, utter)
        cp = r["checkpoint"]
        turns.append({"role": "assistant", "text": utter})
        turns.append({"role": "user", "text": r["user_reply"]})
        if r["terminated"]:
            reason = r["termination_reason"]
            break
    return {"turns": turns, "termination": reason,
            "seed": seed, "costs": llm.usage_report()}


def compute_reward(seed: dict, generated_turns: list[str]) -> dict:
    """终局回报（04§2）+ judge 聚合明细。"""
    task = seed["task"]["task_id"]
    ep = run_episode(seed, generated_turns)
    situation = seed.get("persona", {}).get("situation_en")
    emotion = seed.get("initial_emotion", {}).get("category")
    agg = judge_with_aggregation(LLMClient(), task, ep["turns"],
                                 situation, emotion)
    if task == "esconv":
        return {"reward": (agg["E"] + agg["A"]) / 8.0, "E": agg["E"],
                "A": agg["A"], "termination": ep["termination"]}
    if task == "p4g":
        r = 1.0 if (agg["commitment"] and not agg["withdrawn"]) else 0.0
        return {"reward": r, "commitment": agg["commitment"],
                "termination": ep["termination"]}
    # cb：raw_SL clip（需种子双方 target；程序化，不进 judge 可见信息）
    from cstpo.terminal_judge import raw_sl
    p_buyer = seed["actor_task_view"]["target"]
    p_seller = seed["user_task_view"]["target"]
    sl = raw_sl(agg.get("final_price"), p_seller, p_buyer, agg.get("deal"))
    return {"reward": min(max(sl, 0.0), 1.0), "deal": agg["deal"],
            "raw_sl": sl, "termination": ep["termination"]}


def verl_compute_score(data_source, solution_str, **kwargs):
    """verl RewardManager.compute_score 适配签名（接入时按 verl 0.9 实际
    签名核对；solution_str = 多轮话语的约定拼接格式，接入时定义）。"""
    # 占位：实际接入在 verl smoke 阶段完成（需要 verl 的 rollout 格式定义）
    raise NotImplementedError("接入 verl 时实现：解析 solution_str → "
                              "generated_turns → compute_reward")


if __name__ == "__main__":
    # 离线自检：一个种子的 prompt 构建 + 假生成一轮的 reward 链路
    seeds = sorted((ROOT / "data" / "seeds_draft" / "esconv").glob("*.json"))
    seed = json.loads(seeds[0].read_text())
    p = build_prompt(seed)
    assert p["prompt"][-1]["role"] == "user", "prompt 应以 user 结尾"
    print("prompt 自检 PASS:", len(p["prompt"]), "条消息，种子", p["seed_id"])
    print("GEN_CONFIG:", GEN_CONFIG)
