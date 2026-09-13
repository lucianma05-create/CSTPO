"""CSTPO checkpoint 模块：UserSimulator 完整状态快照与恢复。

修复提案 05 问题 6（方案 B）：快照 API 属于集成要求而非 Cog-Sim 缺陷，
由 CSTPO 侧按 03§3 TaskEnv.reset 合同实现，不修改冻结模拟器。
首版 deterministic 路由无随机流问题（01§2）；bernoulli 模式的 rng 隔离
留待需要时另行裁定最小改动（给 UserSimulator 增加可选 rng 透传）。

快照内容：UserState（深拷贝）、prev_gc、conversation_ended、route_mode、
debug、logs（深拷贝）。LLM 客户端不入快照（网络连接按 01§2 独立管理），
restore 时由调用方提供。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

COG_SIM = Path(__file__).resolve().parents[1] / "Cog-Sim"
if str(COG_SIM) not in sys.path:
    sys.path.insert(0, str(COG_SIM))

from simulator.simulator import UserSimulator


def snapshot(sim: UserSimulator) -> dict:
    """保存分支可恢复的完整模拟器状态，返回纯数据 dict（不含 LLM 客户端）。"""
    return {
        "state": copy.deepcopy(sim.state),
        "prev_gc": sim.prev_gc,
        "conversation_ended": sim.conversation_ended,
        "route_mode": sim.route_mode,
        "debug": sim.debug,
        "logs": copy.deepcopy(sim.logs),
    }


def restore(snap: dict, llm) -> UserSimulator:
    """按快照重建 UserSimulator；state/logs 深拷贝，保证分支间无对象共享。"""
    sim = UserSimulator(
        state=copy.deepcopy(snap["state"]),
        llm=llm,
        route_mode=snap["route_mode"],
        debug=snap["debug"],
    )
    sim.prev_gc = snap["prev_gc"]
    sim.conversation_ended = snap["conversation_ended"]
    sim.logs = copy.deepcopy(snap["logs"])
    return sim
