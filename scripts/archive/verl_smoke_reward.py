"""verl 最小 smoke 的自定义 reward（链路验证用，非正式奖励）。

正式奖励见 verl_env_adapter.compute_reward（终局 judge 聚合）。
本模块只验证 verl→reward→update 链路：长度启发式。
"""
from __future__ import annotations


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    """verl 0.9 自定义 reward 签名（smoke 版）。"""
    n_words = len(str(solution_str).split())
    return {"score": min(n_words / 50.0, 1.0)}
