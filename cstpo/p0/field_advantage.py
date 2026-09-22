"""字段优势计算（03§5）：A_k / A_u 的固定公式实现。

纯函数、无模型依赖；手算验收例（03§5）：

    V=.4, U_k=.6, G11=1, G12=0
    普通节点   A_k = 1 - .4  = .6，  A_u = 1 - .6 = .4
    完整包节点 A_k = (1+0)/2 - .4 = .1，A_u = 1 - .6 = .4（A_u 不变）

k2 的回报不直接进入这两个优势值（只用于 U 监督与诊断，03§5）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FieldAdvantage:
    """一个主干动作的两个字段优势。

    - A_k（strategy）：主干终局回报（有完整同策略尾部时取两同策略实现均值）
      减去旧 critic 的 V 基线。
    - A_u（utterance）：原话语终局回报减去旧 critic 的策略条件基线 U(x,k1)。
    - stop_gradient 由调用方保证：输入必须是冻结 critic 的旧预测（03§5）。
    """
    A_k: float
    A_u: float
    used_branch_mean: bool


def plain_node_advantage(G11: float, V_old: float, U_old_k1: float) -> FieldAdvantage:
    """普通主干节点：A_k = G11 - V_old(x)；A_u = G11 - U_old(x,k1)。"""
    return FieldAdvantage(A_k=G11 - V_old, A_u=G11 - U_old_k1,
                          used_branch_mean=False)


def branched_node_advantage(G11: float, G12: float, V_old: float,
                            U_old_k1: float) -> FieldAdvantage:
    """完整包分叉节点：A_k = (G11+G12)/2 - V_old(x)；A_u 不变。

    仅当同策略替代措辞尾部（G12）完整完成时使用；部分包不得用其均值
    产生策略分支优势（01§3、03§5）。
    """
    return FieldAdvantage(A_k=(G11 + G12) / 2.0 - V_old,
                          A_u=G11 - U_old_k1,
                          used_branch_mean=True)
