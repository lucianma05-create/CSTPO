"""终局 judge 骨架（04§18 草案的落地边界）。

- `judge_esconv/p4g/craigslist` 是评分协议契约（输入输出 schema + 锚点说明），
  真实 judge 提示与模型在 P1 校准后冻结接入；本模块提供 fake judge 供 P0 单包往返。
- CB 的 raw_SL 为程序化计算（04§2.3、§18.4）：judge 只输出成交与价格，
  收益由本模块计算，judge 不读双方初始目标价。
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------- judge 输出契约（04§18） ----------

@dataclass
class ESConvVerdict:
    E: int                      # 0-4 情绪/希望改善
    A: int                      # 0-4 可行行动打算
    evidence_e: list            # [{turn, quote}]
    evidence_a: list
    evidence_sufficient: bool
    notes: str = ""


@dataclass
class P4GVerdict:
    commitment: bool            # 明确、自愿、非纯条件性捐赠承诺
    amount: float | None
    conditional: bool
    withdrawn: bool
    evidence: list
    notes: str = ""


@dataclass
class BargainVerdict:
    deal: bool                  # 双方一致且未撤回的成交承诺
    final_price: float | None
    currency: str | None
    withdrawn: bool
    parse_error: bool           # 成交却无法可靠解析价格 = 评价故障
    evidence: list
    notes: str = ""


# ---------- CB 程序化收益（04§2.3） ----------

def raw_sl(final_price: float | None, p_seller: float, p_buyer: float,
           deal: bool) -> float:
    """raw_SL = (p - p_s) / (p_b - p_s)（Actor=buyer 版本）。

    未成交 raw_SL=0；成交指示独立记录（04§2.3）。越界价格单独统计，
    不自动判对话无效。换 Seller Actor 时公式改为 (p-p_b)/(p_s-p_b)。
    """
    if not deal or final_price is None or p_buyer == p_seller:
        return 0.0
    return (final_price - p_seller) / (p_buyer - p_seller)


def train_reward_bargain(verdict: BargainVerdict, p_seller: float,
                         p_buyer: float) -> float:
    """训练回报 = clip(raw_SL, 0, 1)（04§2.3；未裁剪值单独报告）。"""
    sl = raw_sl(verdict.final_price, p_seller, p_buyer, verdict.deal)
    return min(max(sl, 0.0), 1.0)


def train_reward_p4g(verdict: P4GVerdict) -> float:
    """原始终局 +1/-1 → 1/0 仿射（04§2.2）。"""
    return 1.0 if verdict.commitment and not verdict.withdrawn else 0.0


def train_reward_esconv(verdict: ESConvVerdict) -> float:
    """G = (E + A) / 8（04§2.1）；E、A、联合达标率单独报告。"""
    return (verdict.E + verdict.A) / 8.0


# ---------- fake judge（P0 单包往返用；esconv/p4g 变体已删：全库零引用） ----------

def fake_judge_bargain(text: str) -> BargainVerdict:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    price = float(m.group(1)) if m else None
    deal = ("deal" in text.lower()) or ("成交" in text)
    return BargainVerdict(deal=deal, final_price=price, currency="CNY",
                          withdrawn=False, parse_error=(deal and price is None),
                          evidence=[], notes="fake")
