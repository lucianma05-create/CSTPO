"""Emotion 更新（文档 §33、§34）：Appraisal 决定目标 valence，§34 公式决定目标 arousal。

v_hat = w1*GC + w2*CP + w3*FE，第一版 w1=w2=w3（文档 §33）。
v_{t+1} = (1-lambda_v) * v_t + lambda_v * v_hat_{t+1}

r_hat = gamma1*|ΔC| + gamma2*|ΔGC_t - ΔGC_{t-1}| + gamma3*InteractionPressure（文档 §34）：
- |ΔC|：本轮 BDI 实际变化幅度，Σ|Δs_i| / 4 归一化（程序测量 Updater 约束后的快照差）；
- |ΔGC|：本轮 GC 相对上一轮的突变（程序测量，首轮为 0）；
- InteractionPressure：LLM 语义标注 low/medium/high -> 0.2/0.5/0.8（程序映射，
  定义见 route_features.py / user_response.py，严格任务中立）。
第一版 gamma = (0.6, 0.25, 0.5)（文档未给具体值）。
r_{t+1} = (1-lambda_r) * r_t + lambda_r * r_hat

Influence / Elicit 用 lambda=0.4；Social 只允许低幅度变化，用 lambda=0.25。

category 一致性校验（0910 §16：category 根据 (v,r) 选择）：
LLM 提议的 category 只作参考，程序按最终 (v,r) 检查其效价符号/唤醒水平，
矛盾时改写为由 (v,r) 推导的类别并记录审计 note。
（emotion_proposal 现在只含 category；valence 与 arousal 均由程序计算。）
"""
from __future__ import annotations

from simulator.state.schema import EMOTION_CATEGORIES, Appraisal, Emotion
from simulator.utils import clamp

LAMBDA_INFLUENCE = 0.4
LAMBDA_SOCIAL = 0.25

# §34 第一版系数（文档未给具体值）
GAMMA_BDI = 0.6       # 认知变化幅度
GAMMA_GC = 0.25       # GC 突变
GAMMA_PRESS = 0.5     # 交互压力
DELTA_BDI_SCALE = 4.0  # |ΔC| 归一化：BDI 强度满量程

# 效价符号约束：负向/正向类别与 (v,r) 的一致性检查
NEGATIVE_CATEGORIES = {"sadness", "anxiety", "frustration", "anger"}
POSITIVE_CATEGORIES = {"satisfaction", "interest", "hope", "relief"}
SIGN_TOLERANCE = 0.05   # 惯性平滑后的边界容忍（避免临界值处抖动）
ANGER_MIN_AROUSAL = 0.5  # anger 是高唤醒类别


def measure_bdi_change(before: dict, after: dict) -> float:
    """Σ|Δs_i|：比较 BDI 前后快照（含新增/淘汰节点），未归一化。"""
    total = 0.0
    for key in ("beliefs", "desires", "intentions"):
        b = {i["id"]: float(i.get("strength", 0.0)) for i in before.get(key, [])}
        a = {i["id"]: float(i.get("strength", 0.0)) for i in after.get(key, [])}
        for iid in set(b) | set(a):
            total += abs(a.get(iid, 0.0) - b.get(iid, 0.0))
    return round(total, 4)


def compute_arousal_target(delta_bdi: float, gc_shift: float, pressure: float) -> tuple[float, str]:
    """§34 公式：返回 (r_hat, 审计分解文本)。"""
    delta_norm = min(delta_bdi / DELTA_BDI_SCALE, 1.0)
    r_hat = clamp(GAMMA_BDI * delta_norm + GAMMA_GC * gc_shift + GAMMA_PRESS * pressure, 0.0, 1.0)
    breakdown = (f"r̂ = {GAMMA_BDI}·|ΔC|({delta_norm:.2f}) + {GAMMA_GC}·|ΔGC|({gc_shift:.2f}) "
                 f"+ {GAMMA_PRESS}·压力({pressure:.2f}) = {r_hat:.3f}")
    return r_hat, breakdown


def derive_category(v: float, r: float) -> str:
    """由 (v, r) 推导展示用类别（0910 §16）。

    文档 §5 锚点：悲伤(-0.8,0.25)、焦虑(-0.6,0.85)、平静满足(0.7,0.2)、兴奋(0.8,0.9)。
    hope / relief 依赖面向未来/过去的评价，无法从 (v,r) 推出，仅在 LLM 提议
    且与数值一致时保留。
    """
    if abs(v) < 0.2:
        return "neutral"
    if v < 0:
        if r >= 0.6:
            return "anxiety"
        if r >= 0.3:
            return "frustration"
        return "sadness"
    if r >= 0.6:
        return "interest"
    return "satisfaction"


def _contradicts(category: str, v: float, r: float) -> bool:
    if category in NEGATIVE_CATEGORIES and v >= -SIGN_TOLERANCE:
        return True
    if category in POSITIVE_CATEGORIES and v <= SIGN_TOLERANCE:
        return True
    if category == "anger" and r < ANGER_MIN_AROUSAL:
        return True
    return False


def update_emotion(
    prev: Emotion,
    proposal: dict,
    appraisal: Appraisal,
    mode: str,
    delta_bdi: float = 0.0,
    gc_shift: float = 0.0,
    pressure: float = 0.5,
) -> tuple[Emotion, list[str]]:
    notes: list[str] = []
    lam = LAMBDA_SOCIAL if mode == "social" else LAMBDA_INFLUENCE
    prop = proposal.get("emotion_proposal", {}) or {}

    # 目标 valence 由 Appraisal 公式决定（文档 §33，w1=w2=w3）
    v_hat = clamp(
        (appraisal.goal_congruence + appraisal.coping_potential + appraisal.future_expectancy) / 3.0,
        -1.0, 1.0,
    )
    # 目标 arousal 由 §34 公式决定：认知变化 + GC 突变 + 交互压力
    r_hat, breakdown = compute_arousal_target(delta_bdi, gc_shift, pressure)
    notes.append(breakdown)

    category = str(prop.get("category", prev.category)).strip().lower()
    if category not in EMOTION_CATEGORIES:
        notes.append(f"非法 emotion category {category!r}，退回 neutral")
        category = "neutral"

    valence = round((1 - lam) * prev.valence + lam * v_hat, 3)
    arousal = round((1 - lam) * prev.arousal + lam * r_hat, 3)

    # category 一致性校验：LLM 提案 + 程序按 (v,r) 裁决
    if _contradicts(category, valence, arousal):
        implied = derive_category(valence, arousal)
        notes.append(f"category={category} 与 (v={valence:.2f}, r={arousal:.2f}) 矛盾，改写为 {implied}")
        category = implied

    next_emo = Emotion(valence=valence, arousal=arousal, category=category)
    return next_emo, notes
