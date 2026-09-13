"""Emotion Update Engine（EUE，文档 §32–§34 的程序实现；2026-09-12 由
appraisal.py 与 emotion.py 合并，对应实验文档0912 §7）。

两部分：
1. Appraisal（§32）：当轮中间变量 A_t=(GC, CP, FE)，统一归一化到 [-1,1]，不入长期状态。
   - 提案来源（0912 收敛）：Influence 由独立调用 propose_appraisal 在 Updater
     约束后基于 C_{t+1} 与 Updater 审计产出（CP/FE 不再锚定未生效提案）；
     Elicit / Social 由合并调用（generation/user_response.py）产出。
   - GC 半程序化（§32.1）：LLM 只输出逐 Desire 的语义标注 (relevance, gc_i)，
     程序按公式 GC = Σ s_i·rel_i·gc_i / Σ s_i·rel_i 重算，权重取**更新后 BDI**
     中 active Desire 的强度 s_i（§4 ActiveGoals = TopK(s_i·rel_i)，K=2）；
     标注缺失/非法时退回 LLM 提议的 GC 值。
   - CP/FE 仍由 LLM 提议 + 程序钳制（文档未给公式）。
2. Emotion（§33、§34）：程序公式推导 E_{t+1}。

   v_hat = w1*GC + w2*CP + w3*FE，第一版 w1=w2=w3（文档 §33）。
   v_{t+1} = (1-lambda) * v_t + lambda * v_hat

   r_hat = gamma1*|ΔC| + gamma2*|ΔGC| + gamma3*InteractionPressure（文档 §34）：
   - |ΔC|：本轮 BDI 实际变化幅度，Σ|Δs_i| / 4 归一化（程序测量 Updater 约束后的快照差）；
   - |ΔGC|：本轮 GC 相对上一轮的突变（程序测量，首轮为 0）；
   - InteractionPressure：LLM 语义标注 low/medium/high -> 0.2/0.5/0.8（程序映射，
     定义见 route_features.py / user_response.py，严格任务中立）。
   第一版 gamma = (0.6, 0.25, 0.5)（文档未给具体值）。
   r_{t+1} = (1-lambda) * r_t + lambda * r_hat

   Influence / Elicit 用 lambda=0.4；Social 只允许低幅度变化，用 lambda=0.25。

   category 一致性校验（0910 §16：category 根据 (v,r) 选择）：
   LLM 提议的 category 只作参考，程序按最终 (v_{t+1}, r_{t+1}) 检查其效价符号/唤醒水平，
   矛盾时改写为由 (v,r) 推导的类别并记录审计 note。
   （emotion_proposal 现在只含 category；valence 与 arousal 均由程序计算。）
"""
from __future__ import annotations

from simulator.state.schema import EMOTION_CATEGORIES, Appraisal, Emotion, UserState
from simulator.llm import StructuredCallError
from simulator.utils import clamp

ACTIVE_GOALS_K = 2   # 文档 §4：ActiveGoals = TopK(s_i * rel_i)，第一版 K=2

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


# ---------- Appraisal（§32）----------

def compute_gc(state: UserState, assessments: list | None) -> tuple[float | None, list[str]]:
    """按 §32.1 公式用 (s_i, rel_i, gc_i) 重算 GC，返回 (gc, notes)。

    无有效标注时返回 (None, notes)，调用方退回 LLM 提议值。
    """
    notes: list[str] = []
    scored: list[tuple[str, float, float, float]] = []   # (id, s_i, rel_i, gc_i)
    for a in assessments or []:
        if not isinstance(a, dict):
            continue
        item = state.find(str(a.get("id", "")))
        if item is None or item.type != "desire" or not item.active:
            notes.append(f"desire_assessment 引用无效节点 {a.get('id')!r}，忽略")
            continue
        try:
            rel = clamp(float(a.get("relevance", 0.0)), 0.0, 1.0)
            gc = clamp(float(a.get("gc", 0.0)), -1.0, 1.0)
        except (TypeError, ValueError):
            notes.append(f"desire_assessment[{item.id}] 数值非法，忽略")
            continue
        scored.append((item.id, item.strength, rel, gc))
    if not scored:
        return None, notes
    # ActiveGoals = TopK(s_i * rel_i)，文档 §4 第一版 K=2
    top = sorted(scored, key=lambda x: x[1] * x[2], reverse=True)[:ACTIVE_GOALS_K]
    num = sum(s * rel * gc for _, s, rel, gc in top)
    den = sum(s * rel for _, s, rel, gc in top)
    if den <= 1e-9:
        notes.append("desire_assessment 有效权重为 0，GC 退回 LLM 提议值")
        return None, notes
    gc = clamp(num / den, -1.0, 1.0)
    notes.append(
        f"GC 按 §32.1 公式重算: Σ(s·rel·gc)/Σ(s·rel) = {gc:.3f} "
        f"(top{len(top)}: {[i for i, _, _, _ in top]})"
    )
    return gc, notes


def normalize_appraisal(proposal: dict, state: UserState | None = None) -> tuple[Appraisal, list[str]]:
    """钳制 LLM 提议值；state 传入时用 desire_assessment 按公式重算 GC。

    state 应为**更新后**的 UserState（Influence 在 Updater 之后调用），
    使 GC 权重反映实际生效的 Desire 强度（修复提案被截断时的时点错位）。
    """
    notes: list[str] = []
    prop = proposal.get("appraisal", {}) or {}

    def get_field(name: str) -> float:
        v = prop.get(name, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            notes.append(f"appraisal.{name} 非法值 {v!r}，取 0.0")
            return 0.0
        if not -1.0 <= v <= 1.0:
            notes.append(f"appraisal.{name}={v} 越界，钳制到 [-1,1]")
            return clamp(v, -1.0, 1.0)
        return v

    gc = get_field("goal_congruence")
    if state is not None:
        gc_calc, g_notes = compute_gc(state, proposal.get("desire_assessment"))
        notes += g_notes
        if gc_calc is not None:
            if abs(gc_calc - gc) > 1e-6:
                notes.append(f"GC 提议值 {gc:.2f} -> 公式值 {gc_calc:.3f}")
            gc = gc_calc

    appraisal = Appraisal(
        goal_congruence=round(gc, 3),
        coping_potential=round(get_field("coping_potential"), 3),
        future_expectancy=round(get_field("future_expectancy"), 3),
    )
    return appraisal, notes


# 共享 Prompt 片段（EUE 为规范所有者；SRR/ERR 合并调用复用，见 user_response.py）
CATEGORY_LIST = "neutral, sadness, anxiety, frustration, interest, hope, relief, satisfaction, anger"

DESIRE_ASSESSMENT_DEF = """"desire_assessment": assess each ACTIVE desire against THIS turn's
situation (the assistant's latest reply and its consequences). Only include
desires with relevance > 0. "relevance" in [0, 1] = how directly the situation
concerns that desire. "gc" in [-1, 1] = whether the situation promotes (+) or
hinders (-) that desire from the user's perspective. This is about the
situation's favorability, NOT about whether the desire strength should change."""

APPRAISAL_USER = """Persona:
{persona}

Updated BDI state (JSON, after this turn's cognitive changes):
{state_json}

Current emotion (before this turn's update):
{emotion}

Recent conversation:
{history}

Assistant's latest reply:
{reply}

Cognitive changes actually applied this turn (updater audit):
{cog_changes}

Appraise THIS turn's situation relative to the user's UPDATED state, and return
exactly one JSON object:
{{
  "appraisal": {{"goal_congruence": 0.0, "coping_potential": 0.0, "future_expectancy": 0.0}},
  "desire_assessment": [{{"id": "D1", "relevance": 0.8, "gc": -0.3}}],
  "emotion_proposal": {{"category": "neutral"}}
}}

Operational definitions:
- Appraisal values are in [-1, 1].
- goal_congruence: how favorable the situation is to the user's important
  desires NOW (given the updated state). Judge only the USER's own desires —
  never the assistant's task objective.
- coping_potential: whether the user feels able to act on the situation.
- future_expectancy: whether an acceptable outcome seems achievable.
- {desire_assessment_def}
- The program may recompute goal_congruence from desire_assessment; still fill
  in appraisal.goal_congruence with your best estimate as a fallback.
- (emotion_proposal only takes a category — the program derives valence and
  arousal. category must be one of: {category_list}.)
- Only output the JSON object."""


def propose_appraisal(llm, state, reply: str, updater_notes: list[str]) -> dict:
    """Influence 分支的独立 Appraisal 调用（0912 拆分版 §7.1）。

    在 Updater 约束之后、基于 C_{t+1} 评价本轮局势——CP/FE 不再锚定在
    未生效的认知提案上；updater_notes（截断/禁止/冲突衰减等审计）让评价
    看到认知的实际变化。Elicit / Social 的对应物是合并调用（user_response.py）。
    """
    cog_changes = "\n".join(f"- {n}" for n in updater_notes) or "(none)"
    msgs = [
        {"role": "user", "content": APPRAISAL_USER.format(
            persona=state.persona,
            state_json=str(state.bdi_dict()),
            emotion=f"valence={state.emotion.valence:.2f}, arousal={state.emotion.arousal:.2f}, category={state.emotion.category}",
            history="\n".join(f"{m['role']}: {m['text']}" for m in state.history[-4:]),
            reply=reply,
            cog_changes=cog_changes,
            desire_assessment_def=DESIRE_ASSESSMENT_DEF,
            category_list=CATEGORY_LIST,
        )},
    ]
    try:
        return llm.chat_json(msgs, max_tok=600)
    except StructuredCallError:
        # 保守 fallback：GC=CP=FE=0、无逐 Desire 标注、neutral 提案（最终仍由程序校验）
        return {"appraisal": {"goal_congruence": 0.0, "coping_potential": 0.0,
                              "future_expectancy": 0.0},
                "desire_assessment": [],
                "emotion_proposal": {"category": "neutral"},
                "_fallback": "eue"}


# ---------- Emotion（§33、§34）----------

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
