"""Deterministic Updater（文档 §14）：LLM 输出不是最终状态。

程序执行 BDI_{t+1} = Apply(BDI_t, LLMUpdates, RJConstraints)，检查：
强度范围、更新幅度、Judgment 方向、Peripheral 限制、Intention 支持、数量限制。
所有被约束的动作都会写进 notes，保证过程可审计。
"""
from __future__ import annotations

from simulator.cognitive.rj_contract import limits_for
from simulator.state.schema import (
    BDIItem,
    DEACTIVATE_THRESHOLD,
    MAX_ITEMS,
    POLARITIES,
    STRENGTH_MAX,
    STRENGTH_MIN,
    UserState,
)

INTENTION_EPSILON = 0.5   # 文档 §14 的 epsilon（文档未给具体值，取 0.5）
SUPPORT_STRENGTH = 2.0    # “相关强 Belief/Desire”的强度门槛


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(float(v), lo), hi)


def constrained_apply(
    state: UserState,
    proposal: dict,
    route: str,
    judgment: str,
    related_belief_ids: list[str],
) -> list[str]:
    """就地应用 proposal 到 state，返回约束审计 notes。"""
    notes: list[str] = []
    limits = limits_for(route, judgment)
    applied_deltas: dict[str, float] = {}

    # ---- 1. 更新已有节点 ----
    for u in proposal.get("bdi_updates", []):
        item_id = str(u.get("id", ""))
        item = state.find(item_id)
        if item is None:
            notes.append(f"跳过不存在节点 {item_id}")
            continue
        new_s = _clamp(u.get("new_strength", item.strength), STRENGTH_MIN, STRENGTH_MAX)
        if not item.active:
            # 已退役节点只允许被重新激活：强度需回升到阈值以上
            if new_s >= DEACTIVATE_THRESHOLD + 0.2:
                item.active = True
                notes.append(f"{item_id} 强度回升至 {new_s:.2f}，重新激活")
            else:
                notes.append(f"跳过未激活节点 {item_id}（新强度 {new_s:.2f} 不足以重新激活）")
                continue
        delta = new_s - item.strength
        cap = limits[item.type]

        # 幅度限制（文档 §14.2）
        if abs(delta) > cap:
            new_s = item.strength + cap * (1 if delta > 0 else -1)
            notes.append(f"{item_id} 幅度 {delta:+.2f} 超限 {cap}，截断为 {new_s - item.strength:+.2f}")

        # Judgment=Reject 方向限制（文档 §14.3）：被拒命题相关 core Belief 不允许正向更新
        if judgment == "reject" and item.type == "belief" and item.id in related_belief_ids and new_s > item.strength:
            notes.append(f"{item_id} 与被拒命题相关，禁止正向更新（{new_s - item.strength:+.2f} -> 0）")
            new_s = item.strength

        # Peripheral 核心限制（文档 §14.4）：核心 Desire 不允许大幅改变
        if route == "peripheral" and item.type == "desire" and judgment != "accept" and new_s != item.strength:
            notes.append(f"Peripheral+{judgment} 不允许修改核心 Desire {item_id}")
            new_s = item.strength

        applied_deltas[item.id] = new_s - item.strength
        item.strength = new_s

    # ---- 2. 新增节点 ----
    new_intention_ids: set[str] = set()
    for n in proposal.get("new_items", []):
        type_ = str(n.get("type", "")).strip().lower()
        if type_ not in MAX_ITEMS:
            notes.append(f"未知节点类型 {type_!r}，拒绝新增")
            continue
        content = str(n.get("content", "")).strip()
        if not content:
            notes.append("空内容节点，拒绝新增")
            continue
        strength = _clamp(n.get("strength", 1.0), STRENGTH_MIN, STRENGTH_MAX)
        is_cue = bool(n.get("cue", False))
        is_core = bool(n.get("core", True))
        polarity = str(n.get("polarity", "approach")).strip().lower()
        if type_ == "belief":
            polarity = "approach"   # Belief 不使用极性字段
        elif polarity not in POLARITIES:
            notes.append(f"非法 polarity {polarity!r}，默认 approach")
            polarity = "approach"

        if route == "peripheral":
            # Peripheral 路线下：核心节点受核心限制约束，cue 节点不受限（文档 §11.4）
            cap = limits[type_] if (is_core and not is_cue) else STRENGTH_MAX
            if cap == 0.0:
                notes.append(f"Peripheral+{judgment} 禁止新增 {type_}（限制为 0），拒绝")
                continue
            if strength > cap:
                notes.append(f"新增 {type_} 强度 {strength} 超限 {cap}，截断为 {cap}")
                strength = cap
        if judgment == "reject" and type_ == "intention" and strength > limits["intention"]:
            notes.append(f"Reject 下新增 Intention 强度 {strength} 超限 {limits['intention']}，截断")
            strength = limits["intention"]

        new_item = BDIItem(
            id=f"{type_[0].upper()}{_next_id(state, type_)}",
            type=type_,
            content=content,
            strength=strength,
            core=is_core,
            active=True,
            polarity=polarity,
        )
        # 弱非核心节点直接以退役状态入库，不占用配额（文档 §3.1 保持状态紧凑）
        if not is_core and strength < DEACTIVATE_THRESHOLD:
            new_item.active = False
            notes.append(f"新增 {new_item.id} [{type_}] strength={strength:.2f} 过弱，直接退役")
        lst = state.items(type_)
        if sum(1 for i in lst if i.active) >= MAX_ITEMS[type_]:
            evicted = _evict(state, type_, new_item)
            if evicted == "incoming":
                notes.append(f"{type_} 数量达上限且新节点更弱，拒绝新增")
            else:
                notes.append(f"{type_} 数量达上限 {MAX_ITEMS[type_]}，淘汰 {evicted}，新增 {new_item.id}")
        else:
            lst.append(new_item)
            if new_item.active:
                notes.append(f"新增 {new_item.id} [{type_}] strength={strength:.2f}"
                             + (f" polarity={polarity}" if type_ != "belief" else ""))
        if type_ == "intention" and new_item in state.intentions:
            new_intention_ids.add(new_item.id)

        # Belief 冲突衰减（文档 §3.1 未定义，扩展规则）：
        # 新增 Belief 与已有 Belief 冲突时，已有 Belief 按新节点强度衰减，
        # 避免状态同时强持有 P 和 ¬P。
        if type_ == "belief" and new_item in state.beliefs:
            for cid in [str(c) for c in n.get("conflicts_with", []) or []]:
                victim = state.find(cid)
                if victim is not None and victim.type == "belief" and victim.active:
                    victim.strength = _clamp(victim.strength - strength, STRENGTH_MIN, STRENGTH_MAX)
                    notes.append(f"新 Belief {new_item.id} 与 {cid} 冲突，{cid} 衰减至 {victim.strength:.2f}")

    # ---- 3. Intention 支持检查（文档 §14.5）----
    # |ΔI| > ε 时，必须有：当轮同方向 B/D 更新，或现存强 Belief/Desire 可解释。
    for item in state.intentions:
        # 新增 Intention 节点同样要检查：strength > ε 需有支持，否则截断。
        if item.id in new_intention_ids:
            delta = item.strength  # 从 0 新增，视为全量变化
        else:
            delta = applied_deltas.get(item.id, 0.0)
        if abs(delta) <= INTENTION_EPSILON:
            continue
        bd_same_direction = any(
            d * delta > 0 and abs(d) >= 0.2
            for iid, d in applied_deltas.items()
            if state.find(iid) is not None and state.find(iid).type in ("belief", "desire")
        )
        strong_bd = any(
            i.active and i.strength >= SUPPORT_STRENGTH
            for i in state.beliefs + state.desires
        )
        if not (bd_same_direction or strong_bd):
            capped = item.strength - delta + INTENTION_EPSILON * (1 if delta > 0 else -1)
            notes.append(f"{item.id} 大额 Intention 变化 {delta:+.2f} 无 Belief/Desire 支持，截断到 {capped:.2f}")
            item.strength = _clamp(capped, STRENGTH_MIN, STRENGTH_MAX)

    # ---- 4. 退役扫描（文档 §3.1）：非核心弱节点 deactivate ----
    for lst in (state.beliefs, state.desires, state.intentions):
        for it in lst:
            if it.active and not it.core and it.strength < DEACTIVATE_THRESHOLD:
                it.active = False
                notes.append(f"{it.id} [{it.type}] 强度 {it.strength:.2f} < {DEACTIVATE_THRESHOLD}，退役（inactive）")

    return notes


def _next_id(state: UserState, type_: str) -> int:
    nums = []
    for it in state.items(type_):
        digits = "".join(ch for ch in it.id if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    return max(nums, default=0) + 1


def _evict(state: UserState, type_: str, incoming: BDIItem) -> str:
    """数量超限时淘汰一个节点（文档 §14.6），返回被淘汰的 id。

    淘汰优先级：退役节点 > 非核心弱节点。
    若 incoming 比所有现存活跃节点都弱且非核心，则拒绝新增（返回 "incoming"）。
    """
    lst = state.items(type_)
    cands = sorted(lst, key=lambda i: (i.active, i.core, -i.strength))
    victim = cands[0]
    if (not victim.active) or ((not victim.core) and incoming.strength >= victim.strength):
        lst.remove(victim)
        lst.append(incoming)
        return victim.id
    return "incoming"
