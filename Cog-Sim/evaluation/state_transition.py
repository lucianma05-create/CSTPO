"""Validation 1：State Transition Validity（纯程序化，不调用 LLM）。

对 TurnLog 序列检查 C_t→C_{t+1} 是否符合 CogSim v1.0 定义：
A. Unrelated Change Rate；B. RJ Constraint Compliance（六组）；
C. Cognitive Direction Consistency；D. Intention Support Validity；
E. State Stability（Elicit/Social 冻结）。
"""
from __future__ import annotations

from simulator.cognitive.rj_contract import LIMITS

SUBSTANTIVE = 0.2   # 实质性变化阈值（|Δs| >= 0.2 计入）
INTENTION_EPS = 0.5
SUPPORT_STRENGTH = 2.0
STRONG = 2.5        # 强立场阈值（方向一致性判断用）


def _delta_map(before: dict, after: dict) -> dict:
    """{id: Δs}，含新增（Δ=新强度）与淘汰（Δ=-旧强度）。"""
    d = {}
    for k in ("beliefs", "desires", "intentions"):
        b = {i["id"]: float(i.get("strength", 0.0)) for i in before.get(k, [])}
        a = {i["id"]: float(i.get("strength", 0.0)) for i in after.get(k, [])}
        for iid in set(b) | set(a):
            delta = a.get(iid, 0.0) - b.get(iid, 0.0)
            if abs(delta) > 1e-9:
                d[iid] = round(delta, 3)
    return d


def legacy_unrelated_change_rate(log) -> dict:
    """A. LEGACY（Validation 1.0 口径）：只要节点不在 relevant_state_ids 中即视为
    unrelated。已由 strict_unrelated_change_rate 取代，仅保留历史对比。
    相关 = relevant_state_ids ∪ 本轮新增。"""
    d = _delta_map(log.bdi_before, log.bdi_after)
    if not d or log.mode != "influence":
        return {"rate": None, "unrelated_changed": 0, "unrelated_total": 0, "changed_ids": {}}
    related = set(log.relevant_state_ids or [])
    new_ids = {i["id"] for i in log.bdi_after.get("beliefs", []) + log.bdi_after.get("desires", [])
               + log.bdi_after.get("intentions", [])} - {
        i["id"] for i in log.bdi_before.get("beliefs", []) + log.bdi_before.get("desires", [])
        + log.bdi_before.get("intentions", [])}
    unrelated = {iid: dd for iid, dd in d.items() if iid not in related and iid not in new_ids
                 and abs(dd) >= SUBSTANTIVE}
    unrelated_total = len(set(d.keys()) - related - new_ids)
    rate = len(unrelated) / unrelated_total if unrelated_total else 0.0
    return {"rate": round(rate, 3), "unrelated_changed": len(unrelated),
            "unrelated_total": unrelated_total, "changed_ids": unrelated}


def enrich_cue_flags(log) -> None:
    """bdi_dict() 不序列化 cue（BDIItem 无该字段，cue 仅在提案期存在——
    updater.py 按 proposal 的 cue 应用 Peripheral+Accept 豁免后不落库）。
    从 proposed_bdi.new_items 按 (type, content) 回填 cue 标记，供 rj_violations
    正确识别豁免，避免假阳性（Pilot 第一批 13 起 violation 即此类）。"""
    new_items = (log.proposed_bdi or {}).get("new_items", [])
    for ni in new_items:
        if ni.get("type") != "belief" or not ni.get("cue"):
            continue
        for b in log.bdi_after.get("beliefs", []):
            if b["content"] == ni.get("content") and "cue" not in b:
                b["cue"] = True


def rj_violations(log) -> list[str]:
    """B. RJ 约束违例（代码应已硬保证，这里验证 implementation correctness）。"""
    if log.mode != "influence" or not log.route:
        return []
    enrich_cue_flags(log)
    viol = []
    limit = LIMITS[(log.route, log.judgment)]
    d = _delta_map(log.bdi_before, log.bdi_after)
    btype = {i["id"]: "belief" for i in log.bdi_before["beliefs"]}
    btype.update({i["id"]: "desire" for i in log.bdi_before["desires"]})
    btype.update({i["id"]: "intention" for i in log.bdi_before["intentions"]})
    btype.update({i["id"]: i["type"] for k in ("beliefs", "desires", "intentions")
                  for i in log.bdi_after.get(k, []) if i["id"] not in btype})
    core_d = {i["id"] for i in log.bdi_before["desires"] if i.get("core", True)}
    cue_beliefs = {i["id"] for i in log.bdi_after["beliefs"] if i.get("cue")}
    for iid, dd in d.items():
        t = btype.get(iid)
        # 限幅检查（cue 例外：Peripheral+Accept 的 cue 信念豁免）
        exempt = (log.route == "peripheral" and log.judgment == "accept"
                  and t == "belief" and iid in cue_beliefs)
        if not exempt and abs(dd) > limit.get(t, 0.0) + 1e-9:
            viol.append(f"限幅: {iid} Δ={dd} > {limit.get(t)} ({log.route}+{log.judgment})")
    # Reject 方向：相关 Belief 禁止正向更新
    if log.judgment == "reject":
        for iid, dd in d.items():
            if iid in (log.relevant_state_ids or []) and btype.get(iid) == "belief" and dd > 1e-9:
                viol.append(f"Reject 方向: {iid} 正向更新 Δ={dd}")
    # Peripheral 核心 Desire 限制（J != Accept 时禁止修改）
    if log.route == "peripheral" and log.judgment != "accept":
        for iid, dd in d.items():
            if iid in core_d and btype.get(iid) == "desire":
                viol.append(f"Peripheral 核心 Desire: {iid} 被修改 Δ={dd}")
    return viol


def unsupported_intention(log) -> list[str]:
    """D. Intention 支撑：|ΔI|>0.5 需同轮同方向 B/D 更新(|Δ|>=0.2)或现存强 B/D(>=2.0)。"""
    if log.mode != "influence":
        return []
    d = _delta_map(log.bdi_before, log.bdi_after)
    btype = {i["id"]: i["type"] for k in ("beliefs", "desires", "intentions")
             for i in log.bdi_before.get(k, [])}
    fails = []
    for iid, dd in d.items():
        if btype.get(iid) != "intention" or abs(dd) <= INTENTION_EPS:
            continue
        same_dir_bd = any(abs(dd2) >= SUBSTANTIVE and (dd2 * dd > 0)
                          for iid2, dd2 in d.items()
                          if btype.get(iid2) in ("belief", "desire") and iid2 != iid)
        strong_bd = any(float(i.get("strength", 0.0)) >= SUPPORT_STRENGTH
                        for k in ("beliefs", "desires")
                        for i in log.bdi_after.get(k, []) if i["id"] != iid)
        if not (same_dir_bd or strong_bd):
            fails.append(f"I 无支撑: {iid} Δ={dd}")
    return fails


def direction_consistency(log) -> list[str]:
    """C. 认知方向一致性（启发式，按 p_t/J_t/相关节点判断）。"""
    if log.mode != "influence" or not log.judgment:
        return []
    d = _delta_map(log.bdi_before, log.bdi_after)
    btype = {i["id"]: i["type"] for k in ("beliefs", "desires", "intentions")
             for i in log.bdi_before.get(k, [])}
    related = set(log.relevant_state_ids or [])
    issues = []
    if log.judgment == "reject":
        for iid, dd in d.items():
            if iid in related and btype.get(iid) == "belief" and dd > SUBSTANTIVE:
                issues.append(f"Reject 轮相关信念正向 {iid} Δ={dd}")
    if log.judgment == "noncommit":
        # 不应出现强立场反转：相关节点从 >=2.5 掉到 <=1.5
        for iid, dd in d.items():
            if iid not in related:
                continue
            before = {i["id"]: float(i.get("strength", 0.0))
                      for k in ("beliefs", "desires", "intentions")
                      for i in log.bdi_before.get(k, [])}
            after = {i["id"]: float(i.get("strength", 0.0))
                     for k in ("beliefs", "desires", "intentions")
                     for i in log.bdi_after.get(k, [])}
            if before.get(iid, 0.0) >= STRONG and after.get(iid, 0.0) <= STRONG - 1.0:
                issues.append(f"Noncommit 轮立场反转 {iid}: {before[iid]}->{after[iid]}")
    if log.judgment == "accept":
        # 至少一个相关节点 substantive 变化（松动或增强）
        if not any(abs(dd) >= SUBSTANTIVE for iid, dd in d.items() if iid in related):
            issues.append("Accept 轮无任何相关节点 substantive 变化")
    return issues


def state_stability(log) -> list[str]:
    """E. Elicit/Social 必须 C_{t+1}=C_t 严格成立。"""
    if log.mode in ("elicit", "social"):
        d = _delta_map(log.bdi_before, log.bdi_after)
        return [f"{log.mode} 轮认知变化 {d}"] if d else []
    return []


RELEVANCE_PROMPT = """Classify the relation between the assistant's target proposition and
each changed node in the user's cognitive state.

Target proposition: {target}

User state nodes (id / type / content):
{state_text}

Changed nodes this turn (id / type / content / strength change):
{changed_text}

For EACH changed node output one of:
- "direct": the node is the proposition's direct semantic target.
- "consequence": not a direct target, but a reasonable first-order consequence
  of the proposition's cognitive effect (e.g. a price belief change reasonably
  moving a desire to close the deal today).
- "unrelated": neither.

Return exactly one JSON object: {{"labels": {{"B1": "direct", "D2": "consequence"}}}}"""


def classify_relevance(llm, log) -> dict:
    """对已变化节点做 direct/consequence/unrelated 分层（Validation 1.1）。
    直接相关 = relevant_state_ids 命中；其余节点一次批量 LLM 分类。"""
    d = _delta_map(log.bdi_before, log.bdi_after)
    nodes = {}
    for k in ("beliefs", "desires", "intentions"):
        for i in log.bdi_before.get(k, []) + log.bdi_after.get(k, []):
            nodes[i["id"]] = i
    labels = {}
    changed = {iid: dd for iid, dd in d.items() if abs(dd) >= SUBSTANTIVE}
    for iid in changed:
        labels[iid] = "direct" if iid in (log.relevant_state_ids or []) else None
    pending = {iid: dd for iid, dd in changed.items() if labels[iid] is None}
    if pending and llm is not None:
        state_text = "\n".join(
            f"- {iid} / {nodes[iid]['type']} / {nodes[iid]['content']}"
            for iid in nodes if iid in pending)
        changed_text = "\n".join(
            f"- {iid} / {nodes[iid]['type']} / {nodes[iid]['content']} / Δ={dd}"
            for iid, dd in pending.items())
        try:
            out = llm.chat_json([{"role": "user", "content": RELEVANCE_PROMPT.format(
                target=log.target or "(none)", state_text=state_text,
                changed_text=changed_text)}], max_tok=200)
            for iid, lab in (out.get("labels") or {}).items():
                if iid in pending and lab in ("direct", "consequence", "unrelated"):
                    labels[iid] = lab
        except Exception as e:
            print(f"  [relevance] 分类失败（保守视为 consequence）: {e}")
    for iid in pending:
        if labels[iid] is None:
            labels[iid] = "consequence"   # 保守：不明关系不记入 strict unrelated
    return labels


def strict_unrelated_rates(llm, logs) -> dict:
    """新指标（Validation 1.1 正式口径）：
    direct_relevant_update_rate / consequential_relevant_update_rate /
    strict_unrelated_change_rate（只有 relation=unrelated 的 substantive 变化计入）。"""
    direct = cons = strict = 0
    total = 0
    for log in logs:
        if log.mode != "influence":
            continue
        d = _delta_map(log.bdi_before, log.bdi_after)
        changed = {iid: dd for iid, dd in d.items() if abs(dd) >= SUBSTANTIVE}
        if not changed:
            continue
        labels = classify_relevance(llm, log)
        for iid in changed:
            total += 1
            if labels.get(iid) == "direct":
                direct += 1
            elif labels.get(iid) == "consequence":
                cons += 1
            else:
                strict += 1
    if not total:
        return {"direct_relevant_update_rate": None,
                "consequential_relevant_update_rate": None,
                "strict_unrelated_change_rate": None, "total": 0}
    return {"direct_relevant_update_rate": round(direct / total, 3),
            "consequential_relevant_update_rate": round(cons / total, 3),
            "strict_unrelated_change_rate": round(strict / total, 3),
            "total": total}


def avg_update_magnitude(logs) -> dict:
    """平均 |ΔC| 与 B/D/I 分别平均 update magnitude。"""
    totals = {"beliefs": [], "desires": [], "intentions": [], "all": []}
    for log in logs:
        d = _delta_map(log.bdi_before, log.bdi_after)
        btype = {i["id"]: i["type"] for k in ("beliefs", "desires", "intentions")
                 for i in log.bdi_before.get(k, [])}
        btype.update({i["id"]: i["type"] for k in ("beliefs", "desires", "intentions")
                      for i in log.bdi_after.get(k, []) if i["id"] not in btype})
        for iid, dd in d.items():
            totals["all"].append(abs(dd))
            totals[btype.get(iid, "all") + "s"].append(abs(dd))
    out = {}
    for k in ("beliefs", "desires", "intentions"):
        v = totals[k]
        out[k] = round(sum(v) / len(v), 3) if v else None
    v = totals["all"]
    out["all"] = round(sum(v) / len(v), 3) if v else None
    return out


def summarize_logs(logs, task: str) -> dict:
    """对一组 TurnLog 输出 §2.2 要求的所有统计。"""
    influence = [l for l in logs if l.mode == "influence"]
    total_turns = len(logs)
    rj_dist = {}
    for l in influence:
        rj_dist[f"{l.route}+{l.judgment}"] = rj_dist.get(f"{l.route}+{l.judgment}", 0) + 1
    # A（legacy 口径，标注 Validation 1.0 metric）
    ucr = [legacy_unrelated_change_rate(l) for l in influence]
    rates = [x["rate"] for x in ucr if x["rate"] is not None]
    # B
    rj_v = sum(len(rj_violations(l)) for l in influence)
    # C
    dc = sum(len(direction_consistency(l)) for l in influence)
    # D
    ui = sum(len(unsupported_intention(l)) for l in influence)
    # E
    stab = sum(len(state_stability(l)) for l in logs)
    return {
        "task": task,
        "total_turns": total_turns,
        "influence_turns": len(influence),
        "rj_distribution": rj_dist,
        "legacy_unrelated_change_rate": round(sum(rates) / len(rates), 3) if rates else None,
        "rj_violations": rj_v,
        "direction_issues": dc,
        "unsupported_intentions": ui,
        "stability_violations": stab,
        "avg_update_magnitude": avg_update_magnitude(logs),
    }
