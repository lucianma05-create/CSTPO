"""Validation 3：Long-Horizon Consistency（8 轮轨迹的程序化 + 盲评检查）。

A. Cognitive Drift；B. Sudden Reversal；C. Persona Consistency（盲评）；
D. Emotion Continuity；E. Repetition。
"""
from __future__ import annotations

from evaluation.state_transition import _delta_map

DRIFT_WINDOW = 2      # 无强证据（无 Accept）轮中允许的累积 |Δ|
REVERSAL_DROP = 1.0   # 2 轮内下降超过此值视为可疑反转
REVERSAL_LOW = 1.5
REVERSAL_HIGH = 2.5


def _strength_series(logs, iid: str) -> list[float]:
    out = []
    for l in logs:
        for k in ("beliefs", "desires", "intentions"):
            for i in l.bdi_after.get(k, []):
                if i["id"] == iid:
                    out.append(float(i.get("strength", 0.0)))
                    break
    return out


def drift_score(logs) -> dict:
    """A. 无强证据轮（judgment != accept 或非 influence）的累积 |Δ|。"""
    per_turn = []
    for l in logs:
        if l.mode != "influence" or l.judgment != "accept":
            d = _delta_map(l.bdi_before, l.bdi_after)
            per_turn.append(sum(abs(x) for x in d.values()))
    total = round(sum(per_turn), 2)
    return {"drift_total": total, "drift_per_turn": round(total / len(per_turn), 3) if per_turn else 0.0,
            "drift_turns": len(per_turn)}


def reversal_flags(logs) -> list[str]:
    """B. 强信念 2 轮内大幅下跌（无 Accept 证据）。"""
    flags = []
    for l in logs:
        if l.mode == "influence" and l.judgment == "accept":
            continue
        for k in ("beliefs", "desires", "intentions"):
            for i in l.bdi_before.get(k, []):
                if float(i.get("strength", 0.0)) >= REVERSAL_HIGH:
                    # 之后两轮内是否跌破低值
                    iid = i["id"]
                    later = _strength_series(logs[logs.index(l) + 1: logs.index(l) + 3], iid)
                    if later and min(later) <= REVERSAL_LOW:
                        flags.append(f"{iid}: {float(i['strength']):.1f}->{min(later):.1f} 快速反转")
    return flags


def emotion_continuity(logs) -> dict:
    """D. 效价符号翻转次数；无 GC 突变（|ΔGC|>=0.3）的翻转记为可疑。"""
    flips, suspicious = 0, 0
    prev_v, prev_gc = None, None
    for l in logs:
        v = l.emotion_after["valence"]
        gc = l.appraisal["goal_congruence"]
        if prev_v is not None:
            if (prev_v > 0.05 > v) or (prev_v < -0.05 < v):
                flips += 1
                if prev_gc is not None and abs(gc - prev_gc) < 0.3:
                    suspicious += 1
        prev_v, prev_gc = v, gc
    return {"valence_flips": flips, "suspicious_flips": suspicious}


def repetition_score(logs) -> dict:
    """E. 相邻用户话语的字符 n-gram 相似度（机械重复检测）。"""
    def jaccard(a: str, b: str, n: int = 4) -> float:
        sa = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
        sb = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
    sims = [jaccard(logs[i].user_reply, logs[i + 1].user_reply)
            for i in range(len(logs) - 1)]
    if not sims:
        return {"mean_similarity": 0.0, "max_similarity": 0.0, "high_repeat": 0}
    return {"mean_similarity": round(sum(sims) / len(sims), 3),
            "max_similarity": round(max(sims), 3),
            "high_repeat": sum(1 for s in sims if s >= 0.5)}


def persona_consistency(llm, persona: str, logs) -> dict:
    """C. 盲评：整段轨迹 vs persona 的一致性（1-5）。"""
    convo = "\n".join(f"assistant: {l.assistant_reply}\nuser: {l.user_reply}" for l in logs)
    prompt = f"""You are evaluating a simulated user over a multi-turn conversation.

Persona (who this user is):
{persona}

Conversation:
{convo}

Does the user's behavior across the WHOLE conversation stay consistent with
the persona (background, personality, situation, way of speaking)?

Return exactly one JSON object:
{{"persona_consistency": 4, "reason": "one short sentence"}}"""
    return llm.chat_json([{"role": "user", "content": prompt}], max_tok=200)
