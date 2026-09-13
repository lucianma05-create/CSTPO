"""Validation 10 支撑：Efficiency——按路径/引擎统计 token、调用数、延迟。

runner 在每次运行结束时记录 usage（llm.usage_report 数据）与耗时。
"""
from __future__ import annotations


def record_usage(sim, duration_s: float, path: str, task: str) -> dict:
    return {"task": task, "path": path, "calls": sim.llm.calls,
            "prompt_tokens": sim.llm.prompt_tokens,
            "completion_tokens": sim.llm.completion_tokens,
            "total_tokens": sim.llm.prompt_tokens + sim.llm.completion_tokens,
            "duration_s": round(duration_s, 1)}


def summarize(records: list[dict]) -> dict:
    """按 task+path 汇总 mean/std。"""
    groups = {}
    for r in records:
        g = groups.setdefault((r["task"], r["path"]), [])
        g.append(r)
    out = {}
    for (task, path), items in sorted(groups.items()):
        def stat(k):
            v = [i[k] for i in items]
            m = sum(v) / len(v)
            sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
            return round(m, 1), round(sd, 1)
        out[f"{task}/{path}"] = {
            "n": len(items), "total_tokens": stat("total_tokens"),
            "prompt_tokens": stat("prompt_tokens"),
            "completion_tokens": stat("completion_tokens"),
            "calls": stat("calls"), "duration_s": stat("duration_s"),
        }
    return out


def per_engine_estimate() -> dict:
    """模板体积估算（chars/4，运行时动态输入不计）。"""
    def est(s):
        return round(len(s) / 4)
    from simulator.routing.mode_classifier import MODE_PROMPT
    from simulator.routing.route_features import FEATURES_PROMPT, INTERACTION_PRESSURE_DEF
    from simulator.routing.discrepancy import DISCREPANCY_PROMPT
    from simulator.cognitive.cognitive_engine import ENGINE_SYSTEM, ENGINE_USER
    from simulator.cognitive.rj_contract import CONTRACTS
    from simulator.affect.emotion_engine import APPRAISAL_USER, CATEGORY_LIST, DESIRE_ASSESSMENT_DEF
    from simulator.generation.user_response import NLG_SYSTEM, NLG_USER, RESPOND_USER
    from simulator.generation.conversation_end import DONE_PROMPT
    return {
        "ATC": est(MODE_PROMPT),
        "TRIE": est(FEATURES_PROMPT) + est(INTERACTION_PRESSURE_DEF),
        "JEE": est(DISCREPANCY_PROMPT),
        "Engine": est(ENGINE_SYSTEM) + est(ENGINE_USER),
        "contract": max(est(v) for v in CONTRACTS.values()),
        "EUE": est(APPRAISAL_USER) + est(DESIRE_ASSESSMENT_DEF) + est(CATEGORY_LIST),
        "SRR/ERR": est(RESPOND_USER) + est(DESIRE_ASSESSMENT_DEF)
                    + est(INTERACTION_PRESSURE_DEF) + est(CATEGORY_LIST),
        "NLG": est(NLG_SYSTEM) + est(NLG_USER),
        "CED": est(DONE_PROMPT),
    }
