"""Validation 7：Component-Level Accuracy（人工标注集）。

ATC：三模式标注集（含混合 case）→ Accuracy + Confusion Matrix；
TRIE：特征等级标注集（重点 Argument≠Cue、Pressure≠说服力）→ 等级一致率；
JEE：stance 距离标注集（含"语义相似但立场远"陷阱）→ Accuracy；
EUE：CategoryOverrideRate 由运行日志程序统计。
"""
from __future__ import annotations

from simulator.routing.mode_classifier import classify_mode
from simulator.routing.route_features import extract_route_features
from simulator.routing.discrepancy import estimate_discrepancy
from simulator.run_sim import SCENARIOS, build_state

# ---------- ATC 标注集 ----------
ATC_LABELED = [
    # (assistant reply, 标注 mode)
    ("85是最低价，我今天已经拒绝几个80的报价。", "influence"),
    ("其实小额捐赠的边际价值很高，5元就能让一个孩子吃上一顿热饭。", "influence"),
    ("先给导师发一条消息只需要两分钟，回复一句进展顺利就够。", "influence"),
    ("你心里的预算到底是多少？", "elicit"),
    ("你之前有过捐款的经历吗？", "elicit"),
    ("你今天最担心的是什么？", "elicit"),
    ("听起来这段时间你压力很大，真的辛苦了。", "social"),
    ("今天天气不错，骑车上学挺舒服的。", "social"),
    ("最近工作忙不忙？", "social"),
    ("大家都说这个平台靠谱，你可以了解一下再说。", "influence"),
    ("如果能看到具体项目的账目，你会不会考虑捐一点？", "influence"),   # 提问+主张 → influence
    ("你不需要一次解决所有问题，可以先明天只给导师发一条消息。", "influence"),
    ("那今天先聊到这，你考虑好了随时联系我。", "social"),
    ("你的预算是多少，另外你更看重车况还是价格？", "elicit"),
    ("没关系，你已经做得很好了，慢慢来。", "social"),
]

# ---------- TRIE 标注集（Validation 1.1 扩充：每任务 12 条，覆盖 §4.2 十二类 case）----------
# (task, reply, {relevance, argument, cue, pressure})  标注为期望等级
TRIE_LABELED = [
    # ---- bargain（B1 卖家可能接受80 / D1 低价 / D2 今天成交 / I1 坚持80）----
    ("bargain", "这车市场价95以上，85已经很便宜了，同款成交记录都在90左右。",
     {"relevance": "high", "argument": "high", "cue": "low", "pressure": "medium"}),      # 1 高相关+强论点
    ("bargain", "85真的不贵，你相信我。",
     {"relevance": "high", "argument": "low", "cue": "medium", "pressure": "medium"}),    # 2 高相关+弱论点
    ("bargain", "这款车的刹车用的是禧玛诺的配件，非常耐用。",
     {"relevance": "low", "argument": "high", "cue": "low", "pressure": "low"}),          # 3 低相关+强论点
    ("bargain", "我认识很多你们学校的同学，他们都跟我买过车。",
     {"relevance": "medium", "argument": "low", "cue": "high", "pressure": "low"}),       # 4 低/中相关+强线索
    ("bargain", "别犹豫了，大家都说好，我卖车口碑在这。",
     {"relevance": "medium", "argument": "low", "cue": "high", "pressure": "medium"}),    # 5 高线索+弱论点
    ("bargain", "同款新车现在要卖1200。",
     {"relevance": "medium", "argument": "medium", "cue": "low", "pressure": "low"}),     # 6 低压力信息
    ("bargain", "你要是明天能定，我可以考虑送个车锁。",
     {"relevance": "high", "argument": "medium", "cue": "low", "pressure": "medium"}),    # 7 中压力建议
    ("bargain", "今天下午五点前不定，我就卖给别人了。",
     {"relevance": "high", "argument": "low", "cue": "medium", "pressure": "high"}),      # 8 高压最后通牒
    ("bargain", "这车没有任何暗病，我可以给你看保养记录。",
     {"relevance": "high", "argument": "high", "cue": "low", "pressure": "low"}),         # 9 信念靶向
    ("bargain", "80块的车再纠结，明天开学就没车骑了。",
     {"relevance": "high", "argument": "low", "cue": "low", "pressure": "medium"}),       # 10 愿望靶向
    ("bargain", "你现在就转80，车今晚就是你的。",
     {"relevance": "high", "argument": "low", "cue": "low", "pressure": "high"}),         # 11 意图靶向
    ("bargain", "你们学校很多人都是骑二手车的。",
     {"relevance": "low", "argument": "low", "cue": "medium", "pressure": "low"}),        # 12 同话题低相关
    # ---- donation（B1 小额捐赠有限 / D1 帮助别人 / D2 控制支出 / I1 不准备捐）----
    ("donation", "5元就能让孩子吃上一顿热饭，采购清单和账目都公开可查。",
     {"relevance": "high", "argument": "high", "cue": "low", "pressure": "medium"}),      # 1
    ("donation", "捐款肯定有用，你要相信。",
     {"relevance": "high", "argument": "low", "cue": "medium", "pressure": "medium"}),    # 2
    ("donation", "这家公益平台成立于2010年，总部在上海。",
     {"relevance": "low", "argument": "high", "cue": "low", "pressure": "low"}),          # 3
    ("donation", "听说这家平台创始人上过新闻，人很厉害。",
     {"relevance": "low", "argument": "low", "cue": "high", "pressure": "low"}),          # 4
    ("donation", "这个公益平台明星都在推，大家都说靠谱。",
     {"relevance": "medium", "argument": "low", "cue": "high", "pressure": "low"}),       # 5
    ("donation", "每月定额捐的话，平台会自动扣款。",
     {"relevance": "medium", "argument": "medium", "cue": "low", "pressure": "low"}),     # 6
    ("donation", "如果你愿意，可以先捐5元试试。",
     {"relevance": "high", "argument": "low", "cue": "low", "pressure": "medium"}),       # 7
    ("donation", "今天募捐就截止了，你现在决定捐不捐？",
     {"relevance": "high", "argument": "low", "cue": "medium", "pressure": "high"}),      # 8
    ("donation", "小额捐赠积少成多，很多孩子的午餐就是靠它。",
     {"relevance": "high", "argument": "high", "cue": "low", "pressure": "medium"}),      # 9
    ("donation", "你其实很想帮孩子，只是担心钱的问题，对吧？",
     {"relevance": "high", "argument": "low", "cue": "low", "pressure": "medium"}),       # 10
    ("donation", "现在就捐5元，一分钟就完成。",
     {"relevance": "high", "argument": "low", "cue": "low", "pressure": "high"}),         # 11
    ("donation", "我大学同学现在就在公益行业工作。",
     {"relevance": "low", "argument": "low", "cue": "low", "pressure": "low"}),           # 12
    # ---- support（B1 完全解决不了 / D1 恢复控制感 / I1 不想行动）----
    ("support", "发一条消息只要两分钟，你上次和导师沟通还算顺利，先迈出最小的一步最有效。",
     {"relevance": "high", "argument": "high", "cue": "low", "pressure": "medium"}),      # 1
    ("support", "你肯定可以的，要有信心。",
     {"relevance": "medium", "argument": "low", "cue": "medium", "pressure": "low"}),     # 2
    ("support", "学校的心理咨询中心每周三开放，预约流程是先填表。",
     {"relevance": "low", "argument": "high", "cue": "low", "pressure": "low"}),          # 3
    ("support", "我导师人很好，大家都说找他聊天有用。",
     {"relevance": "medium", "argument": "low", "cue": "high", "pressure": "low"}),       # 4
    ("support", "大家都说先动起来就好了，很多学长都是这么过来的。",
     {"relevance": "medium", "argument": "low", "cue": "high", "pressure": "medium"}),    # 5
    ("support", "研究生阶段论文卡壳很常见，很多人都有这个阶段。",
     {"relevance": "medium", "argument": "medium", "cue": "low", "pressure": "low"}),     # 6
    ("support", "要不要试试今晚先把草稿打开十分钟？",
     {"relevance": "high", "argument": "low", "cue": "low", "pressure": "medium"}),       # 7
    ("support", "导师说了这周五必须看到你的初稿，你现在就得开始。",
     {"relevance": "high", "argument": "low", "cue": "medium", "pressure": "high"}),      # 8
    ("support", "你并不是解决不了，只是卡在了开头。",
     {"relevance": "high", "argument": "medium", "cue": "low", "pressure": "medium"}),    # 9
    ("support", "先做一点点，控制感就会回来。",
     {"relevance": "high", "argument": "low", "cue": "low", "pressure": "medium"}),       # 10
    ("support", "现在就打开文档写第一句话。",
     {"relevance": "high", "argument": "low", "cue": "low", "pressure": "high"}),         # 11
    ("support", "你们实验室其他同学的论文进度怎么样？",
     {"relevance": "low", "argument": "low", "cue": "low", "pressure": "medium"}),        # 12
]

TRIE_STATES = {"bargain": "bargain", "donation": "donation", "support": "support"}

TARGET_QUALITY_PROMPT = """Judge the quality of a target proposition extracted from an
assistant reply. It should be: (a) faithful — the reply actually advances it;
(b) a single concrete proposition/action direction, not a summary or a
question; (c) free of hallucinated content.

Assistant reply:
{reply}

Extracted target proposition:
{target}

Return exactly one JSON object: {{"quality": 4, "reason": "one short sentence"}}
(1 = wrong/unfaithful; 3 = acceptable but vague or compound; 5 = faithful,
single, precise)"""

# ---------- JEE 标注集 ----------
# (task, target proposition, 标注 stance_distance)
JEE_LABELED = [
    ("bargain", "接受85元的价格并今天成交", "medium"),
    ("bargain", "接受80元的价格并今天成交", "low"),          # 与 B1/I1 一致
    ("bargain", "卖家今天不会接受80元", "medium"),
    ("donation", "小额捐赠作用很有限", "low"),                 # 与 B1 一致
    ("donation", "5元捐款效果实打实", "high"),                 # 与 B1 冲突
    ("donation", "小额捐赠值得，因为钱花得值", "high"),
    ("support", "先给导师发一条消息", "medium"),               # 与 I1(avoid) 冲突但非强
    ("support", "不采取任何行动，先缓一缓", "low"),            # 与 I1 一致
    ("support", "这件事你自己完全能解决", "high"),            # 与 B1(3.5) 强冲突
]

TASK_GOALS = {"bargain": "make the user accept the 85 price and close the deal",
              "donation": "make the user donate",
              "support": "make the user feel better and take action"}


def run_atc(llm) -> dict:
    correct = 0
    conf = {"influence": {"influence": 0, "elicit": 0, "social": 0},
            "elicit": {"influence": 0, "elicit": 0, "social": 0},
            "social": {"influence": 0, "elicit": 0, "social": 0}}
    for reply, label in ATC_LABELED:
        mode, _ = classify_mode(llm, [], reply, debug=False)
        conf[label][mode] += 1
        if mode == label:
            correct += 1
    return {"accuracy": round(correct / len(ATC_LABELED), 3),
            "n": len(ATC_LABELED), "confusion": conf}


LEVEL_VALUE = {"low": 0.2, "medium": 0.5, "high": 0.8}


def _route_from(rel: float, arg: float, cue: float, eta_R: float) -> str:
    """正式 Route 公式（§4.2，deterministic）：Central=η·Rel·Arg，
    Peripheral=(1-η)·Cue；弱信号回退 p_C=η。"""
    central = eta_R * rel * arg
    peripheral = (1 - eta_R) * cue
    if central + peripheral < 0.05:
        p_c = eta_R
    else:
        p_c = central / (central + peripheral + 1e-8)
    return "central" if p_c >= 0.5 else "peripheral"


def run_trie(llm) -> dict:
    """Validation 1.1：36 条标注集，分特征 accuracy + low/med/high 混淆矩阵 +
    target proposition 质量（LLM 1-5 盲评）+ Route Decision Stability
    （gold 特征 vs 预测特征经同一 Route 公式比较，见 RouteAgreement）。"""
    feat_key = {"relevance": "relevance", "argument": "argument_strength",
                "cue": "cue_strength", "pressure": "interaction_pressure"}
    agree = {k: 0 for k in feat_key}
    conf = {k: {} for k in feat_key}
    qualities = []
    route_rows = []
    n = len(TRIE_LABELED)
    for task, reply, label in TRIE_LABELED:
        state = build_state(SCENARIOS[task])
        feats = extract_route_features(llm, state, reply, debug=False)
        pred = {}
        for k in feat_key:
            level = {0.2: "low", 0.5: "medium", 0.8: "high"}.get(feats[feat_key[k]], "?")
            pred[k] = level
            if level == label[k]:
                agree[k] += 1
            conf[k][(label[k], level)] = conf[k].get((label[k], level), 0) + 1
        target = feats.get("target_proposition")
        if target:
            q = llm.chat_json([{"role": "user", "content": TARGET_QUALITY_PROMPT.format(
                reply=reply, target=target)}], max_tok=150)
            qualities.append(q.get("quality"))
        # Route Decision Stability
        eta = state.profile.eta_R
        r_gold = _route_from(LEVEL_VALUE[label["relevance"]],
                             LEVEL_VALUE[label["argument"]],
                             LEVEL_VALUE[label["cue"]], eta)
        r_pred = _route_from(feats["relevance"], feats["argument_strength"],
                             feats["cue_strength"], eta)
        feat_err = any(pred[k] != label[k] for k in ("relevance", "argument", "cue"))
        route_rows.append({"task": task, "route_gold": r_gold, "route_pred": r_pred,
                           "feature_error": feat_err})
    # 相邻/严重分歧统计
    adj = sev = 0
    for k in feat_key:
        for (a, p), c in conf[k].items():
            order = {"low": 0, "medium": 1, "high": 2}
            d = abs(order.get(a, 0) - order.get(p, 0))
            if d == 1:
                adj += c
            elif d == 2:
                sev += c
    # Route Agreement 汇总（overall + 分任务）
    def route_stats(rows):
        agree_n = sum(1 for r in rows if r["route_gold"] == r["route_pred"])
        err_same = sum(1 for r in rows if r["feature_error"]
                       and r["route_gold"] == r["route_pred"])
        err_flip = sum(1 for r in rows if r["feature_error"]
                       and r["route_gold"] != r["route_pred"])
        return {"route_agreement": round(agree_n / len(rows), 3),
                "route_flip_rate": round(1 - agree_n / len(rows), 3),
                "feature_error_without_flip": err_same,
                "feature_error_causing_flip": err_flip,
                "n": len(rows)}
    route_stats_out = {"overall": route_stats(route_rows)}
    for task in ("bargain", "donation", "support"):
        route_stats_out[task] = route_stats([r for r in route_rows if r["task"] == task])
    return {
        "accuracy": {k: round(v / n, 3) for k, v in agree.items()},
        "confusion": {k: {f"{a}->{p}": c for (a, p), c in sorted(v.items())}
                      for k, v in conf.items()},
        "adjacent_disagreement": adj, "severe_disagreement": sev,
        "target_quality_mean": round(sum(qualities) / len(qualities), 2) if qualities else None,
        "route_decision_stability": route_stats_out,
        "n": n,
    }


def run_jee(llm) -> dict:
    correct = 0
    rows = []
    for task, target, label in JEE_LABELED:
        state = build_state(SCENARIOS[task])
        d = estimate_discrepancy(llm, state, target)["stance_distance"]
        level = {0.2: "low", 0.5: "medium", 0.8: "high"}.get(d, "?")
        correct += (level == label)
        rows.append({"task": task, "target": target, "expected": label, "got": level})
    return {"accuracy": round(correct / len(JEE_LABELED), 3), "n": len(JEE_LABELED), "rows": rows}


def category_override_rate(logs) -> dict:
    """EUE：category 提案被程序 (v,r) 校验改写的比例（按任务/模式分组）。"""
    groups = {}
    for l in logs:
        overridden = any("矛盾，改写为" in n for n in l.update_notes)
        key = (l.task if hasattr(l, "task") else "?", l.mode)
        g = groups.setdefault(key, [0, 0])
        g[0] += 1
        g[1] += overridden
    return {"per_group": {f"{k[0]}/{k[1]}": round(v[1] / v[0], 3) if v[0] else None
                          for k, v in groups.items()},
            "overall": round(sum(v[1] for v in groups.values()) / sum(v[0] for v in groups.values()), 3)
            if groups else None}
