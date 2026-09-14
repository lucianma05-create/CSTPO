"""SeedAdapter：原始数据 → 种子草稿（02§3 envelope、§12 白名单、§6/§7 规则）。

本模块只做程序化部分（白名单过滤、cutoff、采样、envelope 组装）；
Persona/BDI 语义提取与中文渲染在 generate_seeds.py 用 LLM 完成。

种子 JSON schema（02§3 + 人工预审扩展）：

- provenance / task / actor_task_view / user_task_view
- persona（结构化事实 + 渲染文本 en/zh）
- cognitive_profile（首版固定 0.6/0.35/0.70，controlled_default）
- initial_bdi（content_en/zh、strength∈{1,2,3}、core 显式、polarity、
  evidence_message_ids、source_kind）
- initial_emotion（category 映射 + raw 证据；valence/arousal 待校准）
- context（前缀消息 en/zh + 原始索引 + cutoff 规则 + 下一发言者）
- runtime（prefix_reset、deterministic、v1.0.1-fix）
- evaluation_reference（仅指针，不载入内容）
- review（人工预审区：九项检查表 + review_notes 自定义字段 + custom dict）
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
        "with", "is", "are", "was", "were", "i", "me", "my", "you", "your",
        "it", "its", "that", "this", "he", "she", "they", "we", "do", "does",
        "did", "have", "has", "had", "be", "been", "not", "no", "so", "just",
        "about", "at", "by", "from", "up", "down", "out", "if", "then", "than",
        "too", "very", "can", "could", "would", "should", "will", "what",
        "when", "where", "who", "why", "how", "there", "here", "some", "any",
        "much", "many", "more", "most", "get", "got", "feel", "really", "like",
        "know", "think", "want", "going", "go", "make", "made", "still",
        "even", "also", "because", "but", "them", "their", "his", "her", "him",
        "as", "all", "into", "over", "one", "two", "am", "don", "t", "s",
        "ve", "ll", "m", "re"}

EMOTION_MAP_V1 = {   # 02§12.4 草案
    "anxiety": "anxiety", "nervousness": "anxiety", "anger": "anger",
    "sadness": "sadness", "depression": "sadness", "fear": "anxiety",
    "disgust": "frustration", "shame": "frustration",
    "pain": "sadness", "jealousy": "sadness", "guilt": "sadness",
}

DEFAULT_PROFILE = {"eta_R": 0.6, "tau_A": 0.35, "tau_R": 0.70,
                   "source_kind": "controlled_default", "condition": "default"}

CHECKLIST = [  # 02§10 九项检查表（人工预审勾选）
    "来源与时间：每个 Persona/BDI 事实都有允许原始字段或截止前证据",
    "未来不变性：无未来信息进入初始化路径",
    "可见性：Actor/Cog-Sim/critic/judge 输入符合角色视图",
    "语义分工：固定 Persona 没有锁死当前信念/意图/情绪",
    "结构合法性：数值范围、容量、ID、极性、情绪类别、角色与 cutoff 合法",
    "缺失与覆盖：B/D/I 数量、默认字段率、任务关键约束遗漏率",
    "参数有效性：参数条件与判断表一致",
    "Persona 效用：背景信息改善事实一致性（vs 仅必要任务事实）",
    "可变性：有效依据下可合理变化、无依据时不漂移",
]

REVIEW_TEMPLATE = {"reviewer": "", "review_date": None,
                   "review_notes": "",  # 自定义字段：任意批注
                   "custom": {},        # 自定义字段：结构化附加信息
                   "checklist": {c: None for c in CHECKLIST}}  # True/False/None


def words(t: str) -> set:
    return {w for w in re.findall(r"[a-z']{3,}", t.lower())
            if w not in STOP and not w.endswith("'s") and w not in ("don", "t")}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:10]


def _load(p: Path):
    return json.loads(p.read_text())


# ================= ESConv =================

TURN_WINDOW = 6  # Q2 裁定（2026-09-14）：B 方案，触发点后最多延长 6 个用户回合


def _user_segments(msgs, role: str) -> list[tuple[int, int]]:
    """连续同角色消息段 [(start, end)]，段=一个交互回合的该方发言。"""
    segs, i = [], 0
    while i < len(msgs):
        if msgs[i]["from"] == role:
            j = i
            while j + 1 < len(msgs) and msgs[j + 1]["from"] == role:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


def esconv_cutoff(record: dict, window: int = TURN_WINDOW) -> dict | None:
    """seeker 首条实质披露（实词重叠≥1 或长度≥40，前 8 条内），
    之后最多延长 window 个 seeker 回合，截止于窗口内最后一个 seeker 段。
    返回 None = low_info。"""
    situation = record["metadata"].get("situation", "") or ""
    sw = words(situation)
    msgs = record["conversations"]
    found = None
    for i, m in enumerate(msgs[:8]):
        if m["from"] != "human":
            continue
        w = words(m["value"])
        if (sw and (w & sw)) or len(m["value"]) >= 40:
            found = i
            break
    if found is None:
        return None
    segs = _user_segments(msgs, "human")
    k = next(k for k, (s, e) in enumerate(segs) if s <= found <= e)
    end_seg = segs[min(k + window, len(segs) - 1)]
    return {"end_index": end_seg[1], "trigger_index": found,
            "rule": f"first-substantive-seeker + window-{window}-turns"}


def build_esconv_seed(record: dict, seed_id: str, seed_rng: random.Random) -> dict:
    meta = record["metadata"]
    cut = esconv_cutoff(record)
    prefix = []
    if cut is not None:
        for i in range(cut["end_index"] + 1):
            m = record["conversations"][i]
            prefix.append({"role": "seeker" if m["from"] == "human" else "supporter",
                           "text_en": m["value"], "original_index": i})
    # 采样多样性用：rng 用于选种子，cutoff 本身确定性
    seed_rng.random()
    seed = {
        "seed_id": seed_id,
        "provenance": {
            "dataset": "esconv",
            "source_path": "data/raw/esconv/train.json",
            "source_md5": md5(RAW / "esconv" / "train.json"),
            "source_id": record["id"],
            "orig_split": "train",
            "group_id": record["id"],  # case 级分组（02§13.2）
            "parser_version": "seed_adapter v4.2 frozen 2026-09-15",
        },
        "task": {"task_id": "esconv", "actor_role": "supporter",
                 "user_role": "seeker", "language": "en（原文保留，zh 仅供审阅）",
                 "description": "情感支持：改善 seeker 情绪/希望并促成可行行动打算"},
        "actor_task_view": {"visible": "截止点之前的完整对话"},
        "user_task_view": {"visible": "截止点之前的完整对话"},
        "persona": {"source_fields": ["metadata.situation", "metadata.problem_type",
                                      "metadata.experience_type"],
                    "situation_en": meta.get("situation"),
                    "problem_type": meta.get("problem_type"),
                    "experience_type": meta.get("experience_type"),
                    "facts": [], "rendered_en": "", "rendered_zh": ""},
        "cognitive_profile": dict(DEFAULT_PROFILE),
        "initial_bdi": {"beliefs": [], "desires": [], "intentions": []},
        "initial_emotion": {
            "category": EMOTION_MAP_V1.get(meta.get("emotion_type"), "neutral"),
            "valence": None, "arousal": None,
            "raw": {"emotion_type": meta.get("emotion_type"),
                    "initial_emotion_intensity":
                        meta.get("survey_score", {}).get("seeker", {})
                        .get("initial_emotion_intensity")},
            "mapping_version": "02§12.4 draft", "note": "valence/arousal 待校准",
        },
        "context": {"prefix": prefix,
                    "cutoff": cut if cut else {"rule": "low_info: 前 4 回合无实质披露"},
                    "next_speaker": "supporter",
                    "low_info": cut is None,
                    "message_indices": [p["original_index"] for p in prefix]},
        "runtime": {"init_mode": "prefix_reset", "route_mode": "deterministic",
                    "model": "deepseek-flash", "simulator_version": "v1.0.1-fix"},
        "evaluation_reference": {"note": "原始终局问卷/后续对话为未来信息，"
                                         "独立离线保存，不载入本文件"},
        "review": copy.deepcopy(REVIEW_TEMPLATE),
    }
    return seed


def sample_esconv(n: int, seed: int = 20260914) -> list[dict]:
    recs = _load(RAW / "esconv" / "train.json")
    recs = [r for r in recs if not r["metadata"].get("is_failed_sample")]
    rng = random.Random(seed)
    by_emo: dict[str, list] = {}
    for r in recs:
        by_emo.setdefault(r["metadata"].get("emotion_type") or "unknown", []).append(r)
    picked: list = []
    # 覆盖情绪类别（每类尽量 1-2 个），再随机补齐，并确保覆盖 low_info
    low_info = [r for r in recs if esconv_cutoff(r) is None]
    picked.extend(low_info[:2])
    for emo, lst in sorted(by_emo.items()):
        n_take = 2 if len(lst) >= 8 else 1
        picked.extend(rng.sample(lst, min(n_take, len(lst))))
    seen = {id(x): x for x in picked}
    rest = [r for r in recs if id(r) not in seen]
    rng.shuffle(rest)
    for r in rest:
        if len(seen) >= n:
            break
        seen[id(r)] = r
    chosen = list(seen.values())[:n]
    rng.shuffle(chosen)
    seeds = []
    for i, r in enumerate(chosen, 1):
        seeds.append(build_esconv_seed(r, f"esconv_{i:02d}", rng))
    return seeds


# ================= P4G =================

GREET = {"hi", "hello", "hey", "good", "morning", "afternoon", "evening",
         "thanks", "thank", "yes", "no", "okay", "ok", "sure", "fine",
         "im", "i'm"}


def p4g_cutoff(record: dict, window: int = TURN_WINDOW) -> dict | None:
    """persuadee 首条非纯寒暄（长度≥20 且非短寒暄，前 6 条内），
    之后最多延长 window 个 persuadee 回合，截止于窗口内最后一个 persuadee 段。"""
    msgs = record["conversations"]
    found = None
    for i, m in enumerate(msgs[:6]):
        if m["from"] != "human":
            continue
        v = m["value"].strip().lower()
        if len(v) >= 20 and not (v.split()[0].rstrip(",.!?") in GREET and len(v) < 30):
            found = i
            break
    if found is None:
        return None
    segs = _user_segments(msgs, "human")
    k = next(k for k, (s, e) in enumerate(segs) if s <= found <= e)
    end_seg = segs[min(k + window, len(segs) - 1)]
    return {"end_index": end_seg[1], "trigger_index": found,
            "rule": f"first-substantive-persuadee + window-{window}-turns"}


def build_p4g_seed(record: dict, seed_id: str, seed_rng: random.Random) -> dict:
    seed_rng.random()
    pe = record["metadata"]["participants"]["persuadee"]
    cut = p4g_cutoff(record)
    prefix = []
    if cut is not None:
        for i in range(cut["end_index"] + 1):
            m = record["conversations"][i]
            prefix.append({"role": "persuadee" if m["from"] == "human" else "persuader",
                           "text_en": m["value"], "original_index": i})
    scales = {k: pe[k] for k in pe if k.endswith(".x") and k != "B2.x"}
    demo = {k: pe.get(k) for k in ("age.x", "sex.x", "race.x", "edu.x",
                                   "marital.x", "employment.x", "income.x",
                                   "religion.x", "ideology.x")}
    seed = {
        "seed_id": seed_id,
        "provenance": {"dataset": "p4g", "source_path": "data/raw/p4g/train.json",
                       "source_md5": md5(RAW / "p4g" / "train.json"),
                       "source_id": record["id"], "orig_split": "train",
                       "group_id": pe.get("B3"),  # 未见用户泛化按 B3（02§13.3）
                       "annotated_subset": record["metadata"].get("is_annotated_subset"),
                       "parser_version": "seed_adapter v4.2 frozen 2026-09-15"},
        "task": {"task_id": "p4g", "actor_role": "persuader",
                 "user_role": "persuadee", "language": "en",
                 "description": "劝说捐赠：达成明确、自愿、非纯条件性的捐赠承诺"},
        "actor_task_view": {"visible": "截止点之前对话；无私有任务信息"},
        "user_task_view": {"visible": "截止点之前对话；自身问卷与背景（训练侧用户模型用）"},
        "persona": {"measurements": scales, "demographics": demo,
                    "facts": [], "rendered_en": "", "rendered_zh": "",
                    "note": "问卷为任务前测量（02§5.2）；人口学仅在与任务相关时渲染；"
                            "B5/B6/B7 码表未核对，全排除（02§12.2）"},
        "cognitive_profile": dict(DEFAULT_PROFILE),
        "initial_bdi": {"beliefs": [], "desires": [], "intentions": []},
        "initial_emotion": {"category": "neutral", "valence": None, "arousal": None,
                            "raw": {}, "mapping_version": "—",
                            "note": "P4G 无起点情绪测量，controlled_default neutral"},
        "context": {"prefix": prefix,
                    "cutoff": cut if cut else {"rule": "low_info: 无实质 persuadee 消息"},
                    "next_speaker": "persuader", "low_info": cut is None,
                    "message_indices": [p["original_index"] for p in prefix]},
        "runtime": {"init_mode": "prefix_reset", "route_mode": "deterministic",
                    "model": "deepseek-flash", "simulator_version": "v1.0.1-fix"},
        "evaluation_reference": {"note": "原始捐赠结果/承诺金额为未来信息，"
                                         "独立离线保存，不载入本文件"},
        "review": copy.deepcopy(REVIEW_TEMPLATE),
    }
    return seed


def sample_p4g(n: int, seed: int = 20260914) -> list[dict]:
    recs = _load(RAW / "p4g" / "train.json")
    rng = random.Random(seed)
    ann = [r for r in recs if r["metadata"].get("is_annotated_subset")]
    unann = [r for r in recs if not r["metadata"].get("is_annotated_subset")]
    picked = rng.sample(ann, min(len(ann), n // 2)) + rng.sample(unann, min(len(unann), n - n // 2))
    rng.shuffle(picked)
    picked = picked[:n]
    return [build_p4g_seed(r, f"p4g_{i:02d}", rng) for i, r in enumerate(picked, 1)]


# ================= CraigslistBargain =================

PRICE_INTENTS = {"init-price", "counter-price", "vague-price"}


def cb_cutoff(record: dict, window: int = TURN_WINDOW) -> dict | None:
    """首个价格事件后第一条卖方 message，再最多延长 window 个卖方 message
    （取窗口内最后一条）；价格事件后无卖方 message → 价格事件前最后一条卖方
    message；无任何卖方 message → 空历史。"""
    evs = record["events"]
    kbs = record["scenario"]["kbs"]
    role_of = {str(i): kbs[i]["personal"]["Role"] for i in range(len(kbs))}
    pi = None
    for i, e in enumerate(evs):
        if e["action"] == "offer" or ((e.get("metadata") or {}).get("intent")
                                      in PRICE_INTENTS):
            pi = i
            break
    if pi is not None:
        after = [j for j in range(pi + 1, len(evs))
                 if evs[j]["action"] == "message"
                 and role_of.get(str(evs[j]["agent"])) == "seller"]
        if after:
            end = after[min(window, len(after)) - 1]
            return {"end_index": end, "price_event_index": pi,
                    "rule": f"first-seller-after-price + window-{window}-turns"}
    sm = [j for j, e in enumerate(evs)
          if e["action"] == "message" and role_of.get(str(e["agent"])) == "seller"]
    if sm:
        return {"end_index": sm[-1], "price_event_index": pi,
                "rule": "last-seller-before-price"}
    return {"end_index": -1, "price_event_index": pi,
            "rule": "empty-history (no seller message)"}


def build_cb_seed(record: dict, seed_id: str, seed_rng: random.Random) -> dict:
    seed_rng.random()
    kbs = record["scenario"]["kbs"]
    buyer = next(k for k in kbs if k["personal"]["Role"] == "buyer")
    seller = next(k for k in kbs if k["personal"]["Role"] == "seller")
    cut = cb_cutoff(record)
    prefix, task_events = [], []
    for i in range(cut["end_index"] + 1):
        e = record["events"][i]
        role = "seller" if str(e["agent"]) == "1" or \
            (str(e["agent"]) == "1") else "buyer"
        role = "seller" if kbs[int(e["agent"])]["personal"]["Role"] == "seller" else "buyer"
        if e["action"] == "message":
            prefix.append({"role": role, "text_en": e["data"],
                           "original_index": i,
                           "intent": (e.get("metadata") or {}).get("intent")})
        else:
            task_events.append({"index": i, "action": e["action"],
                                "agent": role,
                                "metadata": e.get("metadata")})
    seed = {
        "seed_id": seed_id,
        "provenance": {"dataset": "craigslistbargain",
                       "source_path": "data/raw/craigslistbargain/train_parsed.json",
                       "source_md5": md5(RAW / "craigslistbargain" / "train_parsed.json"),
                       "source_id": record["uuid"], "orig_split": "train",
                       "group_id": record["scenario_uuid"],  # 未见场景按 scenario（02§13.4）
                       "parser_version": "seed_adapter v4.2 frozen 2026-09-15"},
        "task": {"task_id": "craigslistbargain", "actor_role": "buyer",
                 "user_role": "seller", "language": "en",
                 "description": "讨价还价：达成成交并最大化买方收益（04§2.3）"},
        "actor_task_view": {"item": buyer["item"], "target": buyer["personal"]["Target"],
                            "bottomline": buyer["personal"]["Bottomline"],
                            "note": "Target 是目标价非硬底线（02§5.3）"},
        "user_task_view": {"item": seller["item"], "target": seller["personal"]["Target"],
                           "bottomline": seller["personal"]["Bottomline"],
                           "note": "仅 Cog-Sim 侧可见，Actor 不得读取"},
        "persona": {"facts": [], "rendered_en": "", "rendered_zh": "",
                    "item_facts": {"title": buyer["item"].get("Title"),
                                   "category": buyer["item"].get("Category"),
                                   "price": buyer["item"].get("Price")}},
        "cognitive_profile": dict(DEFAULT_PROFILE),
        "initial_bdi": {"beliefs": [], "desires": [], "intentions": []},
        "initial_emotion": {"category": "neutral", "valence": None, "arousal": None,
                            "raw": {}, "mapping_version": "—",
                            "note": "CB 无起点情绪测量，controlled_default neutral"},
        "context": {"prefix": prefix, "task_events_in_prefix": task_events,
                    "cutoff": cut, "next_speaker": "buyer",
                    "empty_history": cut["end_index"] < 0,
                    "message_indices": [p["original_index"] for p in prefix]},
        "runtime": {"init_mode": "prefix_reset", "route_mode": "deterministic",
                    "model": "deepseek-flash", "simulator_version": "v1.0.1-fix"},
        "evaluation_reference": {"note": "原始成交价/结果为未来信息，"
                                         "独立离线保存，不载入本文件"},
        "review": copy.deepcopy(REVIEW_TEMPLATE),
    }
    return seed


def sample_cb(n: int, seed: int = 20260914) -> list[dict]:
    recs = _load(RAW / "craigslistbargain" / "train_parsed.json")
    rng = random.Random(seed)
    groups = {"first-seller-after-price": [], "last-seller-before-price": [],
              "empty-history (no seller message)": []}
    for r in recs:
        rule = cb_cutoff(r)["rule"]
        if rule.startswith("first-seller-after-price"):
            groups["first-seller-after-price"].append(r)
        elif rule == "last-seller-before-price":
            groups["last-seller-before-price"].append(r)
        else:
            groups["empty-history (no seller message)"].append(r)
    # 按实测比例 76/20/4 取整到 n
    n_a = round(n * 0.76)
    n_b = round(n * 0.20)
    n_c = n - n_a - n_b
    picked = (rng.sample(groups["first-seller-after-price"], min(len(groups["first-seller-after-price"]), n_a))
              + rng.sample(groups["last-seller-before-price"], min(len(groups["last-seller-before-price"]), n_b))
              + rng.sample(groups["empty-history (no seller message)"],
                           min(len(groups["empty-history (no seller message)"]), n_c)))
    while len(picked) < n:
        rest = [r for r in recs if id(r) not in {id(x) for x in picked}]
        if not rest:
            break
        picked.append(rng.choice(rest))
    rng.shuffle(picked)
    picked = picked[:n]
    return [build_cb_seed(r, f"cb_{i:02d}", rng) for i, r in enumerate(picked, 1)]


def sample_seeds(task: str, n: int = 30, seed: int = 20260914) -> list[dict]:
    return {"esconv": sample_esconv, "p4g": sample_p4g,
            "craigslistbargain": sample_cb}[task](n, seed)
