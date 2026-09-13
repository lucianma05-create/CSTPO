"""Seed 场景库（Validation 阶段）：每任务 ~36 个 seed，覆盖三模式与六种 RJ 目标组合。

seed 结构：{id, task, profile(可选覆盖), messages(2 轮脚本), expected(目标标签)}。
expected 只用于定向回归统计（coverage），不作为任何 runtime 输入。
三种任务的初始 BDI/E/Persona 复用 run_sim.SCENARIOS（v1.0 冻结基准）。
"""
from __future__ import annotations

from simulator.run_sim import SCENARIOS

# ---------- 定向消息库（按目标 (route, judgment) / 模式）----------
# 经第二轮定向回归验证的消息模板（Prompt审计0912 §14.3）。
MESSAGE_BANK = {
    "bargain": {
        ("central", "accept"): [
            "行，就按80卖给你，你今天就能骑走。",
            "80可以成交，这样你明天开学就有车骑，我也不用再等别人。",
            "80成交，你明天开学正好用上，两全其美。",
        ],
        ("central", "noncommit"): [
            "这车成色很新，85已经比市场价低了，你今晚就能骑走。",
            "85是最低价，我今天已经拒绝几个80的报价。",
            "85送你一个车锁，等于变相便宜了。",
        ],
        ("central", "reject"): [
            "85是最低价，我今天已经拒绝几个80的报价。",
            "市场价都是95以上，85已经是很便宜的价格了。",
            "我这价格都是实价，不信你去问同城其他卖家。",
        ],
        ("peripheral", "accept"): [
            "行，80卖你。我卖车这么多年，跟学生们都是这么爽快成交的。",
            "就80吧，今天交个朋友，以后车有问题随时找我。",
        ],
        ("peripheral", "noncommit"): [
            "已经有好几个人在问了，你再不决定今晚就卖给别人了。",
            "好几个人都在问这车，你考虑一下，别错过。",
        ],
        ("peripheral", "reject"): [
            "别的不说，已经有五个人约了今晚看车，你要再犹豫就真没了。",
            "实话告诉你，今天下午还有人来看，你不定就没了。",
        ],
        "elicit": [
            "你心里的预算到底是多少？",
            "你主要是担心车况，还是担心价格？",
            "你除了价格，还关心哪些方面？",
        ],
        "social": [
            "今天天气不错，骑车上学挺舒服的。",
            "最近学校那边怎么样，忙不忙？",
            "你们学校骑车的人多吗？",
        ],
    },
    "donation": {
        ("central", "accept"): [
            "那就不劝你了，先不捐，等你看到具体项目的账目再决定，这样最稳妥。",
            "不捐也是负责任的选择，等你信任了平台再说。",
            "不捐没关系的，等你看到公开账目再决定，完全来得及。",
        ],
        ("central", "noncommit"): [
            "很多人都是从小额开始的，公益平台现在都能查到每一笔钱的去向。",
            "小额多次捐款积少成多，平台也支持每月定额捐。",
        ],
        ("central", "reject"): [
            "5元就能让一个孩子吃上一顿热饭，积少成多，效果是实打实的。",
            "每笔钱都用在孩子身上，数据都公开的，5元不少。",
        ],
        ("peripheral", "accept"): [
            "很多同事都说等账目透明了再捐，我朋友也是这么做的。",
            "我认识的同事都建议先了解清楚再捐，这样挺好。",
        ],
        ("peripheral", "noncommit"): [
            "大家都说这个平台靠谱，你可以了解一下再说。",
            "这个平台口碑还可以，你可以慢慢了解。",
        ],
        ("peripheral", "reject"): [
            "我上次捐款的公益平台是知名大机构背书的，大家都说靠谱。",
            "我们小区很多人都捐了，说体验很好。",
        ],
        "elicit": [
            "你之前有过捐款的经历吗？",
            "你更关心钱去哪了，还是项目靠不靠谱？",
            "如果有个项目能让你看到钱具体花在哪，你会考虑吗？",
        ],
        "social": [
            "最近工作忙不忙，下班之后一般做什么？",
            "刚买房子是挺累的，装修的事儿也不少吧。",
            "国庆假期有安排吗？",
        ],
    },
    "support": {
        ("central", "accept"): [
            "那就先别发消息，等你准备好再说，这不是逃避。",
            "不急着今天行动也行，先把状态缓过来最重要。",
            "对，先不逼自己，把状态调整好更重要。",
        ],
        ("central", "noncommit"): [
            "先给导师发一条消息只需要两分钟，回复一句进展顺利就够。",
            "先回复导师一句收到、在推进，成本很低。",
        ],
        ("central", "reject"): [
            "其实先动起来就好，发一条消息成本很低，拖着只会更焦虑。",
            "你越拖导师越着急，发条消息反而能缓解压力。",
        ],
        ("peripheral", "accept"): [
            "很多同学也都是先缓一缓，等状态好了再联系导师，这是正常的。",
            "我身边朋友也都有过这种阶段，缓一缓很正常。",
        ],
        ("peripheral", "noncommit"): [
            "我认识几个研究生都是先发消息之后情况就好了，大家都建议先动起来。",
            "大家都说动起来就好，你也可以先试试。",
        ],
        ("peripheral", "reject"): [
            "我认识几个研究生，都是先发一条消息之后情况就好了。",
            "很多师兄都建议先动起来，你考虑一下。",
        ],
        "elicit": [
            "你今天最担心的是什么？",
            "你觉得最大的阻碍是什么？",
            "你希望我怎么帮你？",
        ],
        "social": [
            "听起来这段时间你压力很大，真的辛苦了。",
            "最近有好好吃饭休息吗？",
            "平时有什么缓解压力的方式吗？",
        ],
    },
}

# ---------- 认知习惯参数变体（controllability 用，seed 也可覆盖）----------
PROFILE_VARIANTS = {
    "base": None,                          # 用 SCENARIOS 默认
    "low_eta": {"eta_R": 0.2},
    "high_eta": {"eta_R": 0.8},
    "open": {"tau_A": 0.25, "tau_R": 0.75},
    "resistant": {"tau_A": 0.45, "tau_R": 0.80},
}


def _second_message(task: str, kind: str) -> str:
    """给两轮脚本补一个与首轮同方向、不同措辞的第二条消息（持续压力/追问）。"""
    bank = MESSAGE_BANK[task]
    return bank[kind][-1]


def build_seeds() -> list[dict]:
    """生成全部 seed：每任务 40 个 =（六 RJ×2 消息 + elicit×2 + social×2）× 2 profile 变体，
    三任务共 120 个（覆盖三模式与六种 RJ 目标组合 + Θ 变体）。"""
    seeds = []
    for task in SCENARIOS:
        bank = MESSAGE_BANK[task]
        n = 0
        kinds = [("central", "accept"), ("central", "noncommit"), ("central", "reject"),
                 ("peripheral", "accept"), ("peripheral", "noncommit"), ("peripheral", "reject"),
                 "elicit", "social"]
        for kind in kinds:
            msgs = bank[kind]
            for i, m1 in enumerate(msgs):
                m2 = _second_message(task, kind)
                # 每条消息配 2 个 profile 变体：base + η 变体（交替 low/high）
                variants = [None, {"eta_R": 0.2 if i % 2 == 0 else 0.8}]
                for pname, prof in (("base", variants[0]), ("eta_var", variants[1])):
                    n += 1
                    seeds.append({
                        "id": f"{task}_{kind if isinstance(kind, str) else kind[0][:3] + '_' + kind[1]}_{i}_{pname}",
                        "task": task,
                        "profile": prof,
                        "messages": [m1, m2],
                        "expected": {"mode": "elicit" if kind == "elicit" else "social" if kind == "social" else "influence",
                                     "route": None if isinstance(kind, str) else kind[0],
                                     "judgment": None if isinstance(kind, str) else kind[1]},
                    })
        print(f"[seeds] {task}: {n} seeds")
    return seeds


if __name__ == "__main__":
    seeds = build_seeds()
    import json
    json.dump(seeds, open("evaluation/cases/seeds.json", "w"), ensure_ascii=False, indent=1)
    print(f"total: {len(seeds)}")
