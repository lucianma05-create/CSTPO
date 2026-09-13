"""真实讨价还价演示：LLM 卖家 agent vs 认知用户模拟器。

卖家有自己的价格目标（不低于 85，理想 90，可附赠车锁），
只看到对话文本，看不到买家内部状态；
买家是 BDI-E 模拟器，内部状态逐轮演化。

终止条件：买家 user_done=True 或达到 --max-turns。
成交与否由驱动层用简单规则粗判（模拟器不做任务判定，文档 §42）。

用法:
    cd Cog-Sim && python -m simulator.demo_live_bargain --max-turns 8
"""
from __future__ import annotations

import argparse
import re

from simulator.llm import LLMClient
from simulator.profile.cognitive_profile import build_habit_card
from simulator.simulator import UserSimulator
from simulator.state.schema import BDIItem, CognitiveProfile, Emotion, UserState

SELLER_PERSONA = """你是二手自行车卖家，在闲鱼上卖一辆成色很新的山地车，市场价约 100 元。
- 你的最低接受价是 85 元，理想成交价 90 元，低于 85 坚决不卖（今天已经拒绝过几个 80 的报价）。
- 可以接受的让步：附赠一个车锁（进价约 10 元），或者帮调刹车。
- 买家接受 85 元及以上时，爽快成交，主动约好看车/提车时间。
- 说话口语化、简短，像真实的闲鱼卖家；不要主动提到"最低接受价是85"这样的内部底线，但可以用"已经拒绝过几个80的报价"这类说法。
- 重要：每轮都要给出实质进展，不要重复上一轮的承诺。如果要发照片/视频，就当作已经发出并直接描述车况内容（车架无锈、刹车灵敏、变速顺畅）；买家答应85带锁时立刻确认成交、约定时间地点。
- 成交确定后，简短确认并自然道别（如"定了，四点见，拜拜"），不要再重复同一句话。"""

SELLER_PROMPT = """{persona}

对话历史：
{history}

买家最新消息：
{buyer_msg}

请回复买家（只输出回复内容，不要输出 JSON 或解释）。"""


def build_buyer() -> UserState:
    state = UserState(
        persona=(
            "你是一个想买二手自行车的学生，预算有限，但明天开学要用车，所以比较着急。"
            "你砍价比较直接，也愿意用小让步换取成交。"
        ),
        profile=CognitiveProfile(eta_R=0.75, tau_A=0.45, tau_R=0.80),
        beliefs=[BDIItem("B1", "belief", "卖家最终可能接受80元", 3.0)],
        desires=[
            BDIItem("D1", "desire", "希望价格尽可能低", 3.8),
            BDIItem("D2", "desire", "希望今天成交", 3.1),
        ],
        intentions=[BDIItem("I1", "intention", "坚持80元", 3.0)],
        emotion=Emotion(valence=0.1, arousal=0.4, category="neutral"),
    )
    state.habit_card = build_habit_card(
        state.profile.eta_R, state.profile.tau_A, state.profile.tau_R
    )
    return state


def seller_reply(llm: LLMClient, history: list[dict], buyer_msg: str) -> str:
    hist_text = "\n".join(f"{m['role']}: {m['text']}" for m in history[-10:])
    out = llm.chat(
        [{"role": "user", "content": SELLER_PROMPT.format(
            persona=SELLER_PERSONA, history=hist_text, buyer_msg=buyer_msg)}],
        max_tok=200, json_mode=False,
    )
    return out.strip()


DEAL_PATTERNS = [
    r"成交", r"85就85", r"90就90", r"8\d就8\d", r"就按8\d", r"就按9\d",
    r"行，8\d", r"行，9\d", r"可以，8\d", r"可以，9\d", r"那就8\d", r"那就9\d",
    r"(?:直接|就)定", r"85带锁", r"接受8\d", r"定了",
]


def rough_deal_check(history: list[dict]) -> bool:
    """驱动层粗判（外部视角，非模拟器职责）：买家的任何话语是否表达了成交意愿。"""
    for m in history:
        if m["role"] == "user" and any(re.search(p, m["text"]) for p in DEAL_PATTERNS):
            return True
    return False


def is_repetition(a: str, b: str, threshold: float = 0.6) -> bool:
    """字符 bigram Jaccard 相似度，用于检测卖家回复陷入循环。"""
    def bigrams(s):
        s = "".join(s.split())
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return False
    return len(A & B) / len(A | B) > threshold


def fmt_state_trace(log) -> str:
    """一行紧凑的内部状态轨迹：变更的 BDI 项 + 情绪。"""
    parts = []
    for key, lst in (("beliefs", log.bdi_after["beliefs"]), ("desires", log.bdi_after["desires"]),
                     ("intentions", log.bdi_after["intentions"])):
        for it in lst:
            before = next(
                (b for b in log.bdi_before.get(key, []) if b["id"] == it["id"]), None
            )
            if before and abs(before["strength"] - it["strength"]) > 1e-6:
                parts.append(f"{it['id']}:{before['strength']:.1f}→{it['strength']:.1f}")
            elif before is None:
                parts.append(f"新{it['id']}({it['strength']:.1f})")
    emo = f"{log.emotion_before['category']}→{log.emotion_after['category']}"
    return " ".join(parts) + (f" | {emo}" if emo != "→" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    llm = LLMClient(model=args.model)
    buyer = UserSimulator(build_buyer(), llm)
    print(f"===== 讨价还价：卖家 = LLM agent（底价85），买家 = 认知模拟器（模型 {llm.model}）=====\n")
    print("买家初始状态: B1 卖家最终可能接受80元(3.0) | D1 低价(3.8) D2 今天成交(3.1) | I1 坚持80(3.0)")
    print(f"买家习惯: eta_R={buyer.state.profile.eta_R} tau_A={buyer.state.profile.tau_A} tau_R={buyer.state.profile.tau_R}\n")

    # 卖家开场
    opening = "你好，车还在，成色很新，可以先看看照片。"
    prev_seller_msg = ""
    stall_count = 0
    for turn in range(1, args.max_turns + 1):
        print(f"── 第 {turn} 轮 ──")
        if turn == 1:
            seller_msg = opening
            print(f"[卖家] {seller_msg}")
        else:
            seller_msg = seller_reply(llm, buyer.state.history[:-1], buyer.state.history[-1]["text"])
            print(f"[卖家] {seller_msg}")
            if is_repetition(seller_msg, prev_seller_msg):
                stall_count += 1
                if stall_count >= 2:
                    print("\n[驱动层] 卖家回复连续重复，检测到循环，停止。")
                    break
            else:
                stall_count = 0
        prev_seller_msg = seller_msg

        user_reply = buyer.simulate_turn(seller_msg)
        log = buyer.logs[-1]
        print(f"[买家] {user_reply}")
        print(f"    （内部: {log.mode} | {log.route or '-'}/{log.judgment or '-'} | {fmt_state_trace(log)}）")

        if buyer.conversation_ended:
            print("\n[驱动层] 买家表示结束对话，停止。")
            break
    else:
        print(f"\n[驱动层] 达到回合上限 {args.max_turns}，停止。")

    print("\n===== 买家最终内部状态 =====")
    print(fmt_state_trace(buyer.logs[-1]))
    print(f"情绪: {buyer.state.emotion}")
    deal = rough_deal_check(buyer.state.history)
    print(f"\n[驱动层粗判] 是否成交: {'✅ 是' if deal else '❌ 否'}（仅正则匹配买家话语，正式评估请用外部 Task Evaluator，文档 §42）")


if __name__ == "__main__":
    main()
