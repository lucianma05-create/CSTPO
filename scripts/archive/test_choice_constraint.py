"""精确复刻 RL agent loop 的 token 流，定位乱码根因。

与 test 版差异：完全复用 verl 的 apply_chat_template helper 和
initialize_system_prompt（AgentLoopBase 同款），12 轮循环。
"""
import sys

sys.path.insert(0, "/data/user21300120/mmh/CSTPO")
sys.path.insert(0, "/data/user21300120/mmh/CSTPO/Cog-Sim")

from openai import OpenAI
from transformers import AutoTokenizer

from verl.utils.chat_template import apply_chat_template as verl_apply_chat_template
from verl.utils.chat_template import initialize_system_prompt

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
MODEL = "/publicdata/model/Qwen3-8B"
TEMPLATE_KWARGS = {"enable_thinking": False}

ESCONV_LABELS = ["Question", "Restatement or Paraphrasing", "Reflection of feelings",
                 "Self-disclosure", "Affirmation and Reassurance",
                 "Providing Suggestions", "Information", "Others"]


def gen(prompt_ids, max_tokens, choice=None):
    body = {}
    if choice:
        body["structured_outputs"] = {"choice": choice}
    r = client.completions.create(model=MODEL, prompt=prompt_ids,
                                  max_tokens=max_tokens, temperature=0.2,
                                  extra_body=body)
    return r.choices[0].text


def main():
    tk = AutoTokenizer.from_pretrained("/publicdata/model/Qwen3-8B")
    system_prompt = initialize_system_prompt(tk, **TEMPLATE_KWARGS)
    print(f"system_prompt len = {len(system_prompt)}")

    msgs = [
        {"role": "system", "content": "You are a helpful and caring friend."},
        {"role": "user", "content": "I am feeling sad. My boyfriend left me out of the blue."},
    ]
    all_ids = verl_apply_chat_template(tk, msgs, add_generation_prompt=True,
                                       tokenize=True, **TEMPLATE_KWARGS)
    nl_ids = tk.encode("\n", add_special_tokens=False)

    user_replies = [
        "Yes, I want to talk. I can't concentrate on anything.",
        "It hurts so much. I keep wondering what I did wrong.",
        "I keep crying at night. I feel so alone.",
    ]

    for turn in range(4):
        # 段 1：标签（choice 约束）
        label_text = gen(all_ids, max_tokens=8, choice=ESCONV_LABELS)
        label_ids = tk.encode(label_text, add_special_tokens=False)
        print(f"轮{turn+1} 标签: {label_text!r} ({len(label_ids)} tokens)")
        all_ids = all_ids + label_ids

        # 段 2：话语
        utter = gen(all_ids + nl_ids, max_tokens=128)
        all_ids = all_ids + nl_ids + tk.encode(utter, add_special_tokens=False)
        print(f"轮{turn+1} 话语: {utter[:80]!r}")

        # user 增量（verl helper 同款 remove_system_prompt）
        add_msgs = [{"role": "assistant", "content": utter},
                    {"role": "user", "content": user_replies[turn]}]
        inc = verl_apply_chat_template(tk, add_msgs, add_generation_prompt=True,
                                       tokenize=True, **TEMPLATE_KWARGS)
        inc = inc[len(system_prompt):]
        print(f"轮{turn+1} user 增量: {len(inc)} tokens, 尾部={tk.decode(inc[-12:], skip_special_tokens=False)!r}")
        all_ids = all_ids + inc

    print("\n最终 all_ids 尾部:", tk.decode(all_ids[-40:], skip_special_tokens=False))


if __name__ == "__main__":
    main()
