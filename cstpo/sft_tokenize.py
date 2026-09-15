"""SFT 样本 tokenize + loss 掩码（Qwen3 chat template 精确实现）。

口径（SFT_SPEC 2.2 / sft_config.MASK_LABEL_LINE）：
- input_ids：messages 按 Qwen3 apply_chat_template 渲染；
- labels：仅最后一条 assistant 消息的"话语部分"参与 loss；
  掩掉 system/user 全部、历史 assistant 全部、目标回合的标签行前缀、
  以及模板结构标记（<|im_start|>/<|im_end|>）。

实现：把最后一条 assistant 拆成 [标签行][话语] 两段单独编码，
使标签行 token 边界可直接定位（不依赖字符串搜索）。

自检：python -m cstpo.sft_tokenize --self-check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cstpo.sft_config import LABEL_SEP, MAX_SEQ_LEN, MODEL_ID

IGNORE = -100


def tokenize_sample(tk, sample: dict) -> dict:
    """→ {"input_ids": list[int], "labels": list[int]}（未截断）。"""
    messages = sample["messages"]
    label = sample["label"]
    last = messages[-1]
    assert last["role"] == "assistant"
    content = last["content"]
    assert content.startswith(label + LABEL_SEP), \
        f"标签前缀不符: {content[:40]!r} vs {label!r}"
    utterance = content[len(label + LABEL_SEP):]

    # 1) 前缀部分：除最后 assistant 外的全部消息（最后 assistant 用空内容占位）
    prefix_msgs = messages[:-1] + [{"role": "assistant", "content": ""}]
    # 渲染到 "<|im_start|>assistant\n"（空内容后的换行含在 prefix 内）
    prefix_ids = list(tk.apply_chat_template(
        prefix_msgs, tokenize=True, add_generation_prompt=False)["input_ids"])

    # 2) 目标回合内容：标签行（mask） + 话语（学习）
    label_ids = tk.encode(label + LABEL_SEP, add_special_tokens=False)
    utter_ids = tk.encode(utterance, add_special_tokens=False)

    # 3) 结尾结构标记（<|im_end|>\n，不参与 loss）
    tail = tk.encode("<|im_end|>\n", add_special_tokens=False)

    input_ids = prefix_ids + label_ids + utter_ids + tail
    labels = [IGNORE] * len(prefix_ids) + [IGNORE] * len(label_ids) \
        + utter_ids + [IGNORE] * len(tail)
    return {"input_ids": input_ids, "labels": labels}


def tokenize_and_pad(tk, sample: dict) -> dict:
    out = tokenize_sample(tk, sample)
    ids = out["input_ids"][:MAX_SEQ_LEN]
    labs = out["labels"][:MAX_SEQ_LEN]
    pad = MAX_SEQ_LEN - len(ids)
    ids += [tk.pad_token_id] * pad
    labs += [IGNORE] * pad
    return {"input_ids": ids, "labels": labs, "attention_mask": [1] * len(ids)}


def load_tokenizer():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL_ID)
    if tk.pad_token_id is None:
        tk.pad_token_id = tk.eos_token_id
    return tk


def self_check():
    tk = load_tokenizer()
    sample = {"messages": [
        {"role": "system", "content": "You are a helpful and caring friend."},
        {"role": "user", "content": "i am feeling sad."},
        {"role": "assistant", "content": "Question\nWhat happened?"},
    ], "label": "Question"}
    out = tokenize_sample(tk, sample)
    ids, labs = out["input_ids"], out["labels"]
    # 断言：有可学习 token；labels 与 input 对齐；话语 tokens 完整保留
    assert len(ids) == len(labs)
    learn = [i for i, l in zip(ids, labs) if l != IGNORE]
    assert learn, "无学习 token"
    text = tk.decode(learn)
    assert "What happened?" in text, f"话语缺失: {text!r}"
    assert "Question" not in text, f"标签行未被 mask: {text!r}"
    n_learn = sum(1 for l in labs if l != IGNORE)
    n_label = len(tk.encode("Question\n", add_special_tokens=False))
    assert n_learn == len(tk.encode("What happened?", add_special_tokens=False))
    print(f"self-check PASS: 学习 token {n_learn} 个，标签行 {n_label} 个已 mask，"
          f"总长 {len(ids)}")
    # 截断自检
    p = tokenize_and_pad(tk, sample)
    assert len(p["input_ids"]) == MAX_SEQ_LEN == len(p["labels"])
    print(f"pad 自检 PASS: {MAX_SEQ_LEN}")


if __name__ == "__main__":
    self_check()
