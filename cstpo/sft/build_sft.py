"""SFT 数据构建（SFT_SPEC_20260915 的步骤 ⑤）。

输入：data/raw 三任务 train 真人语料 +（可选）香草生成对话目录。
输出：data/sft/<task>/{train.jsonl, stats.json}——
  每样本 {"messages": [{role, content}...], "label": str|"<unlabeled>"}
  assistant 回合 content = "<标签>\n<话语>"；标签行在训练脚本里按 "label"
  字段重新 tokenize 得到 loss mask 区间（SPEC 2.2 话语 SFT 口径）。

构造规则（SPEC §4）：
- 样本以 user 消息开头（开头 assistant 消息丢弃）；
- 同角色连续消息合并（换行拼接）；
- 每条 assistant 回合切一条样本（历史 = 该点之前全部）；
- 系统提示 = 论文香草提示（与 agent.py 同一模板来源，逐任务固定值填充）；
- 标签经 label_maps 归一化；无标签回合用占位 <unlabeled>。

用法：cd CSTPO && python -m cstpo.sft.build_sft --task all --max-turns 30
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from cstpo.core.label_maps import cb_buyer_idx, cb_label, esconv_label, p4g_label

RAW = ROOT / "data" / "raw"
OUT = Path(os.environ.get("SFT_DATA_DIR", str(ROOT / "data" / "sft")))

UNLABELED = "<unlabeled>"
LABEL_SEP = "\n"  # SPEC：标签单独一行 + 换行 + 话语

# 系统提示模板（与 cstpo/agent.py 论文香草提示同源；真实语料无种子，
# 用对话内字段填充等价信息）
SYSTEM_TMPL = {
    "esconv": ("You are a helpful and caring friend. Your friend has come to you "
               "with some emotional problem: {situation}\n"
               "Please help your friend by continuing the conversation. Make your "
               "response short and to the point.\n"
               "Respond in this format: assistant: <response>"),
    "p4g": ("Now enter the role-playing mode. In the following conversation, you "
            "will play as a Persuader who is trying to persuade the Persuadee to "
            "donate to the charity called Save the Children.\n"
            "Save the Children is head-quartered in London, and they work to help "
            "fight poverty around the world.\n"
            "You are the Persuader who is trying to convince the Persuadee to "
            "donate to a charity called Save the Children.\n"
            "Please reply with only one short and persuasive sentence."),
    "craigslistbargain": ("Now enter the role-playing mode. In the following "
                          "conversation, you will play as a buyer in a price "
                          "bargaining game.\n"
                          "You are the buyer who is trying to buy the {item} with "
                          "the price of {target}.\n"
                          "Product description: {desc}\n"
                          "Please reply with only one short and succinct sentence.\n"
                          "Now start the game."),
}

ROLE_OF = {
    "esconv": {"seeker": "user", "supporter": "assistant"},
    "p4g": {"persuadee": "user", "persuader": "assistant"},
    "craigslistbargain": {"seller": "user", "buyer": "assistant"},
}
ACTOR_ROLE = {"esconv": "supporter", "p4g": "persuader",
              "craigslistbargain": "buyer"}


def iter_turns(task, dial):
    """产出 [(role: 'user'|'assistant', text, raw_label|None)]，按对话顺序。"""
    if task == "esconv":
        for u in dial.get("metadata", {}).get("utterance_annotations", []):
            r = ROLE_OF[task].get(u.get("role"))
            if r is None:
                continue
            yield r, u["text"], esconv_label(u.get("strategy"))
    elif task == "p4g":
        for u in dial.get("metadata", {}).get("utterance_annotations", []):
            r = ROLE_OF[task].get(u.get("role"))
            if r is None:
                continue
            anns = u.get("annotations", [])
            raw = anns[0].get("persuader_label_1") if anns else None
            yield r, u["text"], p4g_label(raw) if raw else None
    else:
        bi = cb_buyer_idx(dial)
        if bi is None:
            return
        for ev in dial.get("events", []):
            if ev.get("action") != "message":
                continue
            r = "assistant" if ev.get("agent") == bi else "user"
            yield r, ev.get("data", ""), cb_label(
                (ev.get("metadata") or {}).get("intent"))


def system_for(task, dial) -> str:
    if task == "esconv":
        return SYSTEM_TMPL[task].format(
            situation=dial.get("metadata", {}).get("situation") or "(unknown)")
    if task == "p4g":
        return SYSTEM_TMPL[task]
    bi = cb_buyer_idx(dial)
    kb = dial["scenario"]["kbs"][bi]
    item = kb.get("item", {}) or {}
    return SYSTEM_TMPL[task].format(
        item=item.get("Title", "item"),
        target=kb.get("personal", {}).get("Target", "?"),
        desc=" ".join(item.get("Description", []) or []) or "(no description)")


def merge_consecutive(turns):
    out = []
    for role, text, lab in turns:
        if out and out[-1][0] == role:
            pr, pt, pl = out[-1]
            out[-1] = (pr, pt + "\n" + text, pl if pl else pl)
            # 合并的 assistant 段保留第一条的标签（其余回合丢弃）
        else:
            out.append((role, text, lab))
    return out


def drop_leading_assistant(turns):
    while turns and turns[0][0] == "assistant":
        turns.pop(0)
    return turns


def build_samples(task, dial) -> list[dict]:
    turns = merge_consecutive(drop_leading_assistant(list(iter_turns(task, dial))))
    system = system_for(task, dial)
    samples = []
    history = []
    for role, text, lab in turns:
        if role == "user":
            history.append({"role": "user", "content": text})
            continue
        # assistant 回合 → 一条样本（目标 = 本回合标签+话语）
        # 2026-09-18 裁定：训练目标不再带 "assistant:" 前缀（前缀只是香草
        # 提示的格式要求，由推理侧 strip_role_prefix 处理；v2 实验证明其
        # 在标签条件化下只会带来格式噪声）
        label = lab or UNLABELED
        target = text
        samples.append({
            "messages": [{"role": "system", "content": system}] + history
                        + [{"role": "assistant",
                            "content": label + LABEL_SEP + target}],
            "label": label,
        })
        history.append({"role": "assistant", "content": target})
    return samples


def load_dialogues(task):
    if task == "craigslistbargain":
        d = json.loads((RAW / task / "train_parsed.json").read_text())
    else:
        d = json.loads((RAW / task / "train.json").read_text())
    return d if isinstance(d, list) else list(d.values())


def build_vanilla_samples(task, dial) -> list[dict]:
    """香草生成对话 → SFT 样本（agent 回合带补标标签；与真人语料同格式）。"""
    pre_n = dial.get("prefix_n", 0)
    turns = [{"role": t["role"], "text": t["text"],
              "label": t.get("label") if t["role"] == "assistant" else None}
             for t in dial["turns"][pre_n:]]
    # 前导 assistant 丢弃 + 同角色合并（与真人路径一致）
    while turns and turns[0]["role"] == "assistant":
        turns.pop(0)
    merged = []
    for t in turns:
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["text"] += "\n" + t["text"]
        else:
            merged.append(dict(t))
    system = SYSTEM_TMPL[task].format(
        situation=dial.get("situation") or "(unknown)") if task == "esconv" \
        else SYSTEM_TMPL[task]
    samples = []
    history = []
    for t in merged:
        if t["role"] == "user":
            history.append({"role": "user", "content": t["text"]})
            continue
        label = t.get("label") or UNLABELED
        target = t["text"]
        if task == "esconv" and target.startswith("assistant: "):
            # 香草生成的原始文本带前缀——剥离（训练目标为干净话语）
            target = target[len("assistant: "):]
        samples.append({
            "messages": [{"role": "system", "content": system}] + history
                        + [{"role": "assistant", "content": label + LABEL_SEP + target}],
            "label": label,
        })
        history.append({"role": "assistant", "content": target})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["esconv", "p4g", "craigslistbargain", "all"])
    ap.add_argument("--limit", type=int, default=0, help="调试用：只处理前 N 条对话")
    ap.add_argument("--vanilla", action="store_true",
                    help="把香草生成对话（data/sft/vanilla_dialogues）并入输出")
    args = ap.parse_args()
    tasks = (["esconv", "p4g", "craigslistbargain"] if args.task == "all"
             else [args.task])
    for task in tasks:
        d = load_dialogues(task)
        if args.limit:
            d = d[:args.limit]
        out_f = OUT / task / "train.jsonl"
        out_f.parent.mkdir(parents=True, exist_ok=True)
        n_samples = 0
        label_dist = Counter()
        unlabeled = 0
        with out_f.open("w") as f:
            for dial in d:
                for s in build_samples(task, dial):
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    n_samples += 1
                    if s["label"] == UNLABELED:
                        unlabeled += 1
                    else:
                        label_dist[s["label"]] += 1
        if args.vanilla:
            vdir = ROOT / "data" / "sft" / "vanilla_dialogues" / task
            vf = OUT / task / "vanilla.jsonl"
            n_v = 0
            with vf.open("w") as fv:
                for fp in sorted(vdir.glob("*.json")) if vdir.exists() else []:
                    for s in build_vanilla_samples(task, json.loads(fp.read_text())):
                        fv.write(json.dumps(s, ensure_ascii=False) + "\n")
                        n_v += 1
            print(f"{task}: 香草样本 {n_v} → {vf}")
        stats = {"task": task, "n_samples": n_samples,
                 "n_unlabeled": unlabeled,
                 "label_dist": dict(sorted(label_dist.items(),
                                           key=lambda x: -x[1]))}
        (OUT / task / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=1) + "\n")
        print(f"{task}: {n_samples} 样本（无标签占位 {unlabeled}）→ {out_f}")
        print("  标签分布:", dict(sorted(label_dist.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
