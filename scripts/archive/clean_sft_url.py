"""SFT 数据清洗：URL 占位符上下文感知处理（当前数据血缘，2026-09-18 冻结）。

背景：真人 P4G 原始数据把链接替换成了字面 "URL" 占位符（p4g 训练集 1093 条
2.7%），模型学会吐 "URL" 并在多 epoch 下滚雪球成 "URL URL url..." 退化。
规则（按语境）：
  1. 句子已有 "website" → 删除占位符（防冗余）
  2. 占位符在句首/感叹词后（无链接语义）→ 删除
  3. 其余（真实链接语义）→ 替换为 "their official website"
外加省略号串/多空格规范。产物 data/sft_v4/。

用法：python scripts/clean_sft_url.py（读 data/sft_v3 写 data/sft_v4）
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sft_v3"
DST = ROOT / "data" / "sft_v4"
URL = re.compile(r"\bURL\b", re.IGNORECASE)
REPL = "their official website"


def clean_rest(rest: str) -> str:
    if re.search(r"website", rest, re.IGNORECASE):
        c = URL.sub("", rest)
    elif re.match(r"^\s*(?:Absolutely|Yes|No|Well|Oh|Ah)[,.\s]*URL\b", rest,
                  re.IGNORECASE) or re.match(r"^\s*URL\b", rest):
        c = URL.sub("", rest)
    else:
        c = URL.sub(REPL, rest)
    c = re.sub(r"\s*\.{2,}\s*", " ", c)
    c = re.sub(r"  +", " ", c).strip()
    return c


def main():
    if DST.exists():
        shutil.rmtree(DST)
    for task in ("esconv", "p4g", "craigslistbargain"):
        (DST / task).mkdir(parents=True, exist_ok=True)
        for split in ("train", "vanilla"):
            f = SRC / task / f"{split}.jsonl"
            if not f.exists():
                continue
            with open(f) as fin, open(DST / task / f"{split}.jsonl", "w") as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    for m in r.get("messages") or []:
                        if m.get("role") == "assistant":
                            c = m["content"]
                            if "\n" in c:
                                label, rest = c.split("\n", 1)
                                cleaned = clean_rest(rest)
                                c = label + "\n" + cleaned if cleaned \
                                    else label + "\n(keep talking)"
                            else:
                                c = clean_rest(c)
                            m["content"] = c
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("sft_v4 完成")


if __name__ == "__main__":
    main()
