"""SFT 数据清洗：剥离 esconv assistant 消息的 "assistant:"/"user:" 前导前缀。

发现（2026-09-18 交叉分析，消息级）：
  - esconv/train：no_label+pref 69.1%、labelBAD+utt_clean 15.9%、labelOK+utt_pref 15.0%
  - esconv/vanilla：no_label+pref 80.1%、labelOK+utt_pref 17.6%
即 ~85%/98% 的话语带 role 前缀；仅 ~15%/18% 位置有合法标签。

清洗规则：只剥前缀（无标签消息从内容头剥；有标签消息剥标签后的
话语头），标签行不动（labelBAD 仅统计不修改）。产物 data/sft_clean/，
原数据不动。
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLE_RE = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)


def clean_assistant(content: str) -> tuple[str, bool]:
    # 首行即前缀：整个内容是话语（无标签或多行原始轮），从头剥
    if ROLE_RE.match(content):
        return ROLE_RE.sub("", content, count=1), True
    if "\n" in content:
        label, rest = content.split("\n", 1)
        stripped = ROLE_RE.sub("", rest, count=1)
        if stripped != rest:
            return f"{label}\n{stripped}", True
    return content, False


def main():
    for split in ("train", "vanilla"):
        src = ROOT / "data" / "sft" / "esconv" / f"{split}.jsonl"
        if not src.exists():
            continue
        dst_dir = ROOT / "data" / "sft_clean" / "esconv"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{split}.jsonl"
        n = cleaned = 0
        with open(src) as f, open(dst, "w") as out:
            for line in f:
                r = json.loads(line)
                n += 1
                for m in r.get("messages") or []:
                    if m.get("role") == "assistant":
                        m["content"], changed = clean_assistant(m["content"])
                        if changed:
                            cleaned += 1
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"esconv/{split}: {n} 样本, 清洗 {cleaned} 条助手消息 → {dst}")


if __name__ == "__main__":
    main()
