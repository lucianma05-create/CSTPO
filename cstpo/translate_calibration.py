"""校准对话中文翻译（flash，逐对话一次结构化调用）。

为每条校准对话生成中文译文（逐轮对齐），供人工标注前端使用
（04§19 标注协议：标注者看中文、可对照英文原文）。

用法：cd CSTPO && python -m cstpo.translate_calibration --workers 60
产物：data/judge_calibration/annotations/{split}/{task}/{did}.json
      {dialogue_id, turns_zh: [...], annotations: {}}
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient, StructuredCallError

OUT = ROOT / "data" / "judge_calibration"
ANNO = OUT / "annotations"

SYSTEM = """Translate each line of this dialogue into Simplified Chinese.
Keep the turn order and alignment EXACTLY (same number of lines, one per turn).
Do not merge or split turns. Keep names/amounts/units as in the original
(e.g., $215, 8 a.m.). Natural spoken Chinese, not literal word-for-word.
Return exactly one JSON object: {"turns": ["第一行译文", "第二行译文", ...]}"""


def translate_one(args):
    did, turns, llm = args
    src = "\n".join(f"{i}: {t['text']}" for i, t in enumerate(turns))
    try:
        out = llm.chat_json(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": src}],
            max_tok=6000)
    except StructuredCallError:
        return did, None
    zh = out.get("turns", [])
    if not isinstance(zh, list) or len(zh) != len(turns):
        return did, None
    return did, [str(x) for x in zh]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=60)
    args = ap.parse_args()

    jobs = []
    for fp in OUT.rglob("*.json"):
        if fp.name == "smoke_summary.json" or "annotations" in fp.parts:
            continue
        d = json.loads(fp.read_text())
        rel = fp.relative_to(OUT)
        split, task = rel.parts[0], rel.parts[1]
        # 跳过已有完整译文的（只补缺失/重生成的）
        ap = ANNO / split / task / f"{d['dialogue_id']}.json"
        if ap.exists():
            zh = json.loads(ap.read_text()).get("turns_zh", [])
            if len(zh) == len(d["turns"]):
                continue
        jobs.append((d["dialogue_id"], d["turns"], split, task))
    print(f"翻译 {len(jobs)} 条对话（workers={args.workers}）...", flush=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(translate_one, (did, turns, LLMClient())): (did, split, task)
                for did, turns, split, task in jobs}
        for k, fut in enumerate(as_completed(futs)):
            did, zh = fut.result()
            split, task = futs[fut][1], futs[fut][2]
            if zh is None:
                fail += 1
                print(f"  [失败] {did}", flush=True)
                continue
            d = ANNO / split / task
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{did}.json").write_text(
                json.dumps({"dialogue_id": did, "turns_zh": zh,
                            "annotations": {}},
                           ensure_ascii=False, indent=1) + "\n")
            ok += 1
            if (ok + fail) % 20 == 0:
                print(f"  {ok + fail}/{len(jobs)}（失败 {fail}）", flush=True)
    print(f"完成：成功 {ok} / 失败 {fail} → {ANNO}")


if __name__ == "__main__":
    main()
