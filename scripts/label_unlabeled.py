"""无标签目标回合的 flash 补标（2026-09-18 裁定 E）。

与源标注方案语义对齐：固定词表、无 null 出口，catch-all（Others/other）
是唯一兜底——补标后三任务 100% 真实标签，<unlabeled> 从数据消失。
JSON 解析失败重试 2 次；仍失败归 catch-all（等价源标注者必须选一个）。

用法：
  python scripts/label_unlabeled.py --pilot 100        # 先导：只统计不写文件
  python scripts/label_unlabeled.py --task p4g --task craigslistbargain
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "Cog-Sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cstpo.core.label_maps import CB_CANONICAL, ESCONV_CANONICAL, P4G_CANONICAL  # noqa: E402
from simulator.llm import LLMClient  # noqa: E402

DATA_DIR = Path(__import__("os").environ.get("SFT_DATA_DIR", str(ROOT / "data" / "sft_v2")))
LABEL_SETS = {"esconv": ESCONV_CANONICAL, "p4g": P4G_CANONICAL,
              "craigslistbargain": CB_CANONICAL}
CATCHALL = {"esconv": "Others", "p4g": "other", "craigslistbargain": "other"}

LABELER_SYSTEM = """You annotate ONE strategy label for ONE assistant utterance in a
dialogue. You may ONLY read the conversation history UP TO AND INCLUDING the
assistant utterance being labeled (nothing after it). Choose the single most
appropriate label from the given set. If the utterance does not fit any specific
label, choose the catch-all label as the strategy. Return exactly one JSON
object: {"label": "..."}"""


def label_one(task, sample, client) -> tuple[str, int]:
    """返回 (label, 尝试次数)。限流/网络错误指数退避重试（高并发下必然发生）。"""
    import time
    labels = LABEL_SETS[task]
    catch = CATCHALL[task]
    history = [m for m in sample["messages"][:-1]]
    target = sample["messages"][-1]["content"]
    utter = target.split("\n", 1)[1] if "\n" in target else target
    ctx = "\n".join(f"{m['role']}: {m['content']}" for m in history[-12:])
    prompt = (f"Task: {task}\nValid labels: {labels}\nCatch-all label: {catch}\n\n"
              f"Conversation so far:\n{ctx}\n\nAssistant utterance to label:\n{utter}")
    for attempt in range(6):
        try:
            out = client.chat_json(
                [{"role": "system", "content": LABELER_SYSTEM},
                 {"role": "user", "content": prompt}], max_tok=100)
            lab = out.get("label")
            if lab in labels:
                return lab, attempt + 1
            return catch, attempt + 1
        except Exception:
            time.sleep(min(0.3 * (2 ** attempt), 5.0))
    return catch, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", choices=list(LABEL_SETS))
    ap.add_argument("--pilot", type=int, default=0, help="先导：每任务只跑 N 条，统计不写文件")
    ap.add_argument("--workers", type=int, default=1000)
    args = ap.parse_args()
    tasks = args.task or ["p4g", "craigslistbargain"]

    for task in tasks:
        src = DATA_DIR / task / "train.jsonl"
        samples = [json.loads(l) for l in src.open() if l.strip()]
        unlabeled = [s for s in samples if s["label"] == "<unlabeled>"]
        if args.pilot:
            unlabeled = unlabeled[:args.pilot]
        print(f"\n[{task}] 待标注 {len(unlabeled)}/{len(samples)} 条，"
              f"并发 {args.workers}...", flush=True)

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(label_one, task, s, LLMClient()) for s in unlabeled]
            for fut in futs:
                results.append(fut.result())
        dist = {}
        attempts = 0
        for s, (lab, n_att) in zip(unlabeled, results):
            dist[lab] = dist.get(lab, 0) + 1
            attempts += n_att
        retry_rate = (attempts - len(results)) / max(len(results), 1)
        print(f"[{task}] 标签分布: {dict(sorted(dist.items(), key=lambda x: -x[1]))}",
              flush=True)
        print(f"[{task}] 平均尝试 {attempts/len(results):.2f} 次（重试率 {retry_rate:.1%}）",
              flush=True)

        if not args.pilot:
            for s, (lab, _) in zip(unlabeled, results):
                s["label"] = lab
                # 同步内容标签行（tokenize 断言要求 content.startswith(label+LABEL_SEP)）
                last = s["messages"][-1]
                utter = last["content"].split("\n", 1)[1] if "\n" in last["content"] \
                    else last["content"]
                last["content"] = lab + "\n" + utter
            tmp = src.with_suffix(".tmp")
            with tmp.open("w") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            tmp.replace(src)
            print(f"[{task}] 已写回 {src}（{len(unlabeled)} 条补标）", flush=True)
    print("\n完成", flush=True)


if __name__ == "__main__":
    main()
