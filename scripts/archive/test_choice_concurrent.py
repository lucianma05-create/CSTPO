"""并发复现实验：4 个并发客户端 × 多轮 choice/自由段交替（模拟 RL 4 workers）。

RL 与 standalone 串行测试的最大差异 = 并发。若并发下出现垃圾输出，
则 vllm 0.29 structured outputs 的并发 bug 坐实。
"""
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data/user21300120/mmh/CSTPO")

from openai import OpenAI
from transformers import AutoTokenizer

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
MODEL = "/publicdata/model/Qwen3-8B"
LABELS = ["Question", "Restatement or Paraphrasing", "Reflection of feelings",
          "Self-disclosure", "Affirmation and Reassurance",
          "Providing Suggestions", "Information", "Others"]


def one_client(worker_id, rounds, results):
    tk = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "system", "content": "You are a helpful and caring friend."},
            {"role": "user", "content": f"client {worker_id}: I am feeling sad."}]
    text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    all_ids = tk.encode(text, add_special_tokens=False)
    nl = tk.encode("\n", add_special_tokens=False)
    bad = 0
    for turn in range(rounds):
        # 标签段
        r1 = client.completions.create(
            model=MODEL, prompt=all_ids, max_tokens=8, temperature=0.2,
            extra_body={"structured_outputs": {"choice": LABELS}})
        label = r1.choices[0].text.strip()
        if label not in LABELS:
            bad += 1
            results.append(("label", worker_id, turn, label[:60]))
        all_ids = all_ids + tk.encode(label, add_special_tokens=False)
        # 话语段
        r2 = client.completions.create(
            model=MODEL, prompt=all_ids + nl, max_tokens=64, temperature=0.2)
        utter = r2.choices[0].text
        all_ids = all_ids + nl + tk.encode(utter, add_special_tokens=False)
        # 简单垃圾检测：非英文字符占比
        non_ascii = sum(1 for c in utter if ord(c) > 127 and c not in "’“”")
        if len(utter) > 10 and non_ascii / len(utter) > 0.2:
            bad += 1
            results.append(("utter", worker_id, turn, utter[:60]))
    results.append(("done", worker_id, bad, ""))


def main():
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(one_client, w, 6, results) for w in range(4)]
        for f in futs:
            f.result()
    issues = [r for r in results if r[0] in ("label", "utter")]
    print(f"问题数: {len(issues)} / 4 clients × 6 rounds")
    for kind, w, t, txt in issues[:8]:
        print(f"  [{kind}] worker={w} turn={t}: {txt!r}")
    for r in results:
        if r[0] == "done":
            print(f"  worker {r[1]}: {r[2]} 个问题")


if __name__ == "__main__":
    main()
