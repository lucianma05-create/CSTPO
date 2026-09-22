"""HF merge 权重 + vllm 的 10 轮 × 4 并发协议测试（乱码复现关键实验）。"""
import sys
import concurrent.futures

sys.path.insert(0, "/data/user21300120/mmh/CSTPO")
from openai import OpenAI
from transformers import AutoTokenizer

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="x")
MODEL = "/tmp/qwen3_esconv_sft_merged"
LABELS = ["Question", "Restatement or Paraphrasing", "Reflection of feelings",
          "Self-disclosure", "Affirmation and Reassurance",
          "Providing Suggestions", "Information", "Others"]
tk = AutoTokenizer.from_pretrained("/publicdata/model/Qwen3-8B")


def one(wid, results):
    msgs = [{"role": "system", "content": "You are a helpful and caring friend."},
            {"role": "user", "content": f"client {wid}: I am feeling sad. My boyfriend left me."}]
    text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    all_ids = tk.encode(text, add_special_tokens=False)
    nl = tk.encode("\n", add_special_tokens=False)
    bad = 0
    for turn in range(10):
        r1 = client.completions.create(
            model=MODEL, prompt=all_ids, max_tokens=8, temperature=0.2,
            extra_body={"structured_outputs": {"choice": LABELS}})
        label = r1.choices[0].text.strip()
        if label not in LABELS:
            bad += 1
            results.append(("label", wid, turn, label[:50]))
        all_ids += tk.encode(label, add_special_tokens=False)
        r2 = client.completions.create(
            model=MODEL, prompt=all_ids + nl, max_tokens=96, temperature=0.2)
        utter = r2.choices[0].text
        non_ascii = sum(1 for c in utter if ord(c) > 127 and c not in "’“”")
        if len(utter) > 10 and non_ascii / len(utter) > 0.2:
            bad += 1
            results.append(("utter", wid, turn, utter[:60]))
        all_ids += nl + tk.encode(utter, add_special_tokens=False)
        add = tk.apply_chat_template(
            [{"role": "assistant", "content": utter},
             {"role": "user", "content": f"user reply {turn}: I still feel really down."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        all_ids += tk.encode(add, add_special_tokens=False)
    results.append(("done", wid, bad))


def main():
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda w: one(w, results), range(4)))
    issues = [r for r in results if r[0] in ("label", "utter")]
    print(f"merge权重+vllm 10轮×4clients: 问题 {len(issues)} 个", flush=True)
    for r in issues[:6]:
        print(" ", r, flush=True)
    for r in results:
        if r[0] == "done":
            print(f"  worker {r[1]}: {r[2]} 问题", flush=True)


if __name__ == "__main__":
    main()
