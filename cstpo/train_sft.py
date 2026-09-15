"""SFT 训练脚本（accelerate 兜底版；verl 入口另行适配但必须复用
sft_tokenize 的掩码逻辑）。

- 数据装配：真人 train.jsonl 下采样至 4× 香草（8:2），乱序；
- 掩码：sft_tokenize（仅目标回合话语 tokens 计 loss，标签行全 mask）；
- 单卡 full fine-tune（配置见 cstpo/sft_config.py）。

用法：
    cd CSTPO && accelerate launch cstpo/train_sft.py --task esconv
    烟测：--smoke（100 步）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cstpo.sft_config import (BATCH_SIZE, BF16, DATA_DIR, EPOCHS,
                              GRADIENT_CHECKPOINTING, GRAD_ACCUM, LR,
                              MAX_SEQ_LEN, MIX_REAL_RATIO, MODEL_ID, OUT_DIR,
                              FINAL_OUT, SAVE_STEPS, SEED, WARMUP_RATIO,
                              WEIGHT_DECAY)
from cstpo.sft_tokenize import IGNORE, tokenize_sample


def load_mixed(task: str) -> list[dict]:
    """真人下采样至 4× 香草（8:2），合并乱序。"""
    real = [json.loads(l) for l in
            (DATA_DIR / task / "train.jsonl").open() if l.strip()]
    van_f = DATA_DIR / task / "vanilla.jsonl"
    van = [json.loads(l) for l in van_f.open() if l.strip()] if van_f.exists() else []
    rng = random.Random(SEED)
    if van:
        keep_real = int(len(van) * (MIX_REAL_RATIO / (1 - MIX_REAL_RATIO)))
        real = rng.sample(real, min(len(real), keep_real))
    data = real + van
    rng.shuffle(data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["esconv", "p4g", "craigslistbargain"])
    ap.add_argument("--smoke", action="store_true", help="烟测：数据取 200 条、跑 100 步")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    print(f"[{args.task}] 加载数据...", flush=True)
    data = load_mixed(args.task)
    if args.smoke:
        data = data[:200]
    print(f"[{args.task}] 样本 {len(data)} 条", flush=True)

    tk = AutoTokenizer.from_pretrained(MODEL_ID)
    if tk.pad_token_id is None:
        tk.pad_token_id = tk.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto" if BF16 else None)

    class _DS(list):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            return tokenize_sample(tk, data[i])

    def collate(batch):
        import torch
        ids = [b["input_ids"][:MAX_SEQ_LEN] for b in batch]
        labs = [b["labels"][:MAX_SEQ_LEN] for b in batch]
        L = min(MAX_SEQ_LEN, max(len(x) for x in ids))
        in_ids = torch.full((len(batch), L), tk.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), L), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(zip(ids, labs)):
            n = min(len(x), L)
            in_ids[i, :n] = torch.tensor(x[:n])
            labels[i, :n] = torch.tensor(y[:n])
            mask[i, :n] = 1
        return {"input_ids": in_ids, "labels": labels,
                "attention_mask": mask}

    train_args = TrainingArguments(
        output_dir=str(OUT_DIR / args.task),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        seed=SEED,
        max_steps=100 if args.smoke else args.max_steps,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=_DS(data),
                      data_collator=collate)
    trainer.train()
    (FINAL_OUT / args.task).mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_OUT / args.task))
    tk.save_pretrained(str(FINAL_OUT / args.task))
    print(f"[{args.task}] 完成 → {FINAL_OUT / args.task}")


if __name__ == "__main__":
    main()
