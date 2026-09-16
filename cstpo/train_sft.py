"""SFT 训练脚本（accelerate 兜底版；verl 入口另行适配但必须复用
sft_tokenize 的掩码逻辑）。

- 数据装配：真人 train.jsonl 下采样至 4× 香草（8:2），乱序；
- 掩码：sft_tokenize（仅目标回合话语 tokens 计 loss，标签行全 mask）；
- 单卡 full fine-tune（配置见 cstpo/sft_config.py）。

用法（单卡直接运行，勿用 accelerate launch——其默认配置会覆盖
CUDA_VISIBLE_DEVICES 导致跑错卡）：
    cd CSTPO && CUDA_VISIBLE_DEVICES=0 python3 cstpo/train_sft.py --task esconv --lora
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
                              LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET,
                              MAX_SEQ_LEN, MIX_REAL_RATIO, MODEL_ID, OUT_DIR,
                              FINAL_OUT, SAVE_STEPS, SEED, WARMUP_RATIO,
                              WEIGHT_DECAY)
from cstpo.sft_tokenize import IGNORE, left_truncate, tokenize_sample


def load_mixed(task: str, real_ratio: float = MIX_REAL_RATIO) -> list[dict]:
    """真人按对话分层下采样（防长对话主导训练分布），与香草合并乱序。"""
    real = [json.loads(l) for l in
            (DATA_DIR / task / "train.jsonl").open() if l.strip()]
    van_f = DATA_DIR / task / "vanilla.jsonl"
    van = [json.loads(l) for l in van_f.open() if l.strip()] if van_f.exists() else []
    rng = random.Random(SEED)
    if van:
        keep_real = int(len(van) * (real_ratio / (1 - real_ratio)))
        if keep_real < len(real):
            # 按对话分层：samples 无对话 id 字段，用消息数近似分层——
            # 按历史长度分桶，每桶按比例采样，避免长对话垄断
            buckets = {}
            for s_ in real:
                n = len(s_["messages"])
                buckets.setdefault(min(n // 8, 40), []).append(s_)
            real = []
            for k in sorted(buckets):
                b = buckets[k]
                real += rng.sample(b, min(len(b),
                                         max(1, keep_real // len(buckets))))
            real = rng.sample(real, min(len(real), keep_real))
    data = real + van
    rng.shuffle(data)
    return data


def pick_free_a100() -> str:
    """nvidia-smi 找空闲最大的 A100，返回其 UUID。

    CUDA_VISIBLE_DEVICES 用 UUID 指定设备（CUDA 设备序与 nvidia-smi 序
    不一致——本机 CUDA:4 是 A400、nvidia-smi:2 是 A400——索引映射会选错卡，
    UUID 不受序映射影响）。
    """
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free,uuid",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    best, best_free = None, -1
    for line in out.strip().splitlines():
        name, free, uuid = [x.strip() for x in line.split(",")]
        if "A100" not in name:
            continue
        free_mib = int(free.split()[0])
        if free_mib > best_free:
            best, best_free = uuid, free_mib
    return best


def main():
    # 在 torch 导入前固定可见设备：单卡模式，避免 Trainer 的 DataParallel
    # 把模型复写到 A400 等小卡导致 OOM（CUDA 设备序与 nvidia-smi 不一致，
    # 外部 CUDA_VISIBLE_DEVICES 不可靠）
    import os
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu = pick_free_a100()
        if gpu is None:
            raise RuntimeError("无可用 A100")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"[自动选卡] {gpu[:8]}…（空闲 {0}MiB）", flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["esconv", "p4g", "craigslistbargain"])
    ap.add_argument("--smoke", action="store_true", help="烟测：数据 100 条、跑 20 步")
    ap.add_argument("--lora", action="store_true", help="LoRA 模式（默认 full FT）")
    ap.add_argument("--mix-real", type=float, default=MIX_REAL_RATIO,
                    help="真人占比（默认 0.8；统一口径 5:5 用 0.5）")
    ap.add_argument("--epochs", type=float, default=None,
                    help="覆盖训练轮数（默认配置 2）")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    print(f"[{args.task}] 加载数据...", flush=True)
    data = load_mixed(args.task, real_ratio=args.mix_real)
    if args.smoke:
        data = data[:100]
    print(f"[{args.task}] 样本 {len(data)} 条", flush=True)

    tk = AutoTokenizer.from_pretrained(MODEL_ID)
    if tk.pad_token_id is None:
        tk.pad_token_id = tk.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto" if BF16 else None)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    # 显式开启梯度检查点（TrainingArguments 的开关对 peft 模型不生效，
    # 未开时 batch2×4096 激活内存 ~45G，无法多卡并行）
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    class _DS(list):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            return tokenize_sample(tk, data[i])

    def collate(batch):
        import torch
        tr = [left_truncate(b, MAX_SEQ_LEN) for b in batch]
        ids = [b["input_ids"] for b in tr]
        labs = [b["labels"] for b in tr]
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
        num_train_epochs=args.epochs if args.epochs is not None else EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        seed=SEED,
        max_steps=20 if args.smoke else args.max_steps,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=_DS(data),
                      data_collator=collate)
    trainer.train()
    out_dir = FINAL_OUT / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.lora:
        # adapter 单独存（RL 阶段挂载）+ merge 完整权重（A/B 推理用）
        model.save_pretrained(str(out_dir / "adapter"))
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out_dir), safe_serialization=True)
    else:
        trainer.save_model(str(out_dir))
    tk.save_pretrained(str(out_dir))
    print(f"[{args.task}] 完成 → {out_dir}")


if __name__ == "__main__":
    main()
