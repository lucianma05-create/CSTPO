"""SFT 训练配置（SFT_SPEC_20260915 的单一来源；训练/掩码/推理共用）。

任何改动先改 SFT_SPEC 与本文件，再同步训练脚本。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- 模型与数据 ----
MODEL_ID = "Qwen/Qwen3-8B"
DATA_DIR = ROOT / "data" / "sft"
TASKS = ("esconv", "p4g", "craigslistbargain")
MAX_SEQ_LEN = 4096          # SPEC §5：实测 P99×1.5 均低于 4096
MIX_REAL_RATIO = 0.8        # 真人:香草 = 8:2（装配时真人下采样到 4× 香草）
UNLABELED = "<unlabeled>"
LABEL_SEP = "\n"            # 与 build_sft/agent 推理逐字节一致

# ---- 训练超参 ----
# 单卡 A100-80G，full fine-tune（RL 初始化需完整权重；LoRA 另行裁定）
LR = 2e-5
EPOCHS = 2
BATCH_SIZE = 2              # 每卡 micro batch
GRAD_ACCUM = 8              # 等效 batch 16
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
BF16 = True
GRADIENT_CHECKPOINTING = True
SEED = 20260916
SAVE_STEPS = 500
EVAL_STEPS = 500

# ---- loss 掩码口径（SPEC 2.2 话语 SFT）----
# 计算 loss 的 token：仅最后一条 assistant 消息的"话语部分"；
# 掩掉：system/user 全部、历史 assistant 全部、目标回合的标签行前缀、
# 模板结构标记（<|im_start|>/<|im_end|>）。
MASK_LABEL_LINE = True

# ---- 输出 ----
OUT_DIR = ROOT / "outputs" / "sft"            # 不入库（.gitignore /outputs/）
FINAL_OUT = ROOT / "checkpoints" / "sft"      # 不入库（.gitignore /checkpoints/）
