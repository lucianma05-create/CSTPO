#!/usr/bin/env bash
# MC-PPO 启动脚本（esconv、GRPO、12 轮交互、终局 judge 奖励）
# 前置：source scripts/verl_env.sh
# 显存账（GPU 1 单卡）：actor bf16+LoRA ~19G、vllm 权重 16G + KV
#   （utilization 0.40）；总计峰值 ~68G / 79G。
# 权重来源：HF-merge 双存（/publicdata/model/CSTPO/esconv）——初始 adapter 加载
#   路径是 token soup 乱码嫌疑（判别实验已证），弃用 lora_adapter_path；
#   LoRA r64 在 merged 基座上全新初始化，lora.merge=true 只 merge 训练增量，
#   同时绕开 vllm 原生 LoRA serving（vllm 0.28 与 Qwen3 fused QKV 不兼容：
#   'QKVParallelLinear has no attribute base_layer'，merge=false 会崩引擎）。
set -euo pipefail
cd "$(dirname "$0")/.."

# flashinfer JIT 环境（cu13 nvcc + -lcudart 链接）：必须 source——ray 的
# runtime_env 白名单会透传 driver 环境的 PATH/LIBRARY_PATH（constants_ppo.py
# patch），但 driver 环境本身没有的话一切无从谈起
export CSTPO_CONDA_ENV="${ENV_NAME:-verl}"
source "$(dirname "$0")/verl_env.sh"

# 并行实验参数化（默认=正式轮：20 steps、val 20 种子、free_cache 关闭）
GPU_INDEX="${GPU_INDEX:-1}"
LOG_NAME="${LOG_NAME:-mc_ppo}"
EXP_NAME="${EXP_NAME:-mcppo_esconv_v7_formal}"
DUMP_DIR="${DUMP_DIR:-logs/sync_dump}"
TRAJ_DIR="${TRAJ_DIR:-logs/rollout_traj}"
FREE_CACHE="${FREE_CACHE:-false}"
ENV_NAME="${ENV_NAME:-verl}"
STEPS="${STEPS:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
VLLM_UTIL="${VLLM_UTIL:-0.40}"
MAX_RESPONSE="${MAX_RESPONSE:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MODEL_PATH="${MODEL_PATH:-/publicdata/model/CSTPO/esconv}"
TRAIN_FILE="${TRAIN_FILE:-data/mcppo_esconv/train.parquet}"
VAL_FILE="${VAL_FILE:-data/mcppo_esconv/val.parquet}"

export PYTHONPATH="/data/user21300120/mmh/CSTPO:/data/user21300120/mmh/CSTPO/Cog-Sim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# 诊断：每次权重同步时把 verl 发送的 merged dict 与原始 adapter 落盘
# （transformer_impl.py patch 读取此变量；save_freq 弃用——merge=true 下
# verl 的 save_checkpoint 路径有 FSDP 断言 bug，改由同步时落盘替代）
mkdir -p "$DUMP_DIR" "$TRAJ_DIR"
export CSTPO_SYNC_DUMP_DIR="$DUMP_DIR"
export CSTPO_RECV_DUMP_DIR="$DUMP_DIR"
export CSTPO_TRAJ_DIR="$TRAJ_DIR"

nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONUNBUFFERED=1 \
  CSTPO_SYNC_DUMP_DIR="$DUMP_DIR" CSTPO_RECV_DUMP_DIR="$DUMP_DIR" \
  CSTPO_TRAJ_DIR="$TRAJ_DIR" \
  conda run --no-capture-output -n "$ENV_NAME" python -m verl.trainer.main_ppo \
  --config-name ppo_trainer \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  data.train_batch_size=8 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  data.max_prompt_length=1024 \
  data.max_response_length="$MAX_RESPONSE" \
  data.dataloader_num_workers=2 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora.merge=true \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_UTIL" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine="$FREE_CACHE" \
  actor_rollout_ref.rollout.temperature=0.2 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=cstpo/configs/cstpo_agent_loop.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=cstpo_agent \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=12 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=12 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.logger=['console'] \
  trainer.project_name=cstpo_mcppo \
  trainer.experiment_name="$EXP_NAME" \
  custom_reward_function.path=cstpo/rl/verl_cstpo_reward.py \
  custom_reward_function.name=compute_score \
  ${EXTRA_ARGS:-} \
  > "logs/${LOG_NAME}.log" 2>&1 &
echo "MC-PPO PID: $!"
