#!/usr/bin/env bash
# MC-PPO + 认知 critic（PPO 形态）：GAE + value head，轮级 credit。
# 设计（CRITIC_DESIGN.md）：token 级价值网络 + GAE；标签优势 = 标签位置的
# GAE 值（≈该标签选择的边际价值）；γ=0.99 防早轮信用消失；critic 基座 =
# SFT merged 权重 + LoRA r64；critic_warmup 先冻结 actor 训值函数。
# 资源：两卡布局 + critic 分池（verl/main_ppo.py 补丁）。n_gpus_per_node=1
# → global_pool 1 卡（actor 单卡 + vllm 同卡，v10 同款配置实测峰值
# 32GB allocated），critic_pool 1 卡（critic 单卡）。默认 global_pool
# 共置会把 actor+critic+vllm 三份模型挤在同一张卡，必 OOM。
set -euo pipefail
cd "$(dirname "$0")/.."

export CSTPO_CONDA_ENV="${ENV_NAME:-verl}"
source "$(dirname "$0")/verl_env.sh"

GPU_INDEX="${GPU_INDEX:-1,3}"
LOG_NAME="${LOG_NAME:-mc_ppo_critic}"
EXP_NAME="${EXP_NAME:-mcppo_esconv_critic_v1}"
DUMP_DIR="${DUMP_DIR:-logs/sync_dump_critic}"
TRAJ_DIR="${TRAJ_DIR:-logs/rollout_traj_critic}"
MODEL_PATH="${MODEL_PATH:-/publicdata/model/CSTPO_v3/esconv}"
STEPS="${STEPS:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
GAMMA="${GAMMA:-0.99}"
LAM="${LAM:-0.95}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
CRITIC_WARMUP="${CRITIC_WARMUP:-5}"
LABEL_ADV_WEIGHT="${LABEL_ADV_WEIGHT:-1}"   # 默认关：先看纯 critic 的标签迁移
CRITIC_DUMP_DIR="${CRITIC_DUMP_DIR:-$DUMP_DIR}"   # critic 权重落盘（warm-start 续跑用）
VLLM_UTIL="${VLLM_UTIL:-0.25}"   # 共享卡上调低（同卡其他用户任务会游走）
TRAIN_BATCH="${TRAIN_BATCH:-4}"
MINI_BATCH="${MINI_BATCH:-$TRAIN_BATCH}"   # verl 校验：mini_batch ≤ train_batch_size
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"   # prompt 1024 + response 4096 ≈ 5120，8192 封顶 KV
ACTOR_LR="${ACTOR_LR:-1e-5}"   # v1c 续跑：3e-5（20 步无行为迁移 → 提步长）
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"   # v1c 续跑：v1b 落盘 adapter 作 warm-start

export PYTHONPATH="/data/user21300120/mmh/CSTPO:/data/user21300120/mmh/CSTPO/Cog-Sim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$DUMP_DIR" "$TRAJ_DIR"
export CSTPO_SYNC_DUMP_DIR="$DUMP_DIR"
export CSTPO_RECV_DUMP_DIR="$DUMP_DIR"
export CSTPO_TRAJ_DIR="$TRAJ_DIR"
export CSTPO_LABEL_ADV_WEIGHT="$LABEL_ADV_WEIGHT"

nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONUNBUFFERED=1 \
  CSTPO_SYNC_DUMP_DIR="$DUMP_DIR" CSTPO_RECV_DUMP_DIR="$DUMP_DIR" \
  CSTPO_TRAJ_DIR="$TRAJ_DIR" CSTPO_LABEL_ADV_WEIGHT="$LABEL_ADV_WEIGHT" \
  CSTPO_CRITIC_DUMP_DIR="$CRITIC_DUMP_DIR" \
  conda run --no-capture-output -n "$ENV_NAME" python -m verl.trainer.main_ppo \
  --config-name ppo_trainer \
  algorithm.adv_estimator=gae \
  algorithm.gamma="$GAMMA" \
  algorithm.lam="$LAM" \
  data.train_files=data/mcppo_esconv/train.parquet \
  data.val_files=data/mcppo_esconv/val.parquet \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  data.train_batch_size="$TRAIN_BATCH" \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  ${LORA_ADAPTER_PATH:+actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH"} \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
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
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_UTIL" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=false \
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
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=false \
  critic.model.enable_gradient_checkpointing=true \
  critic.model.lora_rank=64 \
  critic.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] \
  +critic.model.override_config.attn_implementation=sdpa \
  critic.fsdp.model_dtype=bf16 \
  critic.optim.lr="$CRITIC_LR" \
  critic.ppo_mini_batch_size="$MINI_BATCH" \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.critic_warmup="$CRITIC_WARMUP" \
  trainer.logger=['console'] \
  trainer.project_name=cstpo_mcppo \
  trainer.experiment_name="$EXP_NAME" \
  custom_reward_function.path=cstpo/rl/verl_cstpo_reward.py \
  custom_reward_function.name=compute_score \
  > "logs/${LOG_NAME}.log" 2>&1 &
echo "CRITIC-RUN PID: $!"
