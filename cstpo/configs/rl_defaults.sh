# CSTPO RL 超参数统一管理（被 scripts/mc_ppo_critic_run.sh source）。
# 覆盖方式：环境变量优先（脚本内已用 ${VAR:-default}），临时实验直接
#   VAR=xxx bash scripts/mc_ppo_critic_run.sh
# 任务差异参数（按任务覆盖）：
GPU_INDEX="${GPU_INDEX:-1,3}"
LOG_NAME="${LOG_NAME:-mc_ppo_critic}"
EXP_NAME="${EXP_NAME:-mcppo_esconv_critic_v1}"
DUMP_DIR="${DUMP_DIR:-logs/sync_dump_critic}"
TRAJ_DIR="${TRAJ_DIR:-logs/rollout_traj_critic}"
MODEL_PATH="${MODEL_PATH:-/publicdata/model/CSTPO_v3/esconv}"
STEPS="${STEPS:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
GAMMA="${GAMMA:-1}"   # DETAILS §5.2：折扣因子 1
LAM="${LAM:-0.95}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
CRITIC_WARMUP="${CRITIC_WARMUP:-5}"
LABEL_ADV_WEIGHT="${LABEL_ADV_WEIGHT:-1}"   # 默认关：先看纯 critic 的标签迁移
CRITIC_DUMP_DIR="${CRITIC_DUMP_DIR:-$DUMP_DIR}"   # critic 权重落盘（warm-start 续跑用）
# 树状分叉（DETAILS §5.2）：主干 1 条 + nodes 节点 × n=2 续演 → n = 1+2×nodes。
# esconv 4 节点 → 9；cb/p4g 3 节点 → 7（按任务覆盖）
VLLM_UTIL="${VLLM_UTIL:-0.25}"
ROLLOUT_N="${ROLLOUT_N:-9}"   # esconv=9；cb/p4g 用 7
# agent workers 数必须整除 TRAIN_BATCH×ROLLOUT_N（verl 等分 chunk 断言）：
# 正式口径 10×9=90 与 10×7=70 的最小公共整除 worker 数为 5；smoke 用 3（2×9=18）
NUM_WORKERS="${NUM_WORKERS:-5}"
TRAIN_BATCH="${TRAIN_BATCH:-10}"   # DETAILS §5.2：每步 10 个种子
MINI_BATCH="${MINI_BATCH:-$TRAIN_BATCH}"   # verl 校验：mini_batch ≤ train_batch_size
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"   # 30 回合口径：prompt 1024 + response 16k
MAX_TURNS="${MAX_TURNS:-30}"   # DETAILS §5.2：30 回合；smoke 用 5
MAX_RESPONSE="${MAX_RESPONSE:-16384}"
ACTOR_LR="${ACTOR_LR:-1e-5}"   # DETAILS §5.2：actor 1e-5
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"   # v1c 续跑：v1b 落盘 adapter 作 warm-start

