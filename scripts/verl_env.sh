# verl 运行环境导出（必须 source 后再跑 verl/vllm）
# 根因（2026-09-16）：flashinfer JIT 编译取 PATH 上的 nvcc——
# base 环境的 CUDA 12.4 不支持 --compress-mode=size，导致
# "nvcc fatal: Unknown option '--compress-mode=size'"。
# 修复：把 verl 环境自带的 CUDA 13 nvcc 目录置于 PATH 前。
ENV_NAME="${CSTPO_CONDA_ENV:-verl}"
ENV_ROOT="${CSTPO_ENV_ROOT:-/data/user21300120/.conda/envs/$ENV_NAME}"
export PATH="$ENV_ROOT/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH"
# flashinfer JIT 链接阶段用 c++ 直接链接，需要 cu13 的 lib 目录
#（否则 "cannot find -lcudart"——bin 目录只修 nvcc，不修链接器）
export LIBRARY_PATH="$ENV_ROOT/lib/python3.12/site-packages/nvidia/cu13/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CONDA_ENV="$ENV_NAME"
# 用法：
#   source scripts/verl_env.sh
#   conda run -n verl python -m verl.trainer.main_ppo --config-name ...
