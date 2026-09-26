# verl 0.8 补丁清单（训练复现必读）

双头 critic RL 依赖对 `verl 0.8`（site-packages）的一组本地补丁。这些补丁**不在本仓库内**，
换机器复现训练前必须按本清单重新打入。所有补丁均以 `CSTPO` 注释标记，可
`grep -n "CSTPO" <file>` 定位。

## 补丁文件与内容

| 文件（site-packages/verl/…） | 补丁内容 |
|---|---|
| `utils/model.py` | `ValueHeadWrapper` 增加 `v_label`/`v_utter` 双头（CSTPO_DUAL_HEAD=1 时 forward 只出双头值、跳过 v_head）；`__getattr__` 先查 `_modules` 注册表（修复 peft Qwen3Model 崩溃） |
| `trainer/ppo/core_algos.py` | `compute_gae_advantage_return_dual`：标签链 + 话语链两层次 GAE，边界交接 δ_l = γ·V_utter(next) − V_label；critic 用 raw returns，actor 优势按流白化并 clamp ±10 |
| `trainer/ppo/ray_trainer.py` | 双头 dispatch（CSTPO_DUAL_HEAD=1 且 label_offsets 可用时走 dual GAE，写 label_mask/returns_label/returns_utter）；标签优势加权补丁（CSTPO_LABEL_ADV_WEIGHT，白化后乘）；健康探针落盘（CSTPO_HEALTH_FILE） |
| `workers/engine/fsdp/transformer_impl.py` | 标签行 choice 掩码重归一化（约束内 ratio）；双头 values 输出（values_utter/values_label dict）；critic 权重落盘与健康探针（CSTPO_CRITIC_DUMP_DIR） |
| `workers/utils/losses.py` | 双头 value loss：两 head 各在 utter_mask/label_mask 上回归；NaN 溯源探针（CSTPO_NAN_DEBUG） |
| `trainer/main_ppo.py` | critic 独立资源池（Role.Critic → "critic_pool"），避免 actor+critic+vllm 三模型挤一卡 OOM |
| `trainer/constants_ppo.py` | critic_pool 资源注册 |
| `workers/rollout/vllm_rollout/vllm_async_server.py` | 多轮生成每轮 max_tokens cap（首轮不吃光全部预算） |
| `workers/rollout/vllm_rollout/utils.py` | agent loop 采样参数透传 |
| `experimental/agent_loop/agent_loop.py` | 采样参数（repetition_penalty 等）透传给 CSTPO agent loop |
| `utils/vllm/vllm_fp8_utils.py` | FP8 路径兼容（flashinfer 不可用时的降级） |
| `models/transformers/monkey_patch.py` | Qwen3 生成兼容 |

## 环境变量开关

| 变量 | 作用 |
|---|---|
| `CSTPO_DUAL_HEAD=1` | 启用双头 critic 路径 |
| `CSTPO_LABEL_ADV_WEIGHT` | 标签优势加权倍数（1=关） |
| `CSTPO_HEALTH_FILE` | 健康探针落盘路径 |
| `CSTPO_CRITIC_DUMP_DIR` | critic 权重落盘目录 |
| `CSTPO_SYNC_DUMP_DIR` / `CSTPO_RECV_DUMP_DIR` | actor 权重同步落盘 |
| `CSTPO_TRAJ_DIR` | rollout 轨迹与分叉快照目录 |
| `CSTPO_NAN_DEBUG=1` | loss NaN 溯源探针 |

## 复现步骤

1. `pip install verl==0.8`（及配套 vllm）
2. 按本清单逐文件打补丁（`grep -n "CSTPO"` 对齐语义）
3. `bash scripts/mc_ppo_critic_run.sh`（超参数在 cstpo/configs/rl_defaults.sh）
