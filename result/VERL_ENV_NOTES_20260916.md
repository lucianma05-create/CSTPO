# verl 环境问题存档（2026-09-16）

记录 verl RL 环境从零到跑通 smoke 的所有坑，避免后人重复踩。

## 1. verl 版本：0.9 → 0.8.0

- verl 0.9 依赖 `transfer_queue`（VeOmni 生态），安装即报 `ModuleNotFoundError: transfer_queue`。
- 降级到 **verl 0.8.0**（配 vllm 0.29.0），API 与 0.9 不同：
  - 自定义 reward 走顶层 `custom_reward_function.path/name`。
  - 入口 `python -m verl.trainer.main_ppo --config-name ppo_trainer`。
  - HFModelConfig 在 `actor_rollout_ref.model` 下。

## 2. flashinfer JIT：nvcc 版本不匹配

**现象**：vllm 启动（standalone 或 rollout）报
`nvcc fatal: Unknown option '--compress-mode=size'`。

**根因**：flashinfer JIT 编译取 PATH 上的 nvcc，base 环境的 CUDA 12.4 nvcc
不支持 flashinfer 需要的参数；verl 环境自带的 CUDA 13 nvcc 才支持。

**修复**：`scripts/verl_env.sh` 把 verl 环境内的 nvcc 目录置于 PATH 前：

```bash
export PATH="/data/user21300120/.conda/envs/verl/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH"
```

任何跑 vllm/verl 的命令都要先 `source scripts/verl_env.sh`。

## 3. FP8 FusedMoE 导入失败（vllm 0.29）

**现象**：`from vllm.model_executor.layers.fused_moe import FusedMoE, ...` 失败。

**根因**：vllm 0.29 移除了 `FusedMoE`，verl 0.8 的 fp8 工具模块仍在导入。

**修复**：patch 了
`/data/user21300120/.conda/envs/verl/lib/python3.12/site-packages/verl/utils/vllm/vllm_fp8_utils.py`：
FusedMoE/LinearBase 置 None + warning，非致命。注意：**conda 环境重装后需重新 patch**。

## 4. LoRA 配置路径（重大坑）

**现象**：配了 lora 但 actor 仍吃 60G+（8B 全量 Adam：权重 16G + 优化器 32G + 梯度 16G）。

**根因**：`actor_rollout_ref.actor`（FSDPActorConfig）**没有 lora 字段**，
lora_rank 只在 `actor_rollout_ref.model`（HFModelConfig）下。
写到 actor 下会被 hydra 忽略（或报错），LoRA 静默不生效。

**正确用法**：

```bash
actor_rollout_ref.model.lora_rank=64
# 不要写 actor_rollout_ref.actor.lora_rank —— 该字段不存在
```

FSDP 引擎按 `model_config.lora_rank > 0` 判定是否走 PEFT
（`verl/workers/engine/fsdp/transformer_impl.py:141`）。

**第二个 LoRA 坑：`use_orig_params` 必须 true**。默认 false 时 FSDP 把 frozen
base + LoRA flatten 成一个 flat param（requires_grad=True），optimizer 对全量 8B
建 Adam 状态 ≈ 64G，进程直接 61G OOM。修复：

```bash
actor_rollout_ref.actor.fsdp_config.use_orig_params=true
```

（PEFT + FSDP 的标准要求；frozen 参数 requires_grad=False 被 torch.optim 跳过，
只有 LoRA 参数进 optimizer。）

**第三个 LoRA 坑：`model_dtype` 默认 fp32**。FSDPEngineConfig.model_dtype 默认
fp32（yaml 里 `model_dtype: fp32`），torch_dtype=None 时 training 强制 fp32 →
8B 模型以 fp32 加载 ≈ 32G（"After FSDP allocated 31.17G"）。修复：

```bash
actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
```

（MixedPrecision param_dtype=bf16 只 cast forward/backward，不改变存储 dtype——
存储 dtype 由 model_dtype 决定。bf16 后 FSDP wrap 仅 ~16G。）

**验证手段**：跑之前先 `--cfg job` 解析配置，grep lora_rank 确认 64 落在 model 下。

## 5. FlashAttention2 未安装

verl FSDP 引擎默认 `attn_implementation=flash_attention_2`，env 里没装 flash-attn，
init_model 报 ImportError。修复：

```bash
+actor_rollout_ref.model.override_config.attn_implementation=sdpa
```

注意 hydra 语法：`override_config` 是普通 dict（非 DictConfig），直接
`actor_rollout_ref.model.override_config.attn_implementation=sdpa` 会报
object_type=dict，必须加 `+` 前缀（hydra 对 dict 的 key 添加）。
（`verl/workers/engine/fsdp/transformer_impl.py:185` 从 override_config 读取，默认 flash_attention_2。）
vllm rollout 侧不受影响（自有 attention 实现）。

## 6. GRPO 必填 rollout log_prob micro batch

**现象**：validate_config 报
`[actor_rollout_ref.rollout] Please set at least one of 'log_prob_micro_batch_size' or 'log_prob_micro_batch_size_per_gpu'`。

GRPO 的 log prob 由 rollout worker 计算，必须显式配
`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1`。

## 6. 单 GPU colocation 内存预算（8B LoRA rank 64）

GPU 1（A100-80G）目标布局：

| 组件 | 内存 |
|---|---|
| actor（FSDP + LoRA r64 + grad ckpt） | ~20G |
| ref（base，forward-only） | ~17G |
| vllm rollout（gpu_memory_utilization 0.35） | ~28G |
| 合计 | ~65G / 81.9G |

- `tensor_model_parallel_size` 默认 2，单卡必须 override 为 1。
- vllm 的 gpu_memory_utilization 按整卡算，不是按剩余内存算——太低会
  "No available memory for the cache blocks"。
- 共享机器上 GPU 内存波动（其他用户任务），跑之前先 nvidia-smi。

## 7. LoRA adapter 同步与 vllm 0.29 不兼容（挂起陷阱）

**现象**：训练进程全链空闲挂起——WorkerDict 等 vllm HTTP 响应、GPU 0%、
CPU 空闲、无报错输出（日志被 conda run 块缓冲）。vLLMHttpServer 的 ray
`.out` 日志里有：
`RuntimeError: Worker failed with error 'QKVParallelLinear has no attribute base_layer'`

**根因**：verl 0.8 的权重同步把 LoRA adapter + peft_config 传给 vllm 0.29，
vllm 0.29 v1 引擎重构了 attention 层，adapter 加载协议不匹配，引擎 worker 崩。

**诊断要点**（挂起类问题）：
- 日志 0 字节 = conda run 块缓冲，不是没输出；看
  `/tmp/ray/session_latest/logs/worker-*.out/.err`（ray 自己落盘）。
- 进程全空闲 + GPU 0% → 用 `curl http://127.0.1.1:<port>/health` 探测
  vllm HTTP 是否还活着；`ps` 找 vLLMHttpServer/EngineCore 进程树。

**修复**：`+actor_rollout_ref.model.lora.merge=true`——LoRA 合并进 base 后
传完整 bf16 权重（vllm 侧无 LoRA 逻辑）。代价：每次同步 merge（8B 模型
~16G 临时显存 + CPU 时间），smoke 无感，正式训练需评估同步频率。

## 8. 其余杂项

- **flash_attn 硬依赖与 shim**：verl 0.8 的 `_update_actor` 无条件调用
  `left_right_2_no_padding`（`ray_trainer.py`），其 unpad_input 依赖
  `flash_attn.bert_padding`。env 未装 flash-attn（CUDA 13 无 wheel），
  patch：`verl/utils/attention_utils.py` 的 CUDA 分支改指向新增的
  `verl/utils/bert_padding_shim.py`（纯 torch 等价实现，语义与 flash-attn
  bert_padding 一致，pad 位置清零）。conda 环境重装后需重新 patch。
- **GRPO + rollout log probs**：`algorithm.rollout_correction.bypass_mode=true`
  需配合 `actor_rollout_ref.rollout.calculate_log_probs=true`（否则报
  "requires rollout_log_probs in batch"）。
- **agent_loop chunk 约束**：`rollout.agent.num_workers`（默认 8）必须整除
  train_batch_size（AssertionError: only support equal chunk）。
- **数据必须含 reward_model 列**：verl 0.8 naive reward manager 读
  `non_tensor_batch["reward_model"]["ground_truth"]`；parquet 需
  `reward_model: struct<ground_truth: string, style: string>` 列。
- **agent_loop 多 worker 时 agent.num_workers 需 ≥ 数据条数吗**：不需要，
  只需整除 batch。

## 9. merge 同步的 FSDP state_dict 坑（已 patch）

`lora.merge=true` 时 `get_per_tensor_param` 原实现调
`self.module.state_dict()`（FSDP FULL_STATE_DICT 展开），触发 torch 断言
`FSDP assumes base_model.model.model.embed_tokens.weight is in the state_dict`
（root unit 的 frozen 参数在 merge 后丢失）。patch：改为
`self.module._fsdp_wrapped_module.state_dict()`（直接取 PeftModel 的 merged
参数，绕开 FSDP hook）。文件：
`verl/workers/engine/fsdp/transformer_impl.py`（merge 分支）。环境重装后需重打。

## 10. flashinfer JIT 第二次坑：cccl 版本检查（MC-PPO 阶段）

**现象**：val 正常（旧 kernel 有缓存），train rollout 采样参数变化触发**新 kernel
JIT**，ninja 编译失败：
`CUDA compiler and CUDA toolkit headers are incompatible`。

**根因**：flashinfer 打包的 cccl（`flashinfer/data/cccl/libcudacxx`）的
`cuda/std/__cccl/cuda_toolkit.h` 检查 nvcc 版本 vs include 路径的
`cuda_runtime.h` 版本：nvcc 是 cu13（PATH 修复），系统默认 include 的
cuda_runtime.h 是 cu12 → 不匹配。之前能编译的 kernel 不 include cccl 头，
新 kernel 变体 include 了就炸。

**修复**：在该头文件首行加
`#define CCCL_DISABLE_CTK_COMPATIBILITY_CHECK 1`（官方逃生门），并清空
`~/.cache/flashinfer/0.6.16.post3/80/cached_ops`。环境重装后需重打。

**还有链接坑**：过检查后链接报 `cannot find -lcudart`——cu13 的 lib 目录
只有 `libcudart.so.13`（无 `.so` symlink），且 verl_env.sh 只加了 bin。
修复：`ln -sf libcudart.so.13 .../nvidia/cu13/lib/libcudart.so` +
verl_env.sh 加 `LIBRARY_PATH=.../nvidia/cu13/lib`。之后 ninja 手动
`sampling/sampling.so` 链接成功（2.6MB）。

**备选路线**（未采用）：vllm 0.29 v1 的 `--attention-backend FLEX_ATTENTION`
经 verl `rollout.engine_kwargs` 传入无效（下划线/连字符参数名问题），
flex backend 需要 torch.compile，风险未知。

最终成功命令（GPU 1 单卡、Qwen3-8B LoRA r64、GRPO、2 steps）要点：
- step:1 actor/ppo_kl=0.0004 → step:2 actor/ppo_kl=3.15（策略在真实更新）
- 已知尾音问题：训练步完成后 DataLoader worker 被 SIGKILL（系统 cgroup
  内存限制，非 GPU；整机 free 409G）。正式跑建议
  `data.dataloader_num_workers=2` 降低进程数。

- parquet 数据用 pyarrow 显式 schema 写（pandas 处理不了 list<struct> 的 prompt 字段）。
- `trainer.logger=['console']` 避免 wandb 联网。
- 训练启动用 nohup + logs/verl_smoke.log，避免丢命令与输出。

## 11. PPO-critic（GAE）兼容链（2026-09-19，环境重装后需重打）

1. **trl shim 缺失**：verl 0.8 + trl 1.13/transformers 5.10 下
   `AutoModelForCausalLMWithValueHead` 顶层已移除 → `verl/utils/model.py`
   自建 `ValueHeadWrapper`（base_model + v_head + config），输出
   `(logits, None, values)` 对齐 FSDPEngineWithValueHead 约定。
   `__getattr__` 委托前**必须先查 `_modules` 注册表**——否则
   `wrapper.base_model` 会委托给内层 CausalLM 的 `.base_model`（裸
   Qwen3Model），peft 拿到裸 transformer 在
   `prepare_inputs_for_generation` 处崩溃（14 次 smoke 连环踩坑的终点）。
2. `load_valuehead_model`：禁用 TokenClassification 首路径（transformers
   5.x 下 Qwen3ForTokenClassification 会意外加载成功），恒走自建包装器；
   attn 用 sdpa（无 flash_attention_2）。
3. `transformer_impl.py` `_build_module_optimizer`：`ValueHeadWrapper` 时
   LoRA 只包内层 base_model（v_head 留外全精度可训），
   `critic.model.target_modules` 显式列 7 个线性层（all-linear 对包装器
   失效）。
4. `monkey_patch.py`：trl 导入段容忍化（AutoModelForCausalLMWithValueHead
   为 None 时跳过）。
5. **critic 分池**：`verl/trainer/main_ppo.py` ——
   `mapping[Role.Critic] = "critic_pool"` + `init_resource_pool_mgr` 里
   `need_critic` 时加 `resource_pool_spec["critic_pool"] = [1] * nnodes`。
   默认 global_pool 共置 = 每卡 actor+critic+vllm 三份模型，共享卡噪声下
   必 OOM。两卡布局（`trainer.n_gpus_per_node=1`）：global_pool 1 卡
   （actor 单卡 + vllm 同卡，v10 同款实测峰值 32GB allocated）、
   critic_pool 1 卡（critic 单卡）。verl 的 n_gpus_per_node=2 会把每个
   角色组按每卡一 worker 起 FSDP rank（2 actor ranks + 2 critic ranks
   共置 2 卡），并非"1 卡 1 角色"。
6. 共享卡运行要点：`rollout.gpu_memory_utilization=0.25`、
   `rollout.max_model_len=8192`（prompt 1024 + response 4096 ≈ 5120）、
   train_batch/mini_batch 联动（verl 校验 mini ≤ batch）、
   `data.dataloader_num_workers=2`（cgroup 内存）。

## 12. critic 权重落盘（2026-09-20）

actor 的 adapter 每步落盘是 vllm 同步的副产品；critic 侧此前无任何持久化
（v1b 的 20 步 V 训练随进程消亡，v1c 被迫重学）。补丁：transformer_impl.py
的 FSDPEngineWithValueHead 加 `_dump_critic_weights`（optimizer_step 尾部
钩子），环境变量 CSTPO_CRITIC_DUMP_DIR 控制，写 critic_adapter.pt（LoRA
state_dict + v_head）。脚本已接线，默认 DUMP_DIR。加载侧：与 actor 同
方案（预合并进基座或构建 PEFT 目录）。

## 13. 双头 critic（label/utterance 两层次价值分解，2026-09-21）

补丁（CSTPO_DUAL_HEAD=1 激活，默认路径不变）：
1. `utils/model.py` ValueHeadWrapper：加 v_label/v_utter 双 Linear，forward 在
   DUAL 模式下返回 `(None, None, {"label":..., "utter":...})`
2. `workers/engine/fsdp/transformer_impl.py` FSDPEngineWithValueHead 的 padded
   分支：output[2] 为 dict 时两通道分别 narrow→nested，返回
   values_utter/values_label 键
3. `trainer/ppo/ray_trainer.py`：_compute_values 双通道转换（注意双头模式下
   无单头 "values" 键，tu.get 默认 None 必须跳过）；compute_advantage 双头
   分发（label_mask 从 non_tensor_batch["label_offsets"] 构建）+ **存混合
   returns/values**（下游 metric_utils 无条件读 batch["returns"]/["values"]）
4. `trainer/ppo/core_algos.py` compute_gae_advantage_return_dual：标签链 δ_l =
   γ·V_utter(下一决策点) − V_label（交界交接）、话语链标准 GAE（bootstrap =
   下一轮标签位 V_label）、各流独立 masked_whiten（标签流天然单位方差）
5. `workers/utils/losses.py` value_loss 双头分支：两 head 各自在
   label_mask / utter_mask 上回归 returns_label/returns_utter；指标
   vf_loss_utter/vf_loss_label/vpred_mean_utter/vpred_mean_label 全部
   .detach().item()（指标层 np.mean 不接受 CUDA 张量）

smoke 验收（2 步）：无崩溃、vf_loss=vf_loss_utter+vf_loss_label≈0.999→0.998、
clipfrac 0、actor 双优势更新执行。teardown DataLoader SIGKILL 已知无害。
