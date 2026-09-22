# MC-PPO 实验记录（2026-09-16 至 09-17）

四轮实验的完整数据链与诊断结论。每轮均为 esconv、8 种子、GRPO、
12 轮交互、batch 8（4 prompts × n=2）、5 steps、lr 1e-6、LoRA r64。

## 轮次总览

| 轮 | 配置 | val reward | 训练 reward 趋势 | 关键指标 | 诊断 |
|---|---|---|---|---|---|
| 1 | base 冷启动、bypass、T=1.0 | 0.417 | 0.27 → 0.02 下滑 | clipfrac 正常、KL 0.0004→3.15 | 策略劣化：base 无行为先验 |
| 2 | +SFT adapter、T=1.0 | 0.156 | 0.27(epoch0) → 0.02 | clipfrac 0.008→0.48、KL 一步 2.7 | 温度失配污染 ratio |
| 3 | +KL 锚定(0.05)、T=0.7 | 0.182 | — | KL 2.7-2.8、clipfrac 48% | 同上 + KL 系数弱 |
| 4 | 关 bypass、T=0.5、KL(0.1) | 0.328→0.25→0.19 | 0.026 → 0 | **clipfrac <3%、ppo_kl≈0**、rewards -191 | ratio 干净但 KL 拉向 base |
| 5 | 关 bypass、T=0.5、**无 KL**、两段式 trie | 0.214（num_turns min 2↑） | 0.154 → ~0.01 | clipfrac <1.1%、ppo_kl≈0 | 首轮冒犯消失；reward 仍第一步后崩 |
| 6 | n=4、lr 3e-6、10 steps、test_freq=5 | 0.156（仅 step0 快照） | 0.171 → ~0.01（epoch1-9） | clipfrac <0.3%、ppo_kl≈0 | 同上模式；val 快照未触发（test_freq 疑无效） |

## 轮 6 后的核心矛盾与裁决实验

**矛盾**：ppo_kl≈0.0002（策略几乎没动）但训练 reward 从 epoch0 的 0.171
崩到 epoch1+ 的 ≈0.01——微小更新不可能造成 10 倍行为变化。

**两个假设**：
- A：RL 毁模型（更新虽小但破坏关键行为）→ 基线多次采样应稳定 ≥0.2
- B：reward 稀疏 + 采样噪声（epoch0 的 0.17 是噪声高点，judge 9 档离散
  下高分轨迹是稀有事件）→ 基线多次采样均值 ≈0.1 且方差大

**裁决实验**：sft_baseline_snapshot.py——SFT adapter 温度 0.5 采样 val 20
种子 × 2 次（与 RL rollout 同口径），看 G 分布。

**裁决结果（2026-09-18 凌晨，40 条轨迹）**：
- G mean = **0.144**，非零占比 **12/40 = 30%**，非零均值 0.48
- 72.5% 轨迹 judge 判 E=A=0（"无证据"档）——**reward 极度稀疏**
- epoch0 的 0.171 完全在基线分布内（不是 RL 毁模型前的"好起点"）
- 但 9 个训练 epoch 均值 ≤0.025 仍显著低于基线期望（8 条轨迹 × 30%
  非零 → 期望 2.4 条非零；连续 9 epoch 近乎全零的概率极低）——
  vllm 两段式生成与 HF 温度采样的生成质量差异待查（或策略更新虽小
  但确实伤行为）

**对照实验**：greedy 口径基线（TEMP=0 + 三件套，A/B 验证同口径）已启动——
若 greedy 下 G≈0.4+（接近 A/B 的 0.498），则**温度 0.5 是 RL rollout 的错误
选择**（SFT 模型高温退化），修复方向为降温度 + 探索由其他机制提供；
若 greedy 也 ≈0.15，则 val 种子集（esconv_21-40）本身比 A/B heldout 难，
温度无关。

## 三个根因（按发现顺序）

### 1. 温度失配（轮 2-3 的元凶）

bypass_mode 用 vllm 采样 logprob 当 π_old——含温度 T 的重归一化分布；
actor 训练 π_θ 用原始 logits（T=1）。ratio = exp(π_θ - π_old) 被系统性
扭曲，T 越低越严重。SFT 模型（只在 greedy 验证过）在高温+失配梯度下
生成极端输出 → 1 轮冒犯终止 → 0 分 → 策略崩。

**修复**：关 bypass（`algorithm.rollout_correction.bypass_mode` 不设），
actor 重算 old logprob（π_old 与 π_θ 同实现同温度）。flash_attn 缺失的
unpad 路径由 bert_padding_shim 补（见 VERL_ENV_NOTES §5b）。

**证据**：轮 4 的 clipfrac 从 48% → <3%，ppo_kl 从 2.8 → ≈0。

### 2. KL 惩罚拉向 base（轮 4 的元凶）

use_kl_in_reward=true 后：
- ref 引擎无 MEMDBG(fwd_only=True) 构建日志——ref 的 SFT adapter 疑似未
  加载（`ref_config.model_config = deepcopy(model_config)` 理论上含 adapter）
- KL 起点 1.25（若 ref=SFT 应为 ≈0）——ref 实为 base 行为
- token 级惩罚口径：rewards = score - 0.1 × KL_token，1800 tokens × KL 2.4
  → 惩罚 -432 量级，judge 分数（~0.02）被淹没 → rewards/mean = -191
- 策略被锁死在 base 附近（ppo_kl≈0 是"不动"不是"稳定"）

**修复（当前轮）**：去掉 use_kl_in_reward，reward = 纯 judge 分数；
漂移控制交给 lr 1e-6 + clip。**TODO**：ref adapter 继承机制排查
（engine_workers.py:520 的 deepcopy 路径），需要时再加回 KL 锚定到 SFT。

### 3. 约束重归一化失配（两段式 trie 的设计动机）

约束采样（choice/grammar）下被 mask 的 token 分布是"合法子集重归一化"，
actor 无约束 forward 的 logprob 是全词表 softmax——若约束 token 参与 loss，
ratio 再次被污染（与温度失配同构）。**解法**：标签 token mask=0。

## 环境修复记录（本轮 4→5 之间）

- flashinfer JIT 编译崩：cccl 版本检查（cu13 nvcc vs cu12 头）→ 头文件
  加 CCCL_DISABLE_CTK_COMPATIBILITY_CHECK；链接 -lcudart 缺失 → cu13 lib
  symlink + LIBRARY_PATH（详见 VERL_ENV_NOTES §10）
- 两段式生成：标签段 choice 约束（vllm structured_outputs → xgrammar），
  话语段自由生成；verl vllm_async_server patch dict→StructuredOutputsParams

## 配置状态（scripts/mc_ppo_run.sh，轮 5）

```
GRPO、无 KL、T=0.5、top_p=0.95、n=2、batch 8、12 轮、
SFT adapter 初始化（is_trainable）、lora.merge=true 同步、
两段式 trie 约束（标签 mask=0）、enforce_eager、flex 未启用
```

## Schema 一致性清单（用户裁定原则：各流程 schema 必须一致，防分布漂移）

| 项 | SFT 训练/推理 | RL rollout | 状态 |
|---|---|---|---|
| chat template | enable_thinking=False（注入空 think 块） | 曾未传（模板默认不注入） | **已修**：`+data.apply_chat_template_kwargs.enable_thinking=false` |
| 两字段格式 | `<标签>\n<话语>` | 两段式拼接 = 同格式 | ✓ |
| 前缀 | prefix_turns 2 轮 user 结尾 | 同 | ✓ |
| tokenizer | Qwen3-8B | 同 | ✓ |
| 采样 | greedy + 三件套（推理口径） | T=0.5（探索口径，评估回 greedy） | 有意区分，评估协议已冻结 |
| suppress_tokens | 推理有 | vllm 侧未配（ngram 不支持） | 待评估：低优先级 |

## rollout 轨迹保留

`logs/rollout_traj/{task}.jsonl`——每轨迹一行（dialogue 全文 + E/A +
termination），由 verl_cstpo_reward.compute_score 落盘。诊断分布漂移用。

## 下一步待办

1. 轮 5 结果验证（训练 reward 是否上升、标签合法性）
2. n=2→4（组内 baseline 强化）
3. 训练步数 5→20+，val 快照（test_freq）
4. ref adapter 排查（若重新引入 KL）
5. 按 MC_PPO_EVAL_PLAN 做 SFT 基线对比
6. **greedy 基线裁决结果**（跑完分析：温度口径决策）
7. **p4g 权重重训**（现存 208 为 think 污染版，无关闭版可用）

## 轮 7-8 与 free_cache 裁决（2026-09-17 夜，收束记录）

### 五类腐败审计与判别实验（三档对照）

轮 7 rollout 全面审计发现 5 种腐败模式（bang 40 条 / token soup 乱码 ~122 条
/ role 前缀 847 条 / 句内循环 725 条 / 中文泄漏 1405 条）。判别实验
（`test_merged_vllm_real.py`，8001 HF-merge 权重 + 真实种子三档）：
乱码/bang/中文全部来自 **verl lora.merge 初始 adapter 路径**（HF-merge 下消失）；
跨轮固定回复循环（greedy 12/19、T=0.2 档 158 条话语）由 **rep_penalty=1.15**
根除（0/19）；`assistant:` 前缀与权重/温度/rep 均无关（三档 17-18/19）——
后处理剥离。

### 修复集（全部落地验证）

1. `model.path` → HF-merge 双存（/publicdata/model/CSTPO/esconv），删
   lora_adapter_path；lora.merge=true 保留（vllm 0.28 原生 LoRA 与 Qwen3
   fused QKV 不兼容：`QKVParallelLinear has no attribute base_layer`）
2. CstpoAgentLoop：注入 repetition_penalty=1.15（verl agent loop 硬编码 1.0，
   agent_loop.py:500）+ strip_role_prefix + 500 token 预算刹车
   （4105/4096 缓冲溢出崩溃）
3. traj 落盘加 global_step/is_validate（verl agent_loop.py kwargs 透传 patch；
   naive.py 方案无效已还原）
4. 环境：ray env 白名单 + vllm EngineCore 环境拷贝名单补 LIBRARY_PATH
   （flashinfer 新 kernel 冷编译的 -lcudart 链接；此前靠热缓存幸免）

### 新配置逐步验证（v5 syncdump 轮）

- **RL 起点大幅改善**：val0 G=0.46（r7 为 0.114，逼近 SFT 基线 0.499）
- 训练更新无罪：2 步 LoRA 增量 max 1e-6（252 目标模块），手工 HF 合并生成 0 腐败
- 发送侧权重干净：396 键逐张量比对 0 超差
- vllm 独立路径干净：磁盘 reload ×4、内存 load_weights ×2、8 并发热更新 ×96 条话语
- 但 v5 归因显示 **gs=2 起 32/52 CJK**——腐败只在真实栈第二次权重同步后出现

### 根因与修复：free_cache_engine 的 sleep/resume

B4（free_cache_engine=false，同配置）逐步审计：gs=0/1/2 乱码脚本 0/0/0、
CJK 0/1/0、bang 全 0——**sleep 释放权重、wake 恢复不全是元凶**（与 ms-swift
PR #7017 GRPO sleep_level=2 → gibberish 同类）。E/A 非零率升至 77%（v2 为 45%）。
上游对标：vllm #42821（MoE 二次 load 腐蚀，仍 Open）机制类但不适用 dense；
vllm #38374（IPC 热换发散，无公开修复）——与本问题类同但触发点不同。

### 遗留与备忘

- verl 0.8 merge 分支的 state_dict 别名 bug（context 退出后 materialize →
  静默发送未合并权重）在 0.9 已修（_merged_lora_per_tensor_param）；本场景
  LoRA 增量 1e-6，影响可忽略
- verl 0.8 save_checkpoint 在 merge=true 下有 FSDP 断言 bug（弃用 save_freq，
  改同步时落盘 adapter/merged 双份）
- 正式评估协议：val 20 种子（esconv_21-40，与基线快照同种子）、greedy+三件套、
  judge 3 次聚合、配对 t 检验

## 正式评估裁决（2026-09-18 凌晨，v7 20 步 GRPO）

- **主指标不通过**：ΔG=+0.0312 数字过 +0.03 线，但配对 t=0.688（df=19）→ p≈0.5，远不满足 p<0.1——GRPO（当前配置）未产生统计显著增益。种子级差异 -0.438~+0.312，7 正 9 负 4 零。
- 提前终止率 RL 17.5% vs SFT 15.0%（轻微升高）。
- **标签组件功能缺失（重大发现）**：RL 模型 greedy 无约束首行 0/251 合法标签（输出直接是 "assistant: ..."）；probe 证实 SFT 基线同样为 0——标签头在推理时完全失效（历史回合无标签先验占优），与 RL rollout 约束下 99.4% 塌缩 Information 互为表里。
- 失败归因链：标签组件失效 + 奖励稀疏（9 档 judge/轨迹级标量/组内 baseline）+ LoRA 更新 1e-6——三个因素互相咬合。
- 下一步（按预注册失败路径）：① 标签组件决策（修 SFT 或废弃标签协议）升级为最高优先，直接决定认知 critic 字段设计；② judge 0.5 粒度重新校准；③ 认知 critic 提前。

## 2026-09-19 深夜：标签未被训练根因确认（v10 复盘 + 静态证据链）

**用户怀疑（标签根本没被训练）证实**：
1. v10 正式评估：G_rl=0.4604 vs G_sft=0.3937，ΔG=+0.0667 t=1.417——比 v9
   （+0.081 t=1.78）差，回落到 v8 水平（+0.057 t=1.42）。×5 加权放大的是
   垃圾梯度。
2. 标签迁移三阶段全冻结：Question 13.0→13.1→12.8%、Reflection
   11.3→11.1→11.5%、平均 G 0.477→0.476→0.474。
3. **根因（静态证据链）**：约束内 ratio 的 choice 掩码只存在于
   `prepare_model_outputs` 的 `use_remove_padding=true`（rmpad）分支，
   而 v8/v9/v10 正式跑全部 `use_remove_padding=false` → 掩码是死代码 →
   标签 token 的 π_θ 在无约束分布、π_old 在 vllm choice 约束分布，
   ratio 空间错配 → 标签梯度为噪声 → 40 步冻结。×5 加权补丁（DataProto
   层）确实生效但放大的就是这些噪声。
4. 数据流确认：extra_fields → _postprocess non_tensor_batch →
   to_tensordict NonTensorStack → index_select_tensor_dict 全程保留
   label_offsets；但旧补丁的 `for off in label_offsets` 迭代会对
   NonTensorStack 元素 int() 崩 TypeError——v10 未崩恰好证明该分支从未
   被执行（死代码）。
5. 修复：choice 掩码补进 padded 分支（行基 = cu_seqlens 前缀和，逐样本
   NonTensorStack 解包）；rmpad 分支同款修正；smoke 加
   CSTPO_RATIO_DEBUG=1 落盘诊断验证数据真实到达。
6. 连带发现：critic 的 ValueHeadWrapper 用 output_hidden_states=True +
   fp32 在 critic_infer_batch 撑到 51GB OOM——forward 改走内层
   Qwen3Model（仅 last_hidden_state、不算 logits）+ critic.fsdp.model_dtype
   =bf16（注意 hydra key 是 critic.fsdp.*，critic.engine.* 会被
   __post_init__ 覆盖）。

## 2026-09-19 深夜（续）：critic smoke 失败-修复链（#17-#20）

- #17 OOM（actor+vllm+游走进程共置 GPU0）：verl 0.8 所有角色共置
  global_pool → critic 分池补丁（main_ppo.py）+ 两卡布局
  n_gpus_per_node=1、GPU_INDEX=1,3。
- #18 OOM（critic_infer_batch 51GB）：ValueHeadWrapper output_hidden_states=
  True（28 层全 hidden ~18GB）+ fp32 模型 28GB → forward 改走内层
  Qwen3Model（仅 last_hidden_state、不算 logits）+ critic.fsdp.model_dtype
  =bf16（hydra key 是 critic.fsdp.*，critic.engine.* 被 post-init 覆盖）。
  修复后 critic 稳定 ~23GB。
- #19 TypeError（get_non_tensor_data 缺 default 参数）——**关键正面证据**：
  崩溃发生在 `if label_offsets is not None` 守卫之内 →
  `/tmp/cstpo_ratio_debug.txt` 落盘 "present=True type=NonTensorStack n=1"
  ——label_offsets 真实到达 padded 分支掩码代码，数据流闭环，v8-v10
  标签冻结的机制解释终审成立。补 default=None 修复。
- #20 RuntimeError（GAE masked_whiten 形状不匹配 4096 vs 16 @dim2）：
  values 从 critic 出来形状不对（v_head 输出 (bs,seq,1) 3D 与
  response_mask 2D 广播失败或更复杂）。已加 CSTPO_SHAPE_DEBUG 打印
  三个张量形状，smoke 重跑取证中。
- smoke 已通过的里程碑：模型加载（ValueHeadWrapper+peft+FSDP bf16）、
  初始验证（val 0.419 / 0.492 两轮）、step0 rollout、critic 值计算
  （23GB，不再 OOM）、label 掩码数据到达（present=True）。

## 2026-09-20 凌晨：critic smoke 验收通过 + 正式 C0 run 启动

- #21 根因与修复：CSTPO-SHAPE 实测 `values (16,4096,1)`——v_head 尾维 1
  未挤，GAE 内 `(bs,1)×(bs,)` 广播成 (bs,bs) 污染优势（verl 非 v_head
  路径有 squeeze，v_head 路径没有）→ wrapper forward 改返回 2D。
- **smoke 全项验收通过**：2 步无崩溃；vf_loss 0.174→0.157（vpred_mean
  -0.60→-0.46）下降；ratio debug 48 行 present=True（全部微批×3 阶段，
  标签 choice 掩码真实执行）；权重同步/轨迹落盘正常；critic 内存 23GB
  稳态。teardown DataLoader SIGKILL = 已知 cgroup 问题（无害）。
- 15+ 次失败的全部根因链闭合：trl shim → __getattr__ 注册表 →
  critic 分池 → fp32/全 hidden states 内存 → label 数据流 default 参数 →
  v_head 尾维。
- **正式 C0 run 启动**（mcppo_esconv_critic_v1）：20 步、batch 8、
  critic_warmup=5、GAE γ=0.99 λ=0.95、GPU 1,3、lr critic 1e-4 /
  actor 1e-5、标签优势加权关闭（CSTPO_LABEL_ADV_WEIGHT=1）。分析点：
  vf_loss 曲线、vpred 收敛、val 曲线、40 步内标签迁移速率（校准
  pre_result/ 的收敛预测图）、20 步后 ΔG 评估。

## 2026-09-20 上午：actor NaN 根因定位与修复（用户"参数更新是否有问题"之问的完整答复）

- 现象：正式 C0 run（v1）20 步 actor/pg_loss、ppo_kl、loss 每步全 NaN，而
  grad_norm 有限（0.17→0.13 递减）、零 skip 警告、critic 正常收敛（vf_loss
  0.214→0.011）。标签分布全程冻结。
- 仪器（losses.py CSTPO_NAN_DEBUG 落盘）：loss 输入三张量 **零 NaN**，但
  log_prob 含 3-12 个 -inf/微批 ≈ 每标签一个。
- 根因：vllm choice 采样产出的标签 token 路径与 build_label_trie 的规范编码
  不一致（前导空格/换行等 token 变体；decode+strip 后仍是合法标签所以
  agent loop 兜底不触发）→ 掩码允许集不含实际 token → 该行 logprob=-inf →
  π_old/π_θ 两处重算同位置 -inf → -inf-(-inf)=NaN 进入 ratio → pg_loss/ppo_kl
  指标 NaN。
- 修复：agent loop 标签段一律按解码文本重编码为规范路径（与 trie 同源）。
  验证（1 步诊断 run）：TOKEN-MISMATCH=0、inf:log_prob=0、pg_loss=1.3e-08
  有限、ppo_kl=0.0、clipfrac=0——全绿。
- 正式 C0 run v1b 重启（全部修复在位：choice 掩码 padded 分支 + 标签重编码
  + critic 分池 + bf16/内层 forward + values squeeze）。

## 2026-09-21 凌晨：v1c 提前终止（38/40 步）与 C0 单 critic 阶段结论

- v1c（40 步、lr 3e-5、v1b 合并基座、200 种子池）在 step 38 按用户指令提前
  终止（8h38m）。最终 LoRA B 范数 ~0.14（起点 0，更新确凿累积）；vf_loss
  0.214→0.0008（训练集拟合趋零）。
- **无收敛迹象**：val 0.488→0.473→0.390→0.415（噪声带内回落）；G 平台
  0.43（vs 起点 0.49）；三窗口配对 Question Δ：+0.008（6-20）→+0.036
  （21-30）→+0.003（31-38）——中间窗口的升势被否定，是噪声；标签分布
  整体无迁移。
- 判读：单标量 critic（C0）+ lr 3e-5 在 40 步内买不到标签迁移与回报
  提升。vf_loss 趋零 + val 下滑 = V 训练集过拟合/泛化不足的典型签名；
  标签 credit 被话语 token 稀释（6% 占比）是结构性质疑点。
- 下一步序列（按优先级）：① 标签优势加权 ×5（CSTPO_LABEL_ADV_WEIGHT，
  零代码成本，直接对症稀释）；② label/utterance 双头（credit 分段各归
  各）；③ GRPO 修复版对照（任务 #13，检验 critic 相对轨迹级 credit 的
  增量）；④ 扩训练池（400 种子方案已备）。
- 资产保留：logs/v1c_final_adapter.pt（38 步 LoRA）、全量轨迹落盘
  （v1c 部分 ~1100 条）、200 种子池、100 测试集（三任务齐）、评估脚本
  已修（冻结 val parquet + 零 adapter 同管道对照 + EVAL_OUT/EVAL_PARQUET
  参数化 + 对话落盘）。
