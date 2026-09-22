# SFT 训练配置（2026-09-16）

权威实现：`cstpo/sft_config.py`（训练/掩码/推理共用）；本文件为可读版说明。
上位规格：`result/SFT_SPEC_20260915.md`。

## 1. 模型与数据

| 项 | 值 |
|---|---|
| 基座模型 | Qwen/Qwen3-8B（三任务各自独立训练，共享基座） |
| 数据 | `data/sft/<task>/train.jsonl`（真人）+ `vanilla.jsonl`（香草） |
| 混合 | 真人:香草 = **8:2**（装配时真人下采样至 4× 香草样本数） |
| 最大长度 | **4096 tokens**（实测 P99×1.5 均低于下限，覆盖全部样本） |
| 系统提示 | 论文香草提示（与 agent.py 推理逐字节一致） |

## 2. 输出格式与 loss 掩码（话语 SFT 口径）

- 每轮输出 = `"<策略标签>\n<话语>"`（纯文本标签行 + 换行 + 话语）
- **loss 只算最后一条 assistant 消息的话语部分**；掩掉：
  system/user 全部、历史 assistant 全部、目标回合的**标签行前缀**、
  Qwen3 模板结构标记（`<|im_start|>`/`<|im_end|>`）
- 实现：`cstpo/sft_tokenize.py::tokenize_sample`——标签行单独编码定位边界，
  自检通过（学习 token 仅含话语，标签行全 mask，pad 至 4096）
- 无标签回合：占位 `<unlabeled>` 前缀 + 同样 mask（占位路径 RL 时不可达）

## 3. 训练超参（单卡 A100-80G，full fine-tune）

| 参数 | 值 | 说明 |
|---|---|---|
| LR | 2e-5 | 8B full FT 常规值 |
| Epochs | 2 | 样本总量 ~2-4 万，2 轮足够 |
| Micro batch | 2 | 4096 长度下 80G 内存约束 |
| Grad accum | 8 | 等效 batch 16 |
| Warmup | 3% | |
| Weight decay | 0.01 | |
| 精度 | bf16 + gradient checkpointing | 必须（内存） |
| 保存/评估 | 每 500 步 | 评估集 = 校准留出 30 条（judge A/B） |

- **LoRA 模式（2026-09-16 裁定）**：rank 64 / alpha 128 / dropout 0.05 /
  target 全线性层——快速验证首选；保存 adapter（RL 阶段挂载，verl 支持
  LoRA policy）+ merge 完整权重（A/B 推理用）。A/B 验证与香草提示打平或
  不如时，先提 rank 或回退 full FT 再下结论。速度实测 1.6×（3.07 vs
  1.87 样本/秒）
- **混合口径（2026-09-16 修订）**：真人:香草 = **5:5**、**1 epoch**
  （首轮 8:2/2 epochs 验证显示真人风格（跑题/复读/表情退化）主导模型
  行为；5:5 让香草生成对话的提示纪律约束占一半权重）。真人下采样为
  **按历史长度分层采样**（防长对话垄断训练分布）
- **格式统一（2026-09-16 修订）**：ESConv 训练目标带 `assistant: `
  前缀（对齐香草基线输出——flash 遵从提示格式行输出此前缀并原样发
  模拟器；前缀计入 loss，让 SFT 模型学会同样输出）。P4G/CB 无此格式行，
  目标不加前缀
- **思考关闭（2026-09-16 补修）**：训练 tokenize 与推理生成统一使用
  `apply_chat_template(enable_thinking=False)`（模板层注入空 think 块，
  think 由输入承载而非模型生成）。首版训练未关思考——Qwen3 基座先验导致
  生成时 <think> 泄漏，生成侧 suppress 是兜底，模板统一是治本；
  训练数据实测 0 污染（46,670 目标无 think）
- **生成配置三件套（推理/验证/RL 三处一致，SPEC 强制）**：贪心解码
  （do_sample=False）+ `suppress_tokens`（禁 `<|url|>`/`<|code|>`/
  `<|audio|>`/`<|video|>`/`<|quote|>`/`<think>`/`</think>`，防 URL 退化
  循环与 Qwen3 思考标签泄漏——实测训练数据 0 污染，泄漏源自基座思考
  先验，训练阶段无需处理）+ `repetition_penalty=1.15` +
  `no_repeat_ngram_size=4`（防多轮复读与拖音）+ 输出首段截断 + max 128
  tokens（防多轮续写泄漏）。首轮 A/B 验证教训：漏 rp 时 p4g 0/5、
  esconv G=0.125；加 rp 后 p4g 3/5
- **已知问题（记录不修）**：(1) PPDPP 提示固有歧义——"with the price of
  {target}" 使模型把买方目标价误作挂牌价（用户已裁定 baseline 保留不修）；
  (2) CB 表情退化（真人语料表情密度高，贪心放大成表情链）——rp 部分缓解，
  若重训后 CB 仍差则数据侧过滤高表情密度样本
- 产物：`outputs/sft/<task>/`（日志，不入库）、`checkpoints/sft/<task>/`（不入库）

## 4. 验证闭环

SFT 后：`judge_with_aggregation`（3 次聚合）在校准留出 30 条上与香草提示
A/B（同预算）；未超过香草提示 → 回炉数据/提示。通过后进入 RL（verl 主候选）。

## 5. 训练脚本（待写）

- 首选 verl 的 SFT 入口（用户裁定 verl 主候选）；兜底自写 accelerate +
  transformers Trainer 脚本（数据 collator 直接复用 sft_tokenize）
- 两者都必须：用 `sft_config` 常量、用 `sft_tokenize` 的掩码逻辑——
  不得各自实现一套掩码
