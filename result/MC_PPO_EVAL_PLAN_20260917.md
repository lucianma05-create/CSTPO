# MC-PPO 评估方案（2026-09-17，训练验收标准）

训练跑通后"怎么算赢"的预定义标准——避免事后挑指标。本文档先于
训练结果最终化之前冻结评估口径。

## 1. 主问题

MC-PPO（GRPO，esconv）相对 SFT 基线（outputs/sft/esconv/checkpoint-604）
在**同一评估协议**下是否有对话质量增益？

## 2. 评估协议（与 A/B 验证口径对齐）

- 采样：**greedy + GEN_CONFIG 三件套**（suppress_tokens、repetition_penalty
  1.15、no_repeat_ngram_size 4、enable_thinking=False）——与 ab_validate 一致；
  RL 训练时用温度 0.5 探索，评估时收敛到确定性口径
- 环境：TaskEnv + Cog-Sim（llm=None 默认 DeepSeek），12 轮 cap
- 打分：judge_with_aggregation（3 次并发，n=3），(E+A)/8
- 数据：val 集从 seeds_draft/esconv 扩到 **20 条**（现 8 条分数波动 ±0.1
  太大；20 条后均值标准误 ~±0.06）

## 3. 主指标与判定

| 指标 | 定义 | 判定 |
|---|---|---|
| **ΔG** | RL 模型与 SFT 基线的 (E+A)/8 均值差（同 20 种子配对） | ΔG > 0 且配对 t 检验 p<0.1 为"有效"；ΔG > +0.03 为"有意义"（A/B 阶段 G 差 0.004 量级，+0.03 已是 7 倍） |
| 提前终止率 | num_turns < 4 的轨迹占比 | RL 不应高于 SFT（行为不退化） |
| 标签合法率 | 首行标签 ∈ 词表占比 | 两段式约束后应为 100% |

## 4. 辅助观察

- 训练曲线：训练中 val reward 的 trend（上升/平台/下降）——每 5 步取
  val 快照（test_freq 配置）
- 对话抽检：RL 与 SFT 各 3 条同种子对话人工读（找行为差异的定性证据）
- 成本账：TurnRecord 的 LLM 调用量对比（RL 是否用更多轮次/更贵话术）

## 5. 已知隐患（评估时注意）

- **种子异质性**：val 多次出现 num_turns min=1——已排查（2026-09-17
  probe_early_term.py）：8 条种子用温和固定话语均能对话 2 轮+，**非种子
  问题**；1 轮终止是模型首轮话语（温度采样下）冒犯/离题触发 simulator
  终止——作为"模型行为"监控项（提前终止率）保留
- **judge 未校准项**：9 条仲裁冲突对话仍未仲裁（前端 8766）——E/A 锚点
  的未决项会传导到 reward 的噪声
- **分数分辨率**：E/A 各 1-4 整数 → (E+A)/8 只有 9 档，20 条种子的
  配对差检出力有限；必要时增加 judge 的 0.5 粒度（需重新校准）

## 6. 执行顺序

1. val 集扩到 20 条（脚本：build_mcppo_data.py 改 N）
2. SFT 基线快照：同协议跑 checkpoint-604 得 G_sft
3. RL 训练（≥20 steps，n≥4 后）快照：得 G_rl
4. 配对对比 + 抽检 + 成本账 → 结论
