# 认知 critic 设计（E 期第二阶段，2026-09-19 起草）

## 动机：从轨迹级 credit 到轮级 credit

- 当前 GRPO：终局 judge 标量 + 组内 baseline → 整条轨迹共享一个优势，
  无时序分解——无法回答"第 3 轮那句 Question 对最终回报贡献多少"
- 标签-回报 3 倍差（Question 0.521 / Reflection 0.179）与 40 步无迁移
  的对照说明：策略选择的优化需要更细的 credit 信号指引
- 认知 critic 的使命：把轨迹级回报分解到轮级，使策略学习
  "何时选什么策略、何时收束"成为可能

## 形态：PPO 家族价值函数（与 GRPO 组内 baseline 的关系）

- critic 网络：价值函数 V(s_t) 估计状态 s_t 下的期望回报
- TD 残差：δ_t = r_t + γV(s_{t+1}) − V(s_t)；GAE 累积成轮级 advantage
- 与 GRPO 组内 baseline 的关系：可共存（组内 baseline 作为低方差底，
  critic 提供时序结构）——第一阶段先做纯 critic 形态（PPO 式），
  与组内 baseline 的消融为后续实验
- 基线估计器 = 标准 TD/GAE（不作为新公式贡献）；策略条件反事实比较
  的精度提升需要独立消融（03 规范）

## 状态表示（预算匹配是硬约束）

| 变体 | 状态内容 | 用途 |
|---|---|---|
| C0 历史 critic（基线） | 仅对话历史文本（策略可见信息） | 与特权版同预算对照 |
| C1 认知 critic（目标） | 历史 + Cog-Sim 训练侧特权状态：BDI、Emotion、用户参数、影响未来转移的环境记忆 | 检验"认知状态对长期贡献的独立价值"（H2 假设本体） |

两者同预算、同架构，唯一差异是特权状态输入——差值的来源即认知状态的贡献。

## 关键约束（03 规范）

1. **Actor 不接触特权状态**：Cog-Sim 状态仅进 critic，不进 Actor/rollout——
   研究边界红线
2. **认知零变化的轮不能自动零 credit**：纯共情/澄清轮可能
   无转移但有长期价值（关系建立）——credit 由价值函数学，不由转移幅度算
3. **转移幅度不直接作奖励**：认知变化是特征不是回报；外部终局回报
   是唯一监督

## 字段 V/U（critic 的两个字段优势）

- 沿用 01 概览的字段设计：critic 输出按字段分解的价值分量，
  验证"状态驱动 vs 用户驱动"两个字段的独立贡献
- 字段分解消融：全量 critic vs 单字段 critic

## 实现路径（verl）

1. `use_critic=true` + value head（Qwen3 加 v_head）——verl 0.8 支持
2. 轮级状态序列化：agent loop 每轮输出状态快照（仅 critic 侧可见），
   critic 输入 = 历史 token + 状态特征拼接
3. reward 时序化：judge 终局分 + 轮级辅助信号（可选：轮级 judge 轻量版）
4. 消融矩阵：C0 vs C1 × 纯 PPO critic vs critic+GRPO baseline

## 验证标准

1. 轮级 credit 预测长期回报的准确性（holdout 轨迹上 V(s) 与实现回报的相关）
2. C1 vs C0 的 credit 质量差（预算匹配下）
3. 端到端：标签分布是否出现方向性迁移（本轮 40 步无迁移的对照）
4. 字段 V/U 的独立贡献显著性
