# CSTPO

认知状态转移驱动的主动对话策略优化（Proactive Dialogue Policy Optimization via Cognitive-State Transition）。

本项目围绕三个主动对话场景开展研究：P4G 劝说捐赠、CraigslistBargain 讨价还价和 ESConv 情感支持。

研究包含两个方向：

- **Cog-Sim**：认知心理学启发的用户模拟器，显式建模用户信念、愿望、意图（BDI）及情绪变化。
- **CSTPO**：利用训练侧认知信息辅助长期价值估计，探索策略标签与话语的分层优化，以及有限 rollout 预算的有效分配。

Actor 仅使用对话历史及自身角色合法可见的任务信息；用户内部认知状态仅用于训练侧。终局任务奖励独立于内部认知变化幅度。

## 目录

```text
Cog-Sim/            用户模拟器源码与示例
experiments/        已有审计、先导实验脚本及结果
shared_work_space/  算法设计、数据规范、实验协议与推进记录
```

## 仓库管理

`Cog-Sim/` 当前作为源码副本由本仓库统一管理，来源为 [Cog-Sim](https://github.com/lucianma05-create/Cog-Sim)，不是 Git submodule。直接克隆 CSTPO 即可获得模拟器源码，不需要递归初始化子仓库。后续如需两个仓库独立维护和发布，可迁移为固定 commit 的 submodule；不要直接在此目录中再次 `git init` 或嵌套克隆。

保留模拟器、评估代码和可复现的诊断脚本；生成的结果、评估报告、运行日志、模型产物及本地临时脚本不入库。`shared_work_space/` 延续本地研究文档的管理方式，不随仓库分发；下方研究文档链接供本地工作区使用。

## 文档入口

先阅读 [UPDATEME：文档职责与推进顺序](shared_work_space/UPDATEME.md)，再按需查看：

1. [算法概览](shared_work_space/01_CSTPO算法概览.md)
2. [Rollout 数据与种子规范](shared_work_space/02_Rollout数据与种子规范.md)
3. [实施规范](shared_work_space/03_CSTPO实施规范.md)：当前算法实现的主要依据。
4. [任务评价与实验协议](shared_work_space/04_任务评价与实验协议.md)
5. [历史审计与先导记录](shared_work_space/90_CogSim历史审计与先导记录.md)

模拟器使用说明见 [Cog-Sim README](Cog-Sim/README.md)；已有诊断脚本说明见 [experiments README](experiments/README.md)。

## 当前状态

目前处于算法设计与验证准备阶段，已有模拟器审计及 36 个 demo 单轮分支的先导诊断，尚未完成正式 RL 训练与三任务实验。认知敏感度、采样效率和外部泛化收益均为待验证假设；具体前置条件与验收标准见实施规范和 UPDATEME。
