# Cog-Sim 反应多样性验证实验报告

日期：2026-09-15（v2 收敛设置，600 条）
对应文档：`shared_work_space/UPDATEME.md` 问题 02 小节；`04_任务评价与实验协议.md` P2。
产物：`data/diversity_probe/`（600 条对话 JSON + summary.json + snr_analysis.json）

## 一、实验构思

**目的**：验证 Cog-Sim 用户模拟器面对不同风格 agent 时是否保留**合理反应多样性**（问题 02 的"保留合理可变性"半边；"减少无依据漂移"半边需盲评，不在此实验范围）。

**对比框架**：提示式用户模拟器阶梯（来自现有工作）作为 baseline：

| 档 | 出处 | 机制 |
|---|---|---|
| cogsim | 本项目 | 完整 TaskEnv：BDI + 情绪 + 转移约束 + CED 结束 |
| std_roleplay | PPDPP Table 11（arXiv:2311.00262） | 纯角色扮演 + 任务视图，无 persona |
| std_persona | ESC-Eval 角色卡 | 强 persona 事实 |
| std_persona_resist | TRIP（arXiv:2403.06769） | persona + 不轻易同意指令（P4G 附 Big-Five 自评） |
| std_bdi | 问题 02 对照档 | 显式 BDI 清单，无转移约束 |

**待证命题（必要条件）**：Cog-Sim 对风格的反应在显著性上不低于提示式基线，且方向合理（施压→反感/僵持）。**不证明**：真实性/无依据漂移（正式 P2 盲评）。

## 二、实验设置（收敛版）

- **刺激**：4 种 agent 风格（`cstpo/agent.py`）——vanilla（论文香草提示：CB=PPDPP Table 8、P4G=TRIP Table 20、ESConv=ESConv-SRA template6）+ cooperative / pushy / neutral（提示末尾追加一句风格约束）
- **规模**：3 任务（ESConv / P4G / CraigslistBargain）× 10 种子（90 冻结种子 dev 前 10）× 4 风格 × 5 模拟器 = **600 条**，同种子跨档配对
- **前缀**：统一前 2 回合（4 条消息；CB 1 回合），agent / Cog-Sim compile_state / 提示式模拟器三方对齐。已知代价：persona/BDI/情绪提取自全量前缀（种子 v4.2），与可见历史存在超前偏差
- **上下文**：全量历史（无滚动窗口），30 轮安全阀封顶
- **结束**：自由结束——Cog-Sim 走 CED（任务中立）；提示式每轮 LLM 自判 end；**15 轮后每 2 轮冗余 judger**（`cstpo/stale_judger.py`：判冗余→生成用户口吻结束语）
- **疲劳机制**：已实现又废弃（v1.0.2 revert，Cog-Sim 保持 v1.0.1-fix）；理由：程序化信号难以区分"僵局"与"支持性长对话"，被 judger 方案替代
- **指标**：结局层（judge rev4 结局、终止分布、平均轮数）、文本层（极性 flash 三分类、长度、distinct-1/2）、状态层（Cog-Sim valence 终点）、多样性（风格 vs 池化 JS）、显著性（ANOVA ω² + 种子分块置换、PERMANOVA、pushy/vanilla 对照、逐轮 SNR）

## 三、实验结果（n=30/风格/档）

### 3.1 结局层

**平均轮数**（新轮，不含前缀）：

| 模拟器 | vanilla | cooperative | pushy | neutral |
|---|---|---|---|---|
| **cogsim** | 7.3 | 6.5 | **10.1** | 7.7 |
| std_roleplay | 5.4 | 7.1 | 5.7 | 6.1 |
| std_persona | 5.6 | 7.0 | 6.1 | 5.6 |
| std_persona_resist | 5.4 | 7.2 | 6.1 | 6.0 |
| std_bdi | 4.5 | 7.1 | 6.1 | 5.8 |

- Cog-Sim pushy 平均 10.1 轮，明显长于其他风格和其他档的 pushy（5.7-6.1）——"面对施压持续抗拒不退出"仍是描述性特征；但统计显著性消失（见 3.4）。

**终止分布**：

| 模拟器 | user_ended | judger_stale_end | 安全阀(30轮) |
|---|---|---|---|
| cogsim | 112 | 7 | **1** |
| std_roleplay | 112 | 8 | 0 |
| std_bdi | 114 | 6 | 0 |
| std_persona_resist | 115 | 5 | 0 |
| std_persona | 117 | 3 | 0 |

- judger 触发 29/600（4.8%），集中在 P4G（17）/ESConv（12），CB 0 次；各档 3-8 次
- 仅 1 条触发安全阀（cogsim pushy）

### 3.2 文本层

Cog-Sim 用户回复正面极性占比（均值 by 风格）：vanilla 0.462 ≈ cooperative 0.455 > neutral 0.361 > **pushy 0.095**——风格单调分化且方向合理。pushy 负极性显著增加为五档共性（见 3.4）。

### 3.2b 极性统计（每种设置：5 档 × 4 风格，pos/neg 占比；完整 60 单元含任务维度见 `data/diversity_probe/polarity_by_condition.json`）

| 模拟器 | vanilla | cooperative | pushy | neutral |
|---|---|---|---|---|
| **cogsim** | .49 / .39 | .43 / .36 | **.10 / .72** | .37 / .46 |
| std_roleplay | .57 / .32 | .64 / .25 | .15 / .79 | .42 / .40 |
| std_persona | .61 / .32 | .69 / .18 | .20 / .71 | .51 / .33 |
| std_persona_resist | .51 / .36 | .60 / .30 | **.06 / .85** | .43 / .39 |
| std_bdi | .62 / .31 | .70 / .21 | .13 / .80 | .49 / .37 |

（表格为 pos/neg 占比；pushy 列五档一致强负面；任务维度差异：CB 整体负极性最高、P4G 非 pushy 风格最正面。）

### 3.3 状态层（仅 Cog-Sim）

终点 valence 均值：cooperative 0.378 > vanilla 0.352 > neutral 0.305 > **pushy 0.097**——施压显著压低情绪。

### 3.4 显著性与多样性

**风格主效应（ANOVA ω² + 种子分块置换，Fisher 跨任务合并）**：

| 模拟器 | n_turns | mean_pol | neg_ratio | distinct1 | mean_len |
|---|---|---|---|---|---|
| cogsim | p=.155 | p<.001 | p<.001 | p=.019 | p<.001 |
| std_roleplay | p=.005 | p<.001 | p<.001 | p=.003 | p<.001 |
| std_persona | p=.023 | p<.001 | p<.001 | p=.002 | p<.001 |
| std_persona_resist | p=.012 | p<.001 | p<.001 | p<.001 | p<.001 |
| std_bdi | p=.015 | p<.001 | p<.001 | p<.001 | p<.001 |

**pushy vs vanilla 对照**：

| 模拟器 | Δ正极性 (p) | Δ负极性 (p) | Δ轮数 (p) |
|---|---|---|---|
| cogsim | −0.73（<.001） | +0.36（<.001） | +2.8（.126 ns） |
| 提示式四档 | −0.83~−1.02（均<.001） | +0.39~+0.54（均<.001） | +0.3~+1.5（基本 ns） |

**JS 散度**（风格 vs 池化）：judge 结局 JS 上 cogsim 最低（0.060），提示式 0.069-0.091；极性 JS cogsim 0.022 也最低。

**PERMANOVA**（轨迹特征向量）：五档全部不显著（n=10 种子仍功效不足）。

### 3.5 重要修正（相对 n=9 小样本版本）

小样本（3 种子）时"pushy 下 Cog-Sim 对话显著拉长（p=.022）"在扩种到 10 种子后**不再显著**（+2.8 轮，p=.126）——之前的小样本显著性是波动，已如实修正。稳健的结论是：

1. **极性层的风格反应是五档共性且高度显著**（p<.001 量级）——"施压→反感"不依赖认知建模
2. **Cog-Sim 保留方向合理的状态层反应**（valence pushy 0.097 vs 其他 0.30-0.38）
3. **描述性差异**：Cog-Sim pushy 对话更长（10.1 vs 5.7-6.1 轮）、全风格轮数普遍更长（用户坚持度更高）
4. 轨迹级主张仍无法用当前设计检验（PERMANOVA 不显著；严格信噪比需重跑设计）

## 三.6 LLM judge 评估（flash，2026-09-15 追加）

### 3.6.1 五档行为相似度（有效结果）

同一 (任务,种子,风格) 下五档模拟续写两两配对（120 对/组 × 120 组），flash 判用户行为相似度（0-1）：

| 配对 | 相似度 |
|---|---|
| cogsim ~ std_roleplay | 0.264 |
| cogsim ~ std_persona | 0.275 |
| cogsim ~ std_persona_resist | 0.311 |
| cogsim ~ std_bdi | 0.311 |
| **提示式四档内部** | **0.558-0.669** |

**发现**：提示式四档（roleplay/persona/persona_resist/bdi）之间的行为高度同质（0.56-0.67）——加 persona 事实、加"不轻易同意"指令、加 BDI 文本，在行为空间里都只是表面扰动；Cog-Sim 与任何一档都只有 0.26-0.31 的相似度——**认知建模产生了实质不同的行为模式，而非提示工程的同质冗余**。这是"显式建模 vs 提示工程"存在实质差异的直接证据。

### 3.6.2 漂移/僵化（结果不可用，天花板效应）

flash 按 P2 rubric 判 600 条的漂移/僵化（0-2），五档几乎全为 0（drift 均值 0.00-0.07、rigidity 0.00-0.17）。**判定无效**：同模型 judge 对同类生成输出系统性宽容（与 judge 校准时发现的 E0A0 崩坏相反方向的偏差），且无锚点样本时 0-2 粗粒度 rubric 无法触发。**不能据此宣称任何一档无漂移**。修复方向：负向提问（"找出最可疑的一处无依据改变"）、锚点样本校准、或人工抽检。此指标留待正式 P2 盲评。

产物：`data/diversity_probe/llm_judge_eval.json`（per_dialogue + pairwise_similarity + agg）

## 四、限制与下一步

- 限制：单次运行；同模型自我对话（agent 与用户均为 flash）；自动指标（flash 判 flash）未人工校验；persona/BDI 与 2 回合可见历史存在超前偏差；无重复采样、无受控干预
- 下一步：① 真人续写匹配对照（provenance→data/raw，离线）；② pro-judge 漂移/僵化评估（LLM 版盲评，合理性评估口径）；③ 审计合规检查；④ 人工盲评（正式 P2）

## 附：执行记录

- 批次 1：180 条（3 种子，--judger），180 workers，一次完成
- 批次 2：420 条（种子 4-10，--judger），经历 200 workers 疑似限流（后证明为"写完才落盘"的观测假象）→ 64 workers 重启 → 420 workers 全并行 + 逐条落盘 + skip-existing + 单条容错后完成
- 合并：600 条统一 judge/极性/汇总（judge 缺失字段补全模式）
- 成本合计：约 1.4 万调用 / ~14M tokens（生成+judge+极性）
