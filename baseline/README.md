# baseline：主动对话基线统一复现

在统一口径下复现三任务（P4G 劝说捐赠 / CraigslistBargain 讨价还价 / ESConv 情感支持）的主动对话基线，与主方法 CSTPO 直接可比。

## 锁定口径（2026-09-20 用户裁定，free100 版）

评估集：data/seeds_test/<task>/ 的每任务 100 个 test 种子（t001–t100，共 300 case），冻结于 baseline/eval_set.json（含每文件 sha256，加载时校验），只读。注：data/judge_calibration/heldout/ 的 30 case 保留为带人工标注的参考集，不进主评估。

底座：提示族用 deepseek-flash（LLMClient 默认，temperature 硬编码 0 = greedy）；SFT 族只读加载现有 labeled-SFT 检查点（/publicdata/model/CSTPO/*），不训练任何新权重（当前暂缓，用户裁定）。

用户模拟：Cog-Sim v1.0.1-fix（TaskEnv，deterministic route）。裁判：我方 judge rev4（judge_with_aggregation，n=3 聚合），与主方法完全相同；CB 奖励的 raw_SL 用种子 target 程序计算。

对话协议：自由交互（2026-09-20 用户裁定，替代固定 12 轮）——第 13 轮起每 2 轮 judge 判一次冗余，判冗余则追加用户口吻结束语并终止（judger_stale_end）；30 轮安全上限（time_limit）。终止形态三类：user_ended / judger_stale_end / time_limit。

后处理：所有策略输出统一 strip_role_prefix（与 RL 评估口径一致）。

边界：baseline/ 只读引用外部（cstpo/、Cog-Sim/、data/、/publicdata/model/），不修改外部任何文件；全部产物只写 baseline/runs/<tag>/<method>/。

账本：每个 case 独立 LLMClient，按 simulator / actor / judge / stale_judger 四分量记账（calls + prompt/completion tokens），manifest 记录源码 sha256、种子 sha256、judge 版本、协议与生成配置。

## 用法

```bash
cd /data/user21300120/mmh/CSTPO
# 冻结评估集（每任务 N 个 test 种子）
python -m baseline.freeze_eval_set --n 100
# 全量 300 case，自由交互协议
python -m baseline.runner --method standard --eval-set baseline/eval_set.json \
  --tag free100 --max-turns 30 --stale-start 13 --workers 20
# 成对比较（cluster bootstrap，按 case 聚类）
python -m baseline.stats --methods standard,proactive,procot,ane,mi_prompt --tag free100
```

产物：baseline/runs/<tag>/<method>/manifest.json、<task>/<dialogue_id>.json（逐对话记录：dialogue、verdict、reward、四分量 token 成本）、summary.json、comparison.json（跨方法 bootstrap）。断点续跑：已存在的 case 记录自动跳过，--force 重跑。

## 方法清单（对应论文 baseline 矩阵）

| 方法 | 状态 | 说明 |
|---|---|---|
| standard | ✅ free100 | 香草提示直生成（PPDPP Table 8 / TRIP Table 20 / ESConv-SRA template6） |
| proactive | ✅ free100 | Deng 2023a a→r：先选策略（canonical 词表）再回应 |
| procot | ✅ free100 | Deng 2023a t→a→r：分析+目标+策略+回应（模板取自 vendor/LLM-Proactive） |
| ane | ✅ free100 | Ask-an-Expert（Zhang 2023b）：先写专家建议再回应 |
| mi_prompt | ✅ free100 | Chen 2023 混合主动策略提示：选标签→译为指令→按指令生成（单步链） |
| sft-labeled | 暂缓 | 现有 labeled-SFT 检查点，greedy 三件套推理（代码已接入，用户裁定暂缓） |
| dialogxpert | ✅ free100 | Q 网络（冻结 BERT + MLP，360 episodes 训练）训练/推理均在本目录内（vendor/dialogxpert） |
| ppdpp | ❌ 不复现 | 与 DialogXpert 同属离散策略训练族（credit 仅落标签选择），已覆盖 |
| astro | ❌ 不复现 | 无可用代码；同族离散策略 planner，重实现成本高，仅引用 |

结果文档：baseline/runs/free100/REPORT.md（flash 骨干全量表，当前有效版本）；baseline/runs/qwen14b/REPORT.md（Qwen3-14B vLLM 骨干重测 + 双骨干对比，含骨干依赖结论）。

与原仓库的适配偏差（已记录，论文中说明）：原仓库（LLM-Proactive）把完整 CoT 输出直接当回复喂给用户模拟器；本实现只提取最终回应段（"the response is" 之后）作为话语，避免元文本污染 Cog-Sim 对话与 judge 判分。策略词表统一用本仓库三任务 canonical 标签（label_maps），不用原仓库的 CB 卖方策略集。

统计约定：以 case/用户为聚类单位做 cluster bootstrap 成对比较（stats.py，2000 次，95% CI，* = p<0.05）；小样本区间宽时记录 inconclusive，不把点估计当支持证据。P4G 捐助金额为模拟承诺金额（judge amount 非空众数，解析规则见 free100 报告），不是实际支付。
