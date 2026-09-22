"""baseline：主动对话基线统一复现包。

边界约定（用户裁定 2026-09-20）：
- 所有代码与运行产物只在 baseline/ 内开展与写入（runs/）；
- 外部环境只读引用：cstpo/（judge rev4、TaskEnv、种子）与 Cog-Sim/（模拟器、
  LLMClient），不修改任何外部文件；
- 评估集 = data/judge_calibration/heldout/ 的 30 个冻结留出 case
  （p4g 5 + craigslistbargain 5 + esconv 20），完整种子按 seed_id 从
  data/seeds_draft/ 解析（case 内置 seed 仅 {seed_id, situation, emotion} 摘要）；
- 底座 = deepseek-flash（LLMClient 默认，temperature 硬编码 0 = greedy）；
- 用户模拟 = Cog-Sim v1.0.1-fix（TaskEnv），裁判 = 我方 judge rev4（n=3 聚合）。
"""
