"""DialogXpert 本地化复现（M3）。

方法来源：DialogXpert（AAAI-26，vendor/dialogxpert 快照 commit 7b35ec9）。
本地化口径（用户裁定 + 记录为论文偏差）：
- 候选动作从策略标签改为完整话语（Cog-Sim 需要话语；Q 网络价值选择本质不变）；
- 用户模拟 = Cog-Sim v1.0.1-fix，奖励 = judge rev4（终局-only → TD 目标退化为终局 G）；
- 训练/评估同自由交互协议（13 轮起判冗余、30 轮上限）；
- BERT 截断 bug 修复（左截断 512，同时截 input_ids/attention_mask）；
- lr 1e-3（vendor 代码默认；论文附录 1e-6 仅记录不一致）。
"""
