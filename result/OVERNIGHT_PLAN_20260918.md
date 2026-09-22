# 夜间自动流水线计划（2026-09-18 深夜，用户入睡期间执行）

## 判据与门槛（预注册，自动应用）

1. 训练完成：`train_runtime` 出现或进程退出（异常则记录日志，不重跑）
2. 标签探针（`scripts/probe_labels_v2.py`，每任务 5 种子 × 6 轮 greedy）：
   合法率 ≥50% 且最大标签占比 ≤80% = PASS；FAIL 只记录不调参
3. A/B 验证（`ab_validate.py`，`AB_CKPT=/publicdata/model/CSTPO_v2`）：
   CB 3/5>1/5、p4g 4/5=4/5、esconv G≥0.494——同 v1 门槛
4. esconv GRPO + 评估：仅在 esconv A/B 通过时执行
   （`mc_ppo_run.sh`，`MODEL_PATH=/publicdata/model/CSTPO_v2/esconv`，
   20 steps、free_cache=false；评估 `eval_rl_vs_sft.py` 同口径）
5. 任何阶段失败：停止后续，把状态写进本文件与记忆，等待用户决策

## 产出位置

- 训练日志：logs/sft_v2_{esconv,p4g,cb}.log
- 探针/A/B 输出：logs/probe_v2_<task>.log、logs/ab_v2_<task>.log
- GRPO：logs/mc_ppo_v2.log + logs/rollout_traj_v2/
- 汇总报告：result/V2_VERDICT.md（醒来即读）

## 不自动做的事（等待用户决策）

- epochs/混合比例/香草标签均衡等调参
- p4g/cb 的 RL（仅 esconv 在范围内）
- verl issue 提交
