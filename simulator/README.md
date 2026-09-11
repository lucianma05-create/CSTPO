# 轻量化认知状态用户模拟器（MVP 实现）

对应 `shared_work_space/开发文档0910.md` 第 23、24、26 节的 MVP 范围。

## 运行

```bash
cd Cog-Sim
python -m simulator.run_sim --task bargain   --turns 2   # 讨价还价
python -m simulator.run_sim --task donation  --turns 2   # 说服捐赠
python -m simulator.run_sim --task support   --turns 3   # 情感支持
```

- 模型：默认 `deepseek-flash`（DeepSeek API，key 从根目录 `.env` 读取），
  可用 `--model deepseek-v4-pro` 切换。
- 日志：写入 `runs/<task>_log.jsonl`，含每轮 mode / route / judgment /
  BDI 前后快照 / appraisal / emotion / reaction_plan / updater 约束审计。

## 每轮流程（文档 §24）

```
a_t -> Mode(§6) -> [Influence] RJ(§9,§10) -> Contract(§11)
    -> CognitiveEngine 提案(§12,§13) -> Deterministic Updater 约束(§14)
    -> Appraisal(§15) -> Emotion 惯性更新(§16) -> NLG(§18,§19)
    -> user_done 分类（任务中立，见下）
Elicit/Social: BDI 冻结，只更新情绪并生成回复（Elicit 记录 revealed_items）。
```

## 对话终止（user_done，任务中立）

- 模拟器每轮对用户话语做一次"是否想结束本次对话"的分类
  （farewell / 离开 / 拒绝继续交谈），结果在 `log.user_done`，
  置真后 `sim.conversation_ended = True`，驱动方应停止追问。
- `user_done != task_done`（文档 §25）："成交/同意"不是结束信号，
  任务结果由外部 Task Evaluator 读 z_t 判定，模拟器不输出。
- 驱动层仍需自行设置回合上限（如 RL 的 max_steps）。

## 模块（文档 §23）

| 模块 | 文件 | 职责 |
|---|---|---|
| state | `state/schema.py` `state/updater.py` | BDI/Emotion/Appraisal 数据结构；确定性约束 |
| profile | `profile/cognitive_profile.py` | η_R 路线公式、τ 阈值、习惯卡片编译 |
| routing | `routing/mode_classifier.py` | Influence/Elicit/Social 分类 |
| cognitive | `cognitive/route_controller.py` `judgment_controller.py` `rj_contract.py` `cognitive_engine.py` | RJ 门控 + 认知更新提案 |
| affect | `affect/appraisal.py` `affect/emotion.py` | 归一化；v̂=(GC+CP+FE)/3 + 惯性 |
| generation | `generation/user_response.py` `generation/conversation_end.py` | reaction_plan → utterance；user_done 分类 |

## 与文档的少量偏差/补白

1. 模型 ID：API 实际暴露 `deepseek-flash` / `deepseek-v4-pro`，无字面 "4.1-flash"。
2. Route 控制器额外输出 `goal_relevance` 用于计算 Mot_t（文档公式需要，§4.1）。
3. Judgment 控制器额外输出 `related_belief_ids`，供 Updater 执行 §14.3 的
   "Reject 方向"检查（否则程序无法定位被拒命题对应的 Belief）。
4. §14.5 的 ε 文档未给值，取 0.5；"相关强 Belief/Desire" 强度门槛取 2.0。
5. Arousal 目标公式文档未给出（§16），第一版由 LLM 提议 r_hat + 惯性。
6. Elicit/Social 分支用一次合并调用产出 appraisal+emotion+utterance
   （文档 §24 伪代码对该分支的 appraisal 来源未定义）。
7. `support_quality` 目前只记录不参与动力学（文档 §10 提到但未给用法）。
8. 对话终止：文档未定义；按 §25 原则实现为任务中立的 user_done 分类
   （每轮多一次 ~100 token 的小调用，LLM 失败时退回规则关键词匹配）。
9. BDI 节点扩展（评审文档时发现的问题，已修复）：
   - `polarity: approach|avoid`（Desire/Intention 的趋近/回避极性，文档 §3.1
     只写了"希望实现或避免"但表示法无方向字段；为 §15.1 的 gc_i 程序化计算打基础）；
   - 非核心节点强度 < 0.5 自动退役（inactive），强度回升可重新激活（文档 §3.1 的
     deactivate 提法未给具体条件）；
   - 新增 Belief 可标 `conflicts_with`，程序对冲突的已有 Belief 按新节点强度衰减，
     避免同时强持有 P 和 ¬P。
10. `evaluation/` 四个模块（文档 §23）与 §27 实验验证暂未实现，属下一阶段。
