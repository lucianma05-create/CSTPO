# 轻量化认知状态用户模拟器（MVP 实现）

对应 `shared_work_space/改进文档0911.md` 第 40、43 节的 MVP 范围。
（0911 版对 0910 版的主要修改：Route 决策公式、Route 特征提取合并
Target Extraction、Discrepancy 输出相关节点，见下"0911 对齐"一节。）

## 运行

```bash
cd Cog-Sim
python -m simulator.run_sim --task bargain   --turns 2   # 讨价还价
python -m simulator.run_sim --task donation  --turns 2   # 说服捐赠
python -m simulator.run_sim --task support   --turns 3   # 情感支持
python -m simulator.demo_live_bargain --max-turns 8      # LLM 卖家 vs 认知买家
```

- 模型：默认 `deepseek-flash`（DeepSeek API，key 从根目录 `.env` 读取），
  可用 `--model deepseek-v4-pro` 切换。
- Route 采样模式（文档 §17、§17.1）：默认 `deterministic`（p_C>=0.5 取 Central，
  可复现），`--route-mode bernoulli` 保留随机性（训练 rollout 用）。
- 日志：写入 `runs/<task>_log.jsonl`，含每轮 mode(+reason) / route 三特征 /
  p_C / judgment / target / relevant_state / BDI 前后快照 / appraisal / emotion /
  reaction_plan / updater 约束审计。

## 每轮流程（文档 §39）

```
a_t -> Mode(§8,§9) -> [Influence]
    Route 特征提取(§13,§14: relevance/argument/cue/交互压力 + target_proposition)
    -> Route 决策(§16,§17: Central=eta_R*Rel*Arg, Peripheral=(1-eta_R)*Cue,
       归一化 p_C；弱信号回退 §17.2)
    -> Discrepancy(§18,§19: stance_distance + relevant_current_state)
    -> Judgment(§20: tau 阈值)
    -> Contract(§22-27) -> Engine 提案(§28-30, 显式 route/judgment)
    -> Deterministic Updater(§31) -> Appraisal(§32) -> Emotion(§33 v̂ 公式,
       §34 arousal 公式) -> NLG(§37,§38)
Elicit/Social: BDI 冻结（§35,§36），只更新情绪并生成回复
    （Elicit 记录 revealed_items）。
```

## 对话终止（user_done，任务中立）

- 模拟器每轮对用户话语做一次"是否想结束本次对话"的分类
  （farewell / 离开 / 拒绝继续交谈），结果在 `log.user_done`，
  置真后 `sim.conversation_ended = True`，驱动方应停止追问。
- `user_done != task_done`（文档 §42）："成交/同意"不是结束信号，
  任务结果由外部 Task Evaluator 读 z_t 判定，模拟器不输出。
- 驱动层仍需自行设置回合上限（如 RL 的 max_steps）。

## 模块（文档 §40）

| 模块 | 文件 | 职责 |
|---|---|---|
| state | `state/schema.py` `state/updater.py` | BDI/Emotion/Appraisal 数据结构；确定性约束（§31） |
| profile | `profile/cognitive_profile.py` | 等级映射、τ 阈值、习惯卡片编译（§7） |
| routing | `routing/mode_classifier.py` | Influence/Elicit/Social 分类 + reason（§8,§9） |
| routing | `routing/route_features.py` | 三特征 + 交互压力 + target_proposition 一次提取（§13,§14,§34） |
| routing | `routing/route_controller.py` | CentralScore/PeripheralScore/p_C/采样（§16,§17） |
| routing | `routing/discrepancy.py` | 立场距离 + relevant_current_state（§18,§19） |
| routing | `routing/judgment_controller.py` | d_t + τ 阈值 -> Accept/Noncommit/Reject（§20） |
| cognitive | `cognitive/rj_contract.py` `cognitive/cognitive_engine.py` | RJ 六种合约（§22-27）；认知更新提案（§28-30） |
| affect | `affect/appraisal.py` `affect/emotion.py` | 钳制 + §32.1 GC 公式重算（desire_assessment）；v̂=(GC+CP+FE)/3、r̂=γ1·\|ΔC\|+γ2·\|ΔGC\|+γ3·压力（§34）+ 惯性 + category 按 (v,r) 校验 |
| generation | `generation/user_response.py` `generation/conversation_end.py` | reaction_plan → utterance（§37,§38）；user_done 分类 |

## 0911 对齐说明

1. **Route 公式**（§16、§17）：`Central=η_R·Rel·Arg`、`Peripheral=(1-η_R)·Cue`、
   `p_C=Central/(Central+Peripheral+ε)`；`Central+Peripheral<0.05` 时回退 `p_C=η_R`
   （§17.2）；`deterministic`（p_C>=0.5，§17.1）/ `bernoulli`（§17）双模式。
2. **Route 特征**（§13、§14）：不再单独维护 motivation/ability/processing_demand，
   只保留 relevance / argument_strength / cue_strength 三特征，并合并输出
   `target_proposition`，Discrepancy 直接使用（§19），不再重复提取。
   2026-09-12 增加 `interaction_pressure` 标注（§34 arousal 公式的语义分量）。
3. **Discrepancy**（§19）：输出 `relevant_current_state`（B/D/I 节点 id，
   程序过滤幻觉 id），替代旧版只输出 belief id + support_quality
   （support_quality 在 0911 文档中已移除，故删除）。
4. **模块归位**（§40）：route 相关控制器从 `cognitive/` 移至 `routing/`
   （route_features / route_controller / discrepancy / judgment_controller）。
5. **Mode 分类**（§9）：按文档 prompt 返回 `{mode, reason}`，reason 入日志。
6. **Engine 输入**（§28）：prompt 显式包含 Processing route 与 Judgment
   （旧版仅隐含在 contract 中）。

## 与文档的少量偏差/补白

1. 模型 ID：API 实际暴露 `deepseek-flash` / `deepseek-v4-pro`，无字面 "4.1-flash"。
2. Arousal 目标公式（§34，2026-09-12 实施）：`r̂ = γ1·|ΔC| + γ2·|ΔGC| + γ3·压力`
   由程序计算——|ΔC| 为 Updater 约束后 BDI 快照差的 Σ|Δs_i|/4（归一化），
   |ΔGC| 为本轮 GC 相对上一轮的突变（首轮为 0），压力为 LLM 语义标注
   `interaction_pressure`（low/medium/high→0.2/0.5/0.8，定义严格任务中立：
   只衡量对即时回应的需求，与说服力/话题重要性/用户情绪解耦）。
   第一版 γ=(0.6, 0.25, 0.5)（文档未给具体值）。原"Engine 提议 r_hat"方案已移除。
3. §31 的 ε 文档未给值，取 0.5；"相关强 Belief/Desire" 强度门槛取 2.0。
4. Elicit/Social 分支用一次合并调用产出 appraisal+emotion+utterance
   （文档 §39 伪代码对该分支的 appraisal 来源未定义）。2026-09-11 修复：
   合并调用增加 Assistant's latest reply 输入（旧版看不到 agent 本轮消息，
   Elicit 下用户答非所问，Social 下不回应告别等消息）。
5. 对话终止：文档未定义；按 §42 原则实现为任务中立的 user_done 分类
   （每轮多一次 ~100 token 的小调用，LLM 失败时退回规则关键词匹配）。
6. BDI 节点扩展（评审 0910 文档时发现的问题，已修复）：
   - `polarity: approach|avoid`（Desire/Intention 的趋近/回避极性，文档 §3.1
     只写了"希望实现或避免"但表示法无方向字段；为 §32.1 的 gc_i 程序化计算打基础）；
   - 非核心节点强度 < 0.5 自动退役（inactive），强度回升可重新激活（文档 §3.2 的
     数量限制未给具体退役条件）；
   - 新增 Belief 可标 `conflicts_with`，程序对冲突的已有 Belief 衰减，避免同时
     强持有 P 和 ¬P。2026-09-11 收紧（旧版按新节点全强度扣减，Central+Noncommit
     限幅 0.4 时一轮 B1 仍可跌 2.5，等于绕过 RJ 幅度约束）：
     (a) 新增节点强度一律以本回合 `limits[type_]` 为上限，`core=False` 不再豁免；
         唯一豁免是 Peripheral+Accept 的 cue 信念（文档 §25 明确允许
         trust/authority/social norm 增强）；
     (b) 冲突衰减计入 victim 本轮变化预算：victim 的直接更新 + 冲突衰减的净变化
         不超过本回合 belief 限幅，矛盾在多轮内渐进消解；
     (c) 新信念过弱已退役（inactive）时不产生冲突衰减压力（用户几乎未接收该命题）。
     代价：Central+Accept 下新增 Belief 首轮上限 1.0（文档示例 2.5 需多轮增强）。
7. Appraisal 的 GC 半程序化（§32.1，2026-09-11 实施）：LLM 输出逐 active
   Desire 的 `(relevance, gc_i)` 语义标注（`desire_assessment`），程序按
   `GC = Σs_i·rel_i·gc_i / Σs_i·rel_i` 重算，ActiveGoals 取 TopK(s_i·rel_i)
   （§4，K=2），权重用 Updater 约束**后**的 Desire 强度（同时修复提案被截断时
   GC 与实际状态脱节的时点问题）；标注缺失/非法时退回 LLM 提议的 GC 值。
   CP/FE 仍走 LLM 提议（文档未给公式）。
8. Emotion 的 category 一致性校验（0910 §16「category 根据 (v,r) 选择」，2026-09-11 实施）：
   LLM 提议的 category 与程序最终 (v,r) 矛盾时（焦虑带正效价、anger 低唤醒等），
   程序改写为由 (v,r) 推导的类别（neutral/sadness/anxiety/frustration/interest/
   satisfaction）并记审计 note；hope/relief 等依赖评价内容的类别在数值一致时保留。
   `emotion_proposal` 现在只含 `category`：valence 由 §33 公式从 appraisal 推导
   （2026-09-11），arousal 由 §34 公式计算（2026-09-12），LLM 提议的
   valence/arousal 字段均已删除。
9. `evaluation/` 四个模块（文档 §40）与 §44 实验验证暂未实现，属下一阶段。
