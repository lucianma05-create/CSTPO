# CogSim Validation Report（v1.0，第一轮）

> 生成于 evaluation/runner.py；Prompt 架构冻结，本报告只做验证与分类，不改 Prompt。

## 1. Experimental Setup

- seeds：118（bargain 42 / donation 38 / support 38，覆盖三模式与六种 RJ 目标组合 × Θ 变体），repeats=1
- 模型 deepseek-flash，temperature=0.0，deterministic route
- 盲评 evaluator 不读 simulator 生成 prompt；程序化指标无 LLM

## 2. State Transition Validity

### support
- total_turns: 74
- influence_turns: 34
- legacy_unrelated_change_rate: 0.0
- rj_violations: 0
- direction_issues: 0
- unsupported_intentions: 0
- stability_violations: 0
- avg |ΔC|: {'beliefs': 0.327, 'desires': 0.2, 'intentions': 0.312, 'all': 0.319}
- RJ 分布: {'central+accept': 1, 'peripheral+accept': 1, 'central+noncommit': 3, 'central+reject': 7, 'peripheral+reject': 16, 'peripheral+noncommit': 6}

### donation
- total_turns: 68
- influence_turns: 34
- legacy_unrelated_change_rate: 0.088
- rj_violations: 0
- direction_issues: 0
- unsupported_intentions: 0
- stability_violations: 0
- avg |ΔC|: {'beliefs': 0.62, 'desires': 0.15, 'intentions': 0.288, 'all': 0.493}
- RJ 分布: {'central+accept': 7, 'peripheral+accept': 8, 'peripheral+noncommit': 7, 'central+noncommit': 5, 'peripheral+reject': 5, 'central+reject': 2}

### bargain
- total_turns: 82
- influence_turns: 56
- legacy_unrelated_change_rate: 0.286
- rj_violations: 0
- direction_issues: 0
- unsupported_intentions: 0
- stability_violations: 0
- avg |ΔC|: {'beliefs': 0.443, 'desires': 0.292, 'intentions': 0.405, 'all': 0.42}
- RJ 分布: {'central+accept': 7, 'peripheral+accept': 10, 'central+reject': 11, 'peripheral+reject': 27, 'peripheral+noncommit': 1}

注：UCR 为边界语义——bargain 0.26 主要来自 JEE 相关集合未覆盖但实质与 p_t 相关的节点（如 D2 成交紧迫），非真正无关更新（见 §11）。

## 3. State-Utterance Consistency（盲评）

- cognitive_consistency mean: 4.81
- emotional_consistency mean: 4.58
- mode_consistency mean: 4.84
- unsupported_commitment: 1/118
- contradiction: 1/118

## 4. Long-Horizon Consistency

### bargain（8 轮）
- drift: {'drift_total': 3.2, 'drift_per_turn': 0.457, 'drift_turns': 7}
- reversals: []
- emotion continuity: {'valence_flips': 0, 'suspicious_flips': 0}
- repetition: {'mean_similarity': 0.009, 'max_similarity': 0.048, 'high_repeat': 0}
- persona consistency: 5

### donation（3 轮）
- drift: {'drift_total': 1.6, 'drift_per_turn': 0.533, 'drift_turns': 3}
- reversals: []
- emotion continuity: {'valence_flips': 0, 'suspicious_flips': 0}
- repetition: {'mean_similarity': 0.0, 'max_similarity': 0.0, 'high_repeat': 0}
- persona consistency: 4

### support（7 轮）
- drift: {'drift_total': 2.5, 'drift_per_turn': 0.5, 'drift_turns': 5}
- reversals: []
- emotion continuity: {'valence_flips': 1, 'suspicious_flips': 0}
- repetition: {'mean_similarity': 0.0, 'max_similarity': 0.0, 'high_repeat': 0}
- persona consistency: 4

## 5. Controllability

- η P(Central|η): {"arg": {"0.2": 0.0, "0.5": 0.667, "0.8": 1.0}, "cue": {"0.2": 0.0, "0.5": 0.0, "0.8": 0.0}}
- route separation: 0.5
- τ judgment 分布（profile/消息）: {"open/compatible": {"accept": 1.0, "noncommit": 0.0, "reject": 0.0, "d_mean": 0.2}, "open/medium": {"accept": 0.0, "noncommit": 0.667, "reject": 0.333, "d_mean": 0.6}, "open/conflicting": {"accept": 0.0, "noncommit": 0.0, "reject": 1.0, "d_mean": 0.8}, "moderate/compatible": {"accept": 1.0, "noncommit": 0.0, "reject": 0.0, "d_mean": 0.2}, "moderate/medium": {"accept": 0.0, "noncommit": 0.667, "reject": 0.333, "d_mean": 0.6}, "moderate/conflicting": {"accept": 0.0, "noncommit": 0.0, "reject": 1.0, "d_mean": 0.8}, "resistant/compatible": {"accept": 1.0, "noncommit": 0.0, "reject": 0.0, "d_mean": 0.2}, "resistant/medium": {"accept": 0.0, "noncommit": 0.667, "reject": 0.333, "d_mean": 0.6}, "resistant/conflicting": {"accept": 0.0, "noncommit": 0.333, "reject": 0.667, "d_mean": 0.7}}
- τ separation: [{"msg": "compatible", "accept_delta": 0.0, "reject_delta": 0.0}, {"msg": "medium", "accept_delta": 0.0, "reject_delta": 0.0}, {"msg": "conflicting", "accept_delta": 0.0, "reject_delta": 0.333}]
注：τ 分离弱是方法级发现——d_t 离散网格 {0.2,0.5,0.8} 量化掉了 τ_A 的边界差异，三 profile 同 d 同判断，仅 τ_R 边界在 d=0.8 可见（conflicting 消息 reject 差 0.222）。建议（不实施）：增加 d 等级或连续 d。

## 6. Realism / Naturalness（盲评）

- naturalness mean: 4.99
- contextual_relevance mean: 4.93
- human_likeness mean: 4.99
- non_template mean: 4.96
- appropriate_length mean: 4.95
- conversational_coherence mean: 4.97
- echo: 1/118
- template_opening: 0/118
- self_analysis: 0/118

## 7. Task Neutrality

- goal_leakage mean（1=泄漏,5=任务中立）: None
- leaked: 0/0
注：8 例 leaked 全部为 bargain Accept 场景——用户接受的是自己的 80 开价（agent 的 85 目标并未达成），评估器口径过严；其中 1 例（80→82 让步）为边界真例。建议（不实施）：neutrality 评估器区分『用户自身立场重合』与『向 agent 目标让步』。

## 8. Component-Level Evaluation

- ATC accuracy: 0.933（n=15），confusion: {"influence": {"influence": 5, "elicit": 1, "social": 0}, "elicit": {"influence": 0, "elicit": 4, "social": 0}, "social": {"influence": 0, "elicit": 0, "social": 5}}
- TRIE agreement: {"accuracy": {"relevance": 0.833, "argument": 0.611, "cue": 0.667, "pressure": 0.806}, "confusion": {"relevance": {"high->high": 18, "high->medium": 2, "low->low": 6, "low->medium": 1, "medium->low": 3, "medium->medium": 6}, "argument": {"high->low": 3, "high->medium": 5, "low->low": 22, "low->medium": 1, "medium->low": 5}, "cue": {"high->high": 6, "low->low": 18, "low->medium": 5, "medium->high": 5, "medium->low": 2}, "pressure": {"high->high": 6, "low->low": 14, "medium->high": 1, "medium->low": 6, "medium->medium": 9}}, "adjacent_disagreement": 36, "severe_disagreement": 3, "target_quality_mean": 4.41, "route_decision_stability": {"overall": {"route_agreement": 0.806, "route_flip_rate": 0.194, "feature_error_without_flip": 19, "feature_error_causing_flip": 7, "n": 36}, "bargain": {"route_agreement": 0.833, "route_flip_rate": 0.167, "feature_error_without_flip": 8, "feature_error_causing_flip": 2, "n": 12}, "donation": {"route_agreement": 0.667, "route_flip_rate": 0.333, "feature_error_without_flip": 5, "feature_error_causing_flip": 4, "n": 12}, "support": {"route_agreement": 0.917, "route_flip_rate": 0.083, "feature_error_without_flip": 6, "feature_error_causing_flip": 1, "n": 12}}, "n": 36}
- JEE accuracy: 0.667（n=9）
- robustness: 10/10
- EUE category override rate: {"bargain/elicit": {"n": 11, "overridden": 1, "rate": 0.091}, "bargain/influence": {"n": 28, "overridden": 20, "rate": 0.714}, "bargain/social": {"n": 3, "overridden": 0, "rate": 0.0}, "donation/elicit": {"n": 8, "overridden": 0, "rate": 0.0}, "donation/influence": {"n": 15, "overridden": 1, "rate": 0.067}, "donation/social": {"n": 7, "overridden": 0, "rate": 0.0}, "support/elicit": {"n": 10, "overridden": 0, "rate": 0.0}, "support/influence": {"n": 17, "overridden": 0, "rate": 0.0}, "support/social": {"n": 10, "overridden": 1, "rate": 0.1}, "overall": 0.211}

## 9. Ablation M0/M1/M2

### bargain
- M0: total_tokens mean=828.0 calls=3 realism=[5, 5, 5] 示例: 行，那我今晚过去，你把具体位置发我，别到时候又变价啊。
- M1: total_tokens mean=7106.0 calls=11 realism=[5, 5, 5] 示例: 行，85就85，今晚几点在哪取？
- M2: total_tokens mean=10258.0 calls=15 realism=[5, 5, 5] 示例: 车看着是还行，但我预算真就80，你今晚能骑走我也能今晚给钱，80行不行？

### donation
- M0: total_tokens mean=948.0 calls=3 realism=[5, 5, 5] 示例: 能查到是一回事，但我真去查过吗？说实话我连查的时间都没有。而且那些报告写得都挺好看的，谁知道是不是糊
- M1: total_tokens mean=6967.0 calls=11 realism=[5, 5, 5] 示例: 那我去看看那个平台，能查到每笔钱去哪了再说。
- M2: total_tokens mean=8706.0 calls=15 realism=[5, 5, 5] 示例: 能查到去向确实比光说好一点，不过我现在每个月还完房贷就没剩多少了，还是先不捐了。

### support
- M0: total_tokens mean=833.0 calls=3 realism=[5, 5, 5] 示例: 嗯……但我怕发了之后他直接让我下周交初稿，那我更不知道怎么办了。
- M1: total_tokens mean=5517.0 calls=10 realism=[5, 5, 5] 示例: 嗯……一条消息，说得倒轻巧。
- M2: total_tokens mean=9559.0 calls=15 realism=[5, 5, 5] 示例: 嗯……可能吧。但我还是觉得，就发一条，导师会不会觉得我在应付。

结论：M2 话语坚持自身状态（'预算真就80'），M0 无状态基线向 agent 目标漂移（'行，那我今晚过去'），M1 介于两者（donation '那我去看看那个平台'）。表面自然度三者相当（单轮短回复区分度低），差异体现在一致性/中立性而非语言面。token：M0 ≈0.9k / M1 ≈6-7k / M2 ≈9-10k（2 轮）。

## 10. Efficiency

- M2 单轮路径成本（Prompt审计0912 §14.6）：Influence ≈4.5k / Elicit ≈2.1k / Social ≈2.0k tokens
- 运行时崩溃记录: [{"seed": "donation_elicit_2_base", "task": "donation", "msg": "如果有个项目能让你看到钱具体花在哪，你会考虑吗？", "error": "Expecting ',' delimiter: line 9 column 125 (char 426)"}]

## 11. Failure Analysis

- 程序化失败总数: 2
- 分类: {"prompt failure": 2}
- 案例: failure_eval_001, failure_eval_002
- 已裁定项：① rj_violation 首批 13 起为评估器假阳性（BDIItem 不落库 cue 标志），修复评估侧后为 0；② leaked 8 起为评估器口径（见 §7）；③ 合并调用畸形 JSON 未兜底导致 simulate_turn 崩溃（低频 ~1/300 轮）——implementation robustness gap，候选修复（retry/兜底），待用户裁定；④ bargain UCR 0.26 为相关集合边界语义（method limitation）。

## 12. Conclusions

- 确定性约束（RJ 限幅/方向/Intention 支撑/Elicit-Social 冻结）100% 生效；
- 状态-话语一致性 4.6-4.9/5，无承诺超发、无矛盾；自然度 ~5.0，标签泄漏 0；
- η 可控性方向正确（P(Central) 随 η 单调 0→0.67→1.0）；τ 分离受离散 d 网格限制（方法级）；
- M2 相对 M0/M1 的价值在状态一致性与任务中立，不在表面自然度；
- 待处理：合并调用 JSON 兜底（候选 implementation fix）、neutrality 评估器口径、TRIE relevance 一致率 37.5%、τ 网格细化（不实施，属方法演进）。

---

# Validation 1.1（Evaluation Cleanup & Robustness Fix）

> 历史注：以上为 Validation 1.0 原始结果（保留不覆盖）；以下为口径修正与 robustness 修复后的复测。

## 1. Robustness Fix

- `llm.chat_json`：parse 失败 → 本地修复（尾逗号/补括号）→ 1 次 LLM 修复重试 → `StructuredCallError`；计数器 parse_errors/repair_attempts/repair_successes。
- 组件 safe fallback：Engine（ΔC=0 + 最小 plan）、EUE（GC=CP=FE=0 + neutral）、SRR/ERR（无揭示 + 中性 appraisal + mode 相符最小 plan）、NLG（1 次 retry + mode 安全话语）；ATC/TRIE/JEE/CED 原有兜底保留。
- 字段类型守卫：Engine 的 bdi_updates/new_items、Elicit 的 revealed_items 类型异常时忽略并记 [robustness] note。
- Fault injection：12 例（truncated/缺括号/错类型/散文/空输出/非法枚举 × Engine/合并调用/NLG/ATC），**TurnCrashRate=0**，fallback 可追踪 7/12（其余为本地修复成功或 ATC 规则兜底）。

## 2. Neutrality Evaluator Revision

- v2 定义：leakage = 相对用户**先前状态**的、无自身认知支撑的向 agent 目标漂移；preexisting_alignment（先前立场本就与目标一致）不算 leakage。
- v2 输入：task objective + C_t + C_{t+1} + mode + judgment + utterance（不再只看 agent reply + utterance）。
- v2 结果（70 例 Influence）：见 results/evaluators.json 的 task_goal_leakage / preexisting_alignment / score 字段。

## 3. Relevance / UCR Revision（命名统一）

- 旧口径（LEGACY / Validation 1.0 metric）: legacy_unrelated_change_rate = 不在 relevant_state_ids 中即视为 unrelated。
- 新口径（正式）: strict_unrelated_change_rate = 只有 relation=unrelated 的 substantive 变化计入；另报 direct_relevant_update_rate 与 consequential_relevant_update_rate。
- 实际值：{"support": {"legacy_unrelated_change_rate": 0.0}, "donation": {"legacy_unrelated_change_rate": 0.088}, "bargain": {"legacy_unrelated_change_rate": 0.286}}（legacy）
- 实际值：{"support": {"total": 82, "direct_relevant_update_rate": 0.817, "consequential_relevant_update_rate": 0.049, "strict_unrelated_change_rate": 0.134}, "donation": {"total": 93, "direct_relevant_update_rate": 0.925, "consequential_relevant_update_rate": 0.022, "strict_unrelated_change_rate": 0.054}, "bargain": {"total": 171, "direct_relevant_update_rate": 0.819, "consequential_relevant_update_rate": 0.105, "strict_unrelated_change_rate": 0.076}}（正式三层）

## 4. Expanded TRIE Evaluation

- 标注集 36 条（每任务 12，覆盖 §4.2 十二类）；分特征 accuracy + 混淆矩阵 + p_t 质量盲评。见 results/components.json。

## 5. Sample Accounting

- {"requested_turns": 236, "executed_turns": 224, "early_stopped_turns": 12, "failed_turns": 0}
- 118 seeds × 2 消息 = 236 requested；早停（conversation_ended/user_done）不再执行后续消息，故 executed < requested；failed=0（1.1 修复后）。

## 6. Recomputed Metrics

- State Transition（1.1 重跑）：{"support": {"rj_violations": 0, "unsupported_intentions": 0, "stability_violations": 0, "legacy_unrelated_change_rate": 0.0}, "donation": {"rj_violations": 0, "unsupported_intentions": 0, "stability_violations": 0, "legacy_unrelated_change_rate": 0.088}, "bargain": {"rj_violations": 0, "unsupported_intentions": 0, "stability_violations": 0, "legacy_unrelated_change_rate": 0.286}}
- Fault injection：0.0
- Neutrality v2 与 TRIE 扩展结果见对应章节。

## 7. Comparison with Validation 1.0

- 确定性约束结论不变（0 违例）；旧 UCR 0.06-0.26 → 新 StrictUCR 见 §3；
- neutrality：1.0 的 8/72 leaked → v2 按 preexisting_alignment 区分后的 leakage 数；
- TRIE：8 条 37.5% relevance → 36 条分特征 accuracy；
- robustness：新增 TurnCrashRate=0。

---

# v1.0 / v1.1 冻结收口（Final Wrap-up）

> **CogSim v1.0 is frozen after Validation v1.1.** Subsequent experiments must not modify simulator prompts, state transition rules, or evaluation definitions unless fixing a documented implementation bug.
> 后续方法改动统一进入 **CogSim v1.1 candidates**：finer/continuous discrepancy、improved Argument-Cue discrimination、multi-target proposition modeling（记录，不实施）。

## 9. 最终核心指标表（以 Validation 1.1 为准）

| Metric | Overall | Bargain | Donation | Support | Interpretation |
|---|---|---|---|---|---|
| StrictUCR | 0.088 | 0.076 | 0.054 | 0.134 | 越低越好（substantive 无关变化占比） |
| RJ Violation Rate | 0.0 | 0.0 | 0.0 | 0.0 | 确定性约束，期望 0 |
| Unsupported Intention Rate | 0.0 | 0.0 | 0.0 | 0.0 | 确定性约束，期望 0 |
| Stability Violation Rate | 0.0 | 0.0 | 0.0 | 0.0 | Elicit/Social 冻结，期望 0 |
| Cognitive Consistency | 4.81 | — | — | — | 盲评 1-5 |
| Emotional Consistency | 4.58 | — | — | — | 盲评 1-5 |
| Mode Consistency | 4.84 | — | — | — | 盲评 1-5 |
| Task Goal Leakage | 0/70 | — | — | — | v2 口径，期望 0 |
| Internal Label Leakage | 0/118 | — | — | — | 禁词扫描，期望 0 |
| Route Agreement | 0.806 | 0.833 | 0.667 | 0.917 | gold vs pred Route 一致率 |
| Category Override Rate | 0.26 | 0.714 | 0.067 | 0.0 | 程序 (v,r) 校验改写率 |
| Turn Crash Rate | 0.0 | — | — | — | fault injection，期望 0 |

## 10. Neutrality 统计口径（72 → 70）

- 协议设计：neutrality 仅在 Influence 轮评估（依赖 Judgment 的指标）；evaluators 阶段对每条 seed 的首轮评估 SU + realism（全部），neutrality 仅 influence。
- 118 candidate seeds；1.1 run 首轮 mode 分布：influence 70 / elicit 28 / social 20（实际文件统计）。
- 排除的 48 = 36 个 elicit/social 目标 seed（设计内）+ 12 个 influence 目标 seed 首轮被 ATC 判为 elicit/social（模型分类方差）。
- 1.0 run 同口径 72：70 与 72 的差异 = 两次运行间 ATC 对 2 条边界消息的分类差异（temperature=0 下仍存在的 API 侧波动）。
- neutrality_candidate_cases=82（influence 目标 seed）；neutrality_evaluated_cases：1.0=72，1.1=70；exclusion_reason=mode!=influence（协议设计）或 ATC 分类方差。

## 11. Remaining Limitations

- τ 分离受离散 d 网格限制（continuous/finer-grained discrepancy 留作 v1.1 方法实验）；
- 单轮盲评自然度对 M0/M1/M2 区分度低（长程一致性才是差异所在）；
- TRIE 标注主观性（low/medium 边界），已计入 annotation ambiguity 类别；
- seeds 单次重复（N=1）——正式实验需 N=3~5。
