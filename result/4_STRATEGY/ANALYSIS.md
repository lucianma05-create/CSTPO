# 策略迁移分析（细粒度 → 元策略两层）

配套图：`pre_result/strategy_distribution/strategy_distribution_3in1.png`（细粒度前后对比）与 `strategy_distribution_meta_3in1.png`（元策略预算结构，四类着色）。

## 实验设置

每任务 100 个冻结种子，统计策略标签在对话各轮的出现占比。前（Before RL）= SFT 基座策略分布；后（After RL）= RL 收敛后的策略分布。三个任务共 26 个细粒度策略标签（esconv 8 / cb 6 / p4g 12），按跨任务共性归并为 4 个元策略。

## 表 1：细粒度策略 → 元策略映射（及颜色示意）

| 元策略 | 色标 | 含义 | esconv | cb | p4g |
|---|---|---|---|---|---|
| Advance 推进 | 橘 #E29135 | 朝任务目标直接施压（报价/请求/说服/建议） | Providing Suggestions | propose_price、stance | donate_request、logical_appeal、credibility_appeal、foot_in_the_door |
| Soften 缓和 | 绿 #98DF8A | 建立信任与共鸣、降低对抗张力 | Affirmation and Reassurance、Reflection of feelings、Self-disclosure | social | praise_user、social、emotion_appeal、self_modeling、personal_story |
| Probe 探询 | 蓝 #AEC7E8 | 信息收集与交换、为下一步铺垫 | Question、Information、Restatement or Paraphrasing | inquire、inform | donation_information、inquiry_response |
| Other 兜底 | 灰 #9B9896 | 泛用/占位 | Others | other | other |

三个任务共享同一套"施压—缓冲—试探"循环语法：安抚靠 Soften、谈判靠 Advance、劝捐靠 Probe 铺垫，兜底类在 RL 后全部收缩。

## 表 2：元策略层前后比例（%，Δ = 后 − 前）

| 任务 | Advance | Soften | Probe | Other |
|---|---|---|---|---|
| ESConv 前 | 17.9 | 43.2 | 26.8 | 12.1 |
| ESConv 后 | 22.3 (+4.4) | **54.4 (+11.2)** | 18.4 (−8.4) | 4.9 (−7.2) |
| CB 前 | 52.1 | 28.3 | 14.2 | 5.4 |
| CB 后 | **54.9 (+2.8)** | 22.7 (−5.6) | 19.8 (+5.6) | 2.6 (−2.8) |
| P4G 前 | 25.8 | 31.9 | 33.1 | 9.2 |
| P4G 后 | **33.6 (+7.8)** | 27.6 (−4.3) | 30.3 (−2.8) | 8.5 (−0.7) |

## 表 3：细粒度层前后比例（%，Δ = 后 − 前）

图中缩写列为配套图 `strategy_distribution_3in1.png` 的横轴标签（单个词、首字母大写；全名以本节表为准）。

ESConv：

| 标签（元类） | 图中缩写 | 前 | 后 | Δ |
|---|---|---|---|---|
| Affirmation and Reassurance（Soften） | Affirm | 30.2 | 39.7 | +9.5 |
| Providing Suggestions（Advance） | Suggest | 17.9 | 22.3 | +4.4 |
| Reflection of feelings（Soften） | Reflect | 8.6 | 10.6 | +2.0 |
| Self-disclosure（Soften） | Disclosure | 4.4 | 4.1 | −0.3 |
| Question（Probe） | Question | 15.8 | 9.8 | −6.0 |
| Information（Probe） | Inform | 6.1 | 4.4 | −1.7 |
| Restatement or Paraphrasing（Probe） | Restate | 4.9 | 4.2 | −0.7 |
| Others（Other） | Others | 12.1 | 4.9 | −7.2 |

CraigslistBargain：

| 标签（元类） | 图中缩写 | 前 | 后 | Δ |
|---|---|---|---|---|
| propose_price（Advance） | Price | 49.8 | 54.3 | +4.5 |
| stance（Advance） | Stance | 2.3 | 0.6 | −1.7 |
| social（Soften） | Social | 28.3 | 22.7 | −5.6 |
| inquire（Probe） | Inquire | 10.9 | 13.4 | +2.5 |
| inform（Probe） | Inform | 3.3 | 6.4 | +3.1 |
| other（Other） | Other | 5.4 | 2.6 | −2.8 |

P4G：

| 标签（元类） | 图中缩写 | 前 | 后 | Δ |
|---|---|---|---|---|
| donate_request（Advance） | Request | 4.9 | 10.9 | +6.0 |
| logical_appeal（Advance） | Logic | 5.1 | 5.2 | +0.1 |
| credibility_appeal（Advance） | Credibility | 13.7 | 13.9 | +0.2 |
| foot_in_the_door（Advance） | Footdoor | 2.1 | 3.6 | +1.5 |
| praise_user（Soften） | Praise | 14.3 | 16.1 | +1.8 |
| social（Soften） | Social | 9.8 | 7.9 | −1.9 |
| emotion_appeal（Soften） | Emotion | 3.8 | 1.4 | −2.4 |
| self_modeling（Soften） | Modeling | 1.7 | 0.6 | −1.1 |
| personal_story（Soften） | Story | 2.3 | 1.6 | −0.7 |
| donation_information（Probe） | Donation | 20.3 | 18.4 | −1.9 |
| inquiry_response（Probe） | Response | 12.8 | 11.9 | −0.9 |
| other（Other） | Other | 9.2 | 8.5 | −0.7 |

## 简要分析：迁移趋势与对 policy 的影响

**迁移趋势有三个一致特征。** 第一，Other 兜底类在三个任务上全部收缩（−7.2 / −2.8 / −0.7）——RL 后策略不再"泛泛而谈"，预算从无信息量的占位动作中撤出。第二，每个任务的预算集中到自己的主杠杆：esconv 是 Soften（+11.2，A&R 一己之力 +9.5），cb 与 p4g 是 Advance（cb 已是主场仍 +2.8，p4g +7.8 其中 donate_request 翻倍）。第三，探询与缓和按任务功能重排：cb 的 Probe 显著上升（+5.6，询价→精准报价），esconv 的 Probe 大幅下降（−8.4，共情任务里提问让位于倾听）。

**对 policy 的影响对应到 MAIN.md 的子指标变化。** esconv 的 Soften 集中直接支撑 A（行动打算）2.364→2.679：先共情建立信任、再以建议推动行动（Advance +4.4 辅助），E 的微升则由 A&R 的加仓解释。cb 的"询价增多 + 报价集中"支撑了 raw_sl 0.531→0.581 的高质量成交——更多信息采集换来更准的报价点。p4g 的 Advance 加仓（尤其 donate_request 4.9→10.9）是承诺率 42%→46% 的直接机制——请求是承诺的充分条件，铺垫类策略相应让位。整体上，RL 让三个 policy 从"均匀铺开"变为"主杠杆集中 + 兜底收缩"，且集中方向与各任务回报的定义结构（esconv=(E+A)/8、cb=clip(raw_sl)、p4g=承诺率）严格对齐。

**两层表示的互补性。** 细粒度层回答"哪个具体动作变了"（如 p4g 的 donate_request 翻倍），元策略层回答"预算结构怎么迁移"（如三任务共享的 Other 收缩 + 主杠杆集中）。26 个细标签聚合到 4 个元类后，每类的监督密度与信噪比都大幅提升——这既是本文策略迁移分析的展示口径，也是标签级 credit 可辨识性的结构前提。
