## 5. 实施细节

### 5.1 数据集与交互种子

我们在 ESConv、PersuasionForGood（P4G）和 CraigslistBargain 三个公开数据集上分别训练策略。ESConv 是 supporter 与 seeker 之间的情感支持任务，使用 8 类官方支持策略；P4G 是 persuader 与 persuadee 之间的慈善捐赠劝说任务，将原始细粒度标注归并为 13 类策略；CraigslistBargain 将 Actor 设为 buyer、Cog-Sim 设为 seller，使用 6 类对话策略，并将报价、接受和退出作为独立任务事件。三个数据集的 train/validation/test 规模分别为 1196/150/150、813/102/102 和 5247/597/838，三个任务独立训练，不共享策略标签或回报尺度。

每个 rollout seed 包含来源与分组标识、双方角色、Actor 与用户各自可见的任务信息、截止点前的对话、用户 Persona、认知参数 \(\Theta_u\)、初始 BDI、初始 Emotion，以及随机流、轮数预算和组件版本等复现信息。ESConv seed 还包含问题情境和起始情绪证据；P4G seed 包含与任务相关的用户问卷特征；CraigslistBargain seed 分开保存商品知识库和买卖双方各自的目标价。Actor 只读取对话历史和自己的任务视图；用户认知私有状态仅供 Cog-Sim 和训练侧认知 critic 使用，不进入 Actor 输入。终局 Judge 只读取任务 rubric 所需的白名单案例事实。原始后续对话与终局结果不进入 rollout 初始化。

### 5.2 SFT 与强化学习

我们采用 Qwen3-14B 作为策略模型，并为三个任务分别训练。SFT 样本统一写成“策略标签 + 换行 + 话语”，由同一自回归模型联合学习。训练数据由官方真人对话与冻结教师策略在 Cog-Sim 中产生的蒸馏轨迹按 1:1 混合。SFT 使用 LoRA（rank 64、alpha 128、dropout 0.05），最大长度为 4096，学习率为 $2\times10^{-5}$，训练 4 个 epoch。

RL 从各任务的 SFT checkpoint 初始化，使用多轮 PPO、Cog-critic 和冻结的 SFT 参考策略。Actor 与 critic 的学习率分别为 $1\times10^{-5}$ 和 $1\times10^{-4}$，LoRA rank 均为 64，PPO 裁剪系数为 0.2，折扣因子为 1。每个训练 step 抽取 10 个种子并分别生成主干轨迹。对于 CraigslistBargain 和 P4G，我们在每条主干上选取 3 个尚未结束的节点，并从每个节点的同一动作前状态独立采样 $n=2$ 条续演；对于 ESConv，则选取 4 个尚未结束的节点并进行相同的 $n=2$ 扩展。Actor 每次生成自然语言回复的长度上限为 128 tokens，整条对话最多持续 30 个 Actor–User 回合。rollout 使用 temperature 1 和 top-p 1，以保持采样分布与训练 log-probability 一致；策略标签通过 trie 约束采样，扩展分支用于训练 critic 和估计字段级优势。训练基于 verl，生成后端采用 vLLM。所有实验在两张 NVIDIA A100 80GB GPU 上完成。SFT 使用单张 GPU；

### 5.3 用户模拟与终局评价

Cog-Sim 的语义推理模块使用冻结的 GPT-5.5、固定提示词和确定性路线设置，训练期间不更新。终局 Judge 使用相同的冻结后端，对每条完整对话进行 3 次独立结构化评价并聚合结果；ESConv 返回情绪/希望改善与行动计划，P4G 返回承诺及撤回状态，CraigslistBargain 返回成交状态和最终价格。能够确定性计算的标量效用由程序完成。Judge 只读取对话和必要任务事实，不读取 Cog-Sim 隐状态或 critic 输出。

评估采用 free100 协议：300 case 冻结评估集（t001–t100 × 3 任务）、Cog-Sim 用户模拟、judge rev4（n=3 聚合）、自由交互协议（13 轮起判冗余、每 2 轮一判、30 轮上限）、统一后处理；统计按 case 聚类 cluster bootstrap（2000 次，* = p<0.05），成对 diff 分两层（方法 − standard 同骨干内、跨骨干逐 case 配对）。
