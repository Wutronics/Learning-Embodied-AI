# 17｜π₀：从 VLM 到 Flow-Based Continuous Action Generation

> 本章属于《VLA 与现代机器人学习基础》的 VLA 核心模型部分。  
> 目标不是把 π₀ 当作一个孤立模型记忆，而是理解它为什么在 RT-2、OpenVLA 之后出现，它如何把预训练视觉语言模型的语义能力与连续机器人动作生成连接起来，以及 Flow Matching、Action Expert、Action Chunk、Cross-Embodiment 这些此前已经学习过的概念如何在一个完整 VLA 中汇合。

---

## 1. 本章导言：π₀ 为什么值得单独学习

在进入 π₀ 之前，已经可以建立一条较清晰的 VLA 演化线索：

```text
Robot Policy
↓
Transformer / VLM
↓
RT-1
↓
RT-2
↓
Open X-Embodiment
↓
OpenVLA
↓
π₀
```

RT-2 展示了一个关键事实：预训练 Vision-Language Model 中的互联网视觉语言知识，可以迁移到机器人控制任务中。OpenVLA 则进一步把这一路线开放化、工程化，使“VLM + action tokenization + robot data fine-tuning”成为一个较清晰的开放 VLA 范式。

但这条路线始终存在一个结构性问题：

> 语言天然是离散 token，而机器人动作天然是连续控制量。  
> 是否有必要为了适配语言模型，而把连续机器人动作强行离散成 token？

π₀ 的核心变化正是从这里开始。

它仍然保留预训练 VLM，用来理解视觉、语言和任务语义；但在动作生成端，不再主要依赖“把 action 当作语言 token”这一思路，而是引入专门面向机器人连续控制的 Action Expert，并使用 Flow Matching 直接生成连续 Action Chunk。

因此，理解 π₀ 的关键不是背网络模块，而是抓住这一分工：

```text
VLM Backbone
负责理解“世界里有什么、任务要做什么”

Action Expert
负责生成“机器人具体应该怎么连续运动”
```

两者不是前后串联的简单两阶段系统，而是在同一个 Transformer 计算框架中进行信息交互。

学完本章后，应能够回答：

- π₀ 为什么要引入 Action Expert；
- 为什么不直接继续使用 action tokenization；
- Flow Matching 在 π₀ 中到底学习什么；
- 一个 50-step Action Chunk 是怎样从 Gaussian noise 生成的；
- robot state 如何进入模型；
- π₀ 如何处理不同 embodiment；
- π₀ 和 RT-2、OpenVLA 的本质差异是什么；
- π₀ 仍然没有解决哪些真实机器人问题。

---

# 2. 问题从哪里来：连续机器人动作为什么不一定适合 tokenization

## 2.1 语言输出和机器人输出在本质上不同

自然语言序列天然是离散符号：

```text
pick
up
the
cup
```

语言模型可以把这些符号转换成 token ID，再进行自回归建模。

而机器人动作通常是连续向量。例如一个基于末端执行器增量的动作可能写成：

```text
[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
```

这些量本质上属于连续空间。

如果为了复用语言模型输出头，把每个动作维度离散到若干 bin，再转换为 action token，就会引入两类问题。

第一类是量化误差。

连续值被划分成有限区间以后，不可能完全保留原始精度。

第二类是生成形式与控制问题之间的不匹配。

语言模型擅长的是离散 token 序列生成，而机器人控制往往希望直接得到一段具有时间连续性的动作轨迹。

因此，一个更自然的问题是：

> 能不能保留 VLM 的视觉语言理解能力，但让机器人动作按照连续生成模型的方式产生？

π₀ 的答案是：可以。

---

## 2.2 π₀ 的基本方向

π₀ 可以先抽象为：

```text
Vision-Language Backbone
        +
   Action Expert
        ↓
   Flow Matching
        ↓
Continuous Action Chunk
```

它并没有否定 RT-2 的“互联网语义知识帮助机器人”这一核心思想。

相反，π₀ 继承了 VLM 预训练带来的视觉与语义能力，只是在动作生成端做了更适合连续控制的设计。

因此，从研究演化角度，可以把它理解为：

```text
RT-2：
VLM 可以控制机器人

OpenVLA：
把 VLA 做成开放、可训练、可适配的系统

π₀：
保留 VLM，但重新设计连续机器人动作生成机制
```

---

## 阶段思考 A：为什么 π₀ 不直接继续使用离散 action token？

### 问题

如果 RT-2 和 OpenVLA 已经证明 action tokenization 可以工作，那么 π₀ 为什么还要改成连续 Flow Matching？

### 解析

一个只回答“离散化会损失精度”的答案方向正确，但不完整。

完整原因至少包括三层：

```text
连续动作本来就不是离散语言符号
↓
离散化会引入量化误差
↓
自回归 token 生成与高频连续控制的归纳偏置并不完全匹配
↓
因此更自然的做法是直接建模连续动作分布
```

π₀ 的重点不是简单追求“更精确的小数”，而是重新选择一种更贴近机器人控制问题结构的动作生成方式。

---

# 3. π₀ 的整体架构

## 3.1 输入是什么

一个典型 observation 可以写成：

$$
o_t
=
\left(
I_t^1,
I_t^2,
\dots,
I_t^n,
l,
q_t
\right)
$$

其中：

- I_t^i：第 i 个摄像头在时刻 t 的 RGB 图像；
- l：语言指令；
- q_t：机器人 proprioceptive state；
- n：摄像头数量。

q_t 可以包含：

- joint position；
- gripper state；
- base state；
- 与具体机器人控制接口有关的其他状态。

视觉信息描述“外部世界现在是什么样”，proprioception 描述“机器人自己现在是什么样”。

二者缺一不可。

---

## 3.2 输出是什么

π₀ 并不只预测一个单步动作，而是生成 Action Chunk：

$$
A_t
=
\left[
a_t,
a_{t+1},
\dots,
a_{t+H-1}
\right]
$$

其中：

- a_t：单个控制时间步的连续动作；
- H：Action Chunk horizon。

在原始 π₀ 设计中，H 约为 50。

如果单步动作维度为 D，则：

$$
A_t
\in
\mathbb{R}^{H \times D}
$$

例如：

```text
H = 50
D = 18
```

那么一个完整 Action Chunk 中包含：

```text
50 × 18 = 900
```

个连续标量。

---

## 3.3 两个核心专家

π₀ 的整体结构可以用下面的概念图理解：

```text
Image Tokens + Language Tokens
            ↓
        VLM Expert
            ↕
      Attention 交互
            ↕
        Action Expert
            ↑
Robot State + Noisy Action + Flow Time
            ↓
      Velocity Field
            ↓
      Flow Integration
            ↓
Continuous Action Chunk
```

这里最重要的是：

> Action Expert 不是简单接在 VLM 后面的一个 MLP action head。

它本身也是 Transformer 侧的重要计算模块，负责面向机器人状态和连续动作进行建模。

---

# 4. Vision-Language Backbone：负责“理解世界”

π₀ 使用预训练 VLM 作为视觉语言骨干。

其作用可以理解为：

```text
Image
+
Language
↓
形成与物体、语义、任务相关的多模态表示
```

例如任务：

```text
Fold the shirt.
```

VLM 需要帮助系统理解：

- 图像中的物体是什么；
- 哪部分是衣服；
- “fold” 的语义是什么；
- 指令所指向的目标对象是什么。

这种能力不是单靠少量机器人数据容易学到的。

因此，π₀ 保留了 RT-2 路线中非常重要的一条思想：

```text
Internet-scale visual-language prior
↓
帮助机器人进行语义理解和任务泛化
```

需要特别强调：

> VLM Backbone 并不是“替机器人下达高层指令”的独立 planner。

它主要负责形成视觉语言语义表示，并把这些信息提供给动作生成部分。

---

## 阶段思考 B：VLM Backbone 和 Action Expert 分别做什么？

### 问题

能否简单说成：

```text
VLM 负责下达指令
Action Expert 负责执行指令
```

### 解析与纠偏

这种说法太像传统分层规划器，会误导对 π₀ 的理解。

更准确的心智模型是：

```text
VLM Backbone
→ 提供视觉语言语义特征

Action Expert
→ 在视觉语言语义 + robot state + noisy action 条件下，
   学习连续动作的生成过程
```

两者通过 Transformer 内部的信息交互共同产生动作。

因此，π₀ 并不是：

```text
VLM 生成一句文字计划
↓
Action Expert 根据文字执行
```

而更接近：

```text
视觉、语言、机器人状态和动作生成过程
在统一 Transformer 框架中发生条件交互
```

---

# 5. Action Expert：负责“机器人到底怎么动”

## 5.1 为什么需要独立 Action Expert

机器人动作生成和语言生成有不同需求。

语言模型更关心：

```text
语义连贯
token 概率
文本序列
```

而机器人动作模型更关心：

```text
连续控制
时间相关性
动作轨迹
高频执行
跨 embodiment
```

因此，π₀ 引入 Action Expert，让动作生成模块拥有更适合机器人控制的参数与表示能力。

---

## 5.2 Action Expert 的输入

简化表示为：

$$
v_\theta
\left(
A^\tau,
\tau
\mid
o_t
\right)
$$

其中：

- A^τ：Flow Matching 过程中的中间 noisy action chunk；
- τ：Flow time；
- o_t：当前 observation；
- θ：模型参数；
- v_θ：Action Expert 预测的 velocity field。

这里的 velocity field 不是机器人真实关节速度。

它是：

> 在“动作生成空间”中，当前 noisy action 应该朝哪个方向变化，才能逐步变成合理动作。

---

## 阶段思考 C：velocity field 是机器人速度吗？

### 问题

符号 v_θ 看起来像 velocity，那么它是不是 joint velocity 或 end-effector velocity？

### 解析

不是。

这里的 velocity 是生成模型中的“流速”。

可以理解为：

```text
当前动作样本位于动作空间中的某个位置
↓
模型预测一个方向
↓
沿这个方向走一步
↓
得到更接近真实动作分布的样本
```

所以：

```text
机器人物理速度
≠
Flow Matching velocity field
```

阅读 flow-based robot policy 论文时，这是非常容易混淆的术语。

---

# 6. 数学核心：π₀ 如何使用 Flow Matching 生成 Action Chunk

这一部分是本章最重要的数学机制。

---

## 6.1 训练数据中的真实动作

设专家 demonstration 中有一个真实 Action Chunk：

$$
A^*
=
\left[
a_t^*,
a_{t+1}^*,
\dots,
a_{t+H-1}^*
\right]
$$

若 H = 50，单步动作维度为 D，则：

$$
A^*
\in
\mathbb{R}^{50 \times D}
$$

---

## 6.2 采样 Gaussian noise

训练时先采样与真实 Action Chunk 同形状的高斯噪声：

$$
\epsilon
\sim
\mathcal{N}(0,I)
$$

并且：

$$
\epsilon
\in
\mathbb{R}^{H \times D}
$$

例如 H = 50、D = 18：

$$
\epsilon
\in
\mathbb{R}^{50 \times 18}
$$

---

## 6.3 构造中间动作状态

定义 Flow time τ。

在本章采用的方向中：

```text
τ = 0
→ pure noise

τ = 1
→ expert action
```

中间状态写为：

$$
A^\tau
=
(1-\tau)\epsilon
+
\tau A^*
$$

当 τ = 0：

$$
A^0
=
\epsilon
$$

当 τ = 1：

$$
A^1
=
A^*
$$

因此可以把整个过程理解为：

```text
Gaussian Noise
↓
中间 noisy action
↓
中间 noisy action
↓
...
↓
Expert-like Action Chunk
```

---

## 6.4 真实 velocity target 从哪里来

对上式关于 τ 求导：

$$
\frac{dA^\tau}{d\tau}
=
A^*
-
\epsilon
$$

因此目标 velocity field 为：

$$
u^*
=
A^*
-
\epsilon
$$

训练时之所以可以直接得到这个监督目标，是因为：

- A* 来自 demonstration；
- ε 是人工采样得到的；
- 两者都已知。

于是无需额外标注 velocity field。

---

## 6.5 Flow Matching Loss

模型预测：

$$
v_\theta
\left(
A^\tau,
\tau
\mid
o_t
\right)
$$

训练目标是让它逼近：

$$
A^*
-
\epsilon
$$

因此可以写出：

$$
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}
\left[
\left\|
v_\theta
\left(
A^\tau,
\tau
\mid
o_t
\right)
-
\left(
A^*
-
\epsilon
\right)
\right\|_2^2
\right]
$$

直觉上：

```text
输入：
当前 observation
+
当前 noisy action
+
当前 flow time

模型：
预测 noisy action 应该往哪里移动

监督：
真实动作 A* 相对于噪声 ε 的方向
```

---

## 阶段思考 D：Action Expert 在训练时到底学什么？

### 问题

下面三种说法中，哪一种最准确？

1. 直接预测最终 Action Chunk；
2. 预测加入到 action 上的 Gaussian noise；
3. 预测动作生成空间中的 velocity field。

### 解析

π₀ 这里应选第 3 种。

它并不把问题写成：

```text
observation → final action
```

而是学习：

```text
当前 noisy action
+
condition
+
flow time
↓
应该朝哪个方向移动
```

从而建立一个连续生成过程。

---

# 7. 推理：从 Gaussian noise 到机器人动作

训练时可以看到 A*。

推理时没有专家答案，因此只能从随机噪声出发。

初始：

$$
A^0
\sim
\mathcal{N}(0,I)
$$

Action Expert 预测：

$$
v_\theta
\left(
A^\tau,
\tau
\mid
o_t
\right)
$$

然后进行数值积分。

最简单的 Euler 更新可以写成：

$$
A^{\tau+\Delta\tau}
=
A^\tau
+
\Delta\tau
\cdot
v_\theta
\left(
A^\tau,
\tau
\mid
o_t
\right)
$$

不断更新以后：

```text
Noise
↓
第一次更新
↓
第二次更新
↓
...
↓
最终 Continuous Action Chunk
```

需要注意：

> 多次 Flow integration 并不是“每次生成一个机器人动作”。

每一次更新处理的是整个 H × D 动作块。

例如：

```text
H = 50
D = 18
```

那么：

```text
每一次积分
都在更新完整的 50 × 18 张量
```

而不是只产生 1 个动作。

---

## 阶段思考 E：10 次 Flow integration 等于 10 个动作吗？

### 解析

不是。

这两个“时间”必须严格区分：

```text
Robot Time
t, t+1, ..., t+49

Flow Time
τ₀, τ₁, ..., τ₁₀
```

Robot Time 表示动作序列内部的控制时间。

Flow Time 表示生成模型把噪声变成动作的计算过程。

因此：

```text
50 个 robot action steps
≠
10 个 flow integration steps
```

这也是生成式机器人策略中非常关键的二维时间结构。

---

# 8. Robot State 为什么必须输入模型

假设摄像头看到同一个杯子，语言指令都是：

```text
Pick up the cup.
```

但机器人当前关节状态不同：

```text
情况 A：
机械臂已经靠近杯子

情况 B：
机械臂仍在桌子另一侧
```

即使图像相似、语言完全相同，下一步动作也不应相同。

因此机器人策略需要 proprioception。

可以把信息分成：

```text
Vision
→ 外部世界现在是什么样

Language
→ 任务希望完成什么

Robot State
→ 机器人自己现在是什么状态
```

只有三者结合，动作生成才有充分条件。

---

## 8.1 Robot State 如何进入 Action Expert

Robot State 不能直接以原始数字“裸输入” Transformer。

通常需要先通过投影层映射到模型内部 embedding dimension，再参与 Transformer 计算。

概念上：

```text
q_t
↓
Linear / Projection
↓
State Representation
↓
Action Expert
```

同理，Noisy Action Chunk 也要转换成适合 Transformer 处理的表示。

因此不能把 Action Expert 简化成：

```text
数字输入
↓
黑盒 Transformer
↓
动作
```

更正确的是：

```text
Robot State
→ embedding

Noisy Action
→ embedding

Flow Time
→ time conditioning

VLM Representation
→ semantic context

共同进入 Action Expert
```

---

# 9. Action Chunk：为什么一次生成一整段动作

单步策略写成：

$$
o_t
\rightarrow
a_t
$$

Action Chunk 策略写成：

$$
o_t
\rightarrow
\left[
a_t,
a_{t+1},
\dots,
a_{t+H-1}
\right]
$$

π₀ 的 Flow Matching 不是独立地生成每个动作维度，而是在整个 H × D 连续空间中联合建模。

这意味着模型可以学习时间相关结构。

例如 Pick-and-Place：

```text
靠近物体
↓
减速
↓
调整姿态
↓
闭合夹爪
↓
抬起
↓
移动
↓
放置
```

这些动作之间不是独立样本，而是连续技能的一部分。

Action Chunk 可以让模型直接学习这种时间结构。

---

## 9.1 Action Chunk 的优势

主要包括：

- 提高连续动作的时间一致性；
- 降低每个控制周期都必须进行完整大模型推理的压力；
- 允许模型在较长时间跨度内学习动作相关性；
- 更适合复杂 manipulation skill。

---

## 9.2 Action Chunk 的风险

Action Chunk 同样存在严重问题：

> 未来环境变化以后，之前生成的后半段动作可能已经过时。

例如：

```text
t = 0
模型看到杯子位置
↓
生成未来 50 步动作

执行到第 20 步时
杯子被人移动
↓
剩余动作仍然基于旧 observation
```

此时会出现 stale action。

因此：

```text
长 Chunk
→ 推理效率更高、时间一致性更好

但
→ 对突发环境变化反应可能更慢
```

这会自然连接到后续的实时 VLA、asynchronous inference、action queue 和 latency 问题。

---

# 10. Cross-Embodiment：不同机器人为什么还能一起训练

π₀ 希望利用多机器人、多任务数据训练 generalist policy。

但不同机器人存在明显差异：

```text
Franka
UR5e
双臂机器人
移动操作机器人
```

它们可能具有不同：

- 关节数量；
- action dimension；
- state dimension；
- 控制接口；
- gripper；
- camera layout；
- 动力学。

因此 Cross-Embodiment 的第一步不是“学习泛化”，而是先解决数据接口问题。

---

## 10.1 统一 tensor shape

一种直接方法是设定最大维度 D_max。

例如：

$$
D_{\max}
=
18
$$

如果某机器人只有 7 维动作：

```text
[a₁, a₂, ..., a₇]
```

可以 padding 为：

```text
[a₁, a₂, ..., a₇, 0, 0, ..., 0]
```

最终统一为：

$$
a
\in
\mathbb{R}^{18}
$$

如果 H = 50，则所有机器人可以被整理成：

$$
A
\in
\mathbb{R}^{50 \times 18}
$$

这样至少可以进入同一个网络。

---

## 阶段思考 F：Padding 是否等于动作空间真正统一？

### 解析

绝对不是。

Zero Padding 只解决：

```text
tensor shape
```

没有解决：

```text
action semantics
kinematics
dynamics
controller interface
```

例如：

```text
维度 1
```

在机器人 A 上可能代表某个 joint position，

在机器人 B 上可能代表另一种控制量。

因此：

```text
统一 shape
≠
统一物理意义
```

真正的 Cross-Embodiment 泛化仍需要依赖：

- 数据混合；
- embodiment context；
- normalization；
- 网络学习；
- 任务与动作表示的设计。

---

# 11. 数据尺度：为什么 π₀ 需要多机器人、多任务数据

π₀ 的目标并不是只训练一个单任务策略。

它希望学习更广泛的 manipulation prior。

因此训练数据需要覆盖：

```text
不同机器人
+
不同任务
+
不同场景
+
不同物体
+
不同视觉条件
```

这种数据规模可以帮助模型学到：

- 更通用的视觉物体表示；
- 多样的 manipulation pattern；
- 不同 embodiment 下的运动规律；
- 一定程度的跨任务和跨场景泛化。

但“大规模预训练”并不意味着下游机器人不再需要适配。

现实中仍然常见：

```text
General Robot Pretraining
↓
Specific Robot / Task Fine-tuning
```

原因包括：

- 新机器人运动学不同；
- camera setup 不同；
- action representation 不同；
- task distribution 不同；
- observation distribution 不同。

---

# 12. 完整训练数据流

将前面内容合起来，可以得到 π₀ 的训练流程。

```text
Robot Demonstration Dataset
↓
Image
Language Instruction
Robot State
Expert Action Chunk
↓
视觉 / 语言预处理
Robot State embedding
↓
采样 Gaussian Noise ε
采样 Flow Time τ
↓
构造 A^τ
↓
VLM Backbone
+
Action Expert
↓
预测 velocity field
↓
与 A* - ε 比较
↓
Flow Matching Loss
↓
Optimizer
↓
更新模型参数
```

数学核心是：

$$
A^\tau
=
(1-\tau)\epsilon
+
\tau A^*
$$

以及：

$$
u^*
=
A^*
-
\epsilon
$$

最终：

$$
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}
\left[
\left\|
v_\theta
\left(
A^\tau,
\tau
\mid
o_t
\right)
-
u^*
\right\|_2^2
\right]
$$

---

# 13. 完整推理数据流

部署时：

```text
Camera Images
+
Language Instruction
+
Robot State
↓
VLM Backbone
↓
视觉语言语义特征
↓
Gaussian Noise Action Chunk
↓
Action Expert
↓
Velocity Field
↓
Flow Integration
↓
Continuous Action Chunk
↓
Robot Controller
↓
Actuator
↓
环境变化
↓
New Observation
↓
重新推理
```

注意控制层级：

```text
High-level semantic understanding
↓
VLA / π₀
↓
Continuous Action Chunk
↓
IK / Operational Space Controller / Joint Controller
↓
Motor command
↓
Robot
```

π₀ 并不是直接输出电机 PWM。

它仍然处在 learned policy / trajectory generation 这一层。

---

# 14. 完整机器人实例：Pick-and-Place

下面用一个完整任务贯穿 π₀。

任务：

```text
把桌上的药盒抓起来，放进右侧篮子。
```

---

## 14.1 Observation

摄像头获得：

```text
Image 1：
正面 RGB

Image 2：
腕部 RGB
```

语言：

```text
Put the medicine box into the right basket.
```

Robot State：

```text
当前关节位置
当前 gripper 状态
可能的末端状态
```

---

## 14.2 VLM Backbone

VLM 负责帮助系统理解：

```text
medicine box 是哪个物体
right basket 在哪里
任务目标是什么
```

它提供语义条件。

---

## 14.3 初始化动作

模型采样：

$$
A^0
\sim
\mathcal{N}(0,I)
$$

假设：

```text
H = 50
D = 18
```

则：

$$
A^0
\in
\mathbb{R}^{50 \times 18}
$$

此时这 900 个数还没有合理机器人意义。

---

## 14.4 Action Expert 迭代生成

Action Expert 读取：

```text
视觉语言语义
+
robot state
+
当前 noisy action
+
flow time
```

预测 velocity field。

通过多次积分，动作块逐渐形成：

```text
向药盒移动
↓
末端对准
↓
闭合 gripper
↓
向上抬起
↓
向右移动
```

注意：

这不是 VLM 先显式输出这五条文字步骤，再让 Action Expert 执行。

这些行为结构是连续动作分布的一部分，由模型内部条件建模形成。

---

## 14.5 执行与重规划

得到 Action Chunk 后，机器人执行其中一部分。

随后重新读取：

```text
New Image
+
New Robot State
```

再次推理。

于是形成：

```text
Observation
↓
π₀
↓
Action Chunk
↓
Robot
↓
New Observation
↓
π₀
```

这才是真正的 closed-loop robot policy。

---

# 15. π₀ 与 RT-2、OpenVLA 的核心差异

这是本章最重要的比较之一。

| 维度 | RT-2 | OpenVLA | π₀ |
|---|---|---|---|
| 基础范式 | VLM → Robot Action | Open VLM-based VLA | VLM + Action Expert |
| 视觉语言能力 | 是 | 是 | 是 |
| 主要动作形式 | 离散 action token | 离散 action token | 连续 action |
| 主要生成机制 | token prediction | token prediction | Flow Matching |
| Action Expert | 非核心 | 非核心 | 核心模块 |
| Continuous Action Chunk | 非核心设计 | 非核心设计 | 核心设计 |
| Cross-Embodiment | 有限 | 强调 | 强调 |
| 动作连续性归纳偏置 | 较弱 | 较弱 | 更强 |

最核心的变化不是：

```text
single action
→
action chunk
```

因为这会把真正的结构变化说得过于简单。

更准确的是：

```text
离散 action token 自回归生成
↓
连续 Action Chunk 的 Flow Matching 生成
```

这才是 OpenVLA → π₀ 在 action generation 上的核心变化。

---

## 阶段思考 G：π₀ 相比 OpenVLA 的变化是不是“从单步变多步”？

### 解析与纠偏

这种说法不准确。

“是否预测 Action Chunk”只是动作时间结构的一部分。

真正需要抓住的是：

```text
OpenVLA：
把 action 离散成 token，
沿用语言模型式 token prediction

π₀：
直接在连续动作空间中，
通过 Flow Matching 生成整个 Action Chunk
```

因此研究论文时，应该优先比较：

```text
Action Representation
+
Action Generation Objective
+
Action Decoder / Expert
```

而不是只看 chunk 长度。

---

# 16. π₀ 与 Diffusion Policy 的关系

π₀ 和 Diffusion Policy 都属于生成式连续动作策略。

共同点：

```text
不是简单 MSE regression
↓
而是学习一个动作分布的生成过程
```

Diffusion Policy 更接近：

```text
Noise
↓
多步 denoising
↓
Action Trajectory
```

π₀ 的 Flow Matching 更接近：

```text
Noise
↓
Learned Velocity Field
↓
ODE Integration
↓
Action Trajectory
```

因此可以把知识链理解为：

```text
Action Chunking
↓
Diffusion Policy
↓
Flow Matching
↓
π₀
```

前面的生成式 policy 学习并不是孤立章节，而是理解 π₀ 的必要基础。

---

# 17. π₀ 与传统 Robot Learning 的联系

π₀ 虽然使用了大型 VLM 和复杂生成模型，但它仍然属于 robot policy。

训练逻辑的本质仍然可以追溯到 demonstration learning。

```text
Robot Demonstration
↓
Observation + Expert Action
↓
学习条件动作分布
↓
Policy Rollout
```

它没有自动摆脱 imitation learning 的基本问题。

尤其是 distribution shift。

训练时：

$$
o
\sim
p_{\mathrm{expert}}(o)
$$

部署时：

$$
o
\sim
p_{\pi}(o)
$$

如果策略犯错，机器人可能进入训练数据中少见的状态。

因此：

```text
π₀
≠
自动解决 compounding error
```

这会自然连接到后续：

- VLA Fine-tuning；
- online correction；
- RL post-training；
- failure recovery；
- Agent + VLA。

---

# 18. 常见误区与概念辨析

## 18.1 VLM Backbone ≠ 高层 Planner

VLM 提供视觉语言语义表征。

它不一定显式输出：

```text
Step 1
Step 2
Step 3
```

再让 Action Expert执行。

---

## 18.2 Action Expert ≠ 简单 MLP Action Head

Action Expert 是模型中的重要 Transformer 专家模块。

它需要处理：

- robot state；
- noisy action；
- flow time；
- 来自 VLM 的条件信息。

---

## 18.3 Flow Velocity ≠ Robot Velocity

生成空间中的：

```text
velocity field
```

与：

```text
joint velocity
Cartesian velocity
```

不是同一个概念。

---

## 18.4 Action Token ≠ 一定是离散 Token ID

阅读论文时要看上下文。

有些论文会把动作序列中的位置也叫 action token，但这不代表它一定采用 OpenVLA 式离散 action tokenizer。

应检查：

```text
是否 quantize
是否映射成 token ID
是否使用 language vocabulary
是否输出 continuous vector
```

---

## 18.5 Padding ≠ Cross-Embodiment 已经解决

Padding 只让不同机器人数据可以进入同一个 tensor。

真正的跨本体泛化仍然非常困难。

---

## 18.6 Action Chunk ≠ Open-Loop Forever

一次生成 Action Chunk 不意味着机器人永远不再观测环境。

实际系统仍然可以：

```text
执行一部分 chunk
↓
重新 observation
↓
重新规划
```

形成 receding-horizon closed loop。

---

# 19. π₀ 仍然不能解决什么

## 19.1 Distribution Shift

大规模数据和强模型并不意味着部署时永远处于训练分布。

只要策略产生错误：

```text
小偏差
↓
进入少见状态
↓
预测质量下降
↓
更大偏差
```

这一问题仍然存在。

---

## 19.2 Stale Action

长 Action Chunk 会使后半段动作基于旧 observation。

如果外部环境变化，原动作计划可能不再合适。

---

## 19.3 Latency

大型 VLM + Action Expert + 多次 Flow integration 都带来计算开销。

机器人控制通常存在三个不同频率：

```text
Camera Frequency
Policy Inference Frequency
Robot Control Frequency
```

它们一般不相等。

因此真实部署中还必须研究：

- inference delay；
- asynchronous inference；
- action queue；
- real-time chunking；
- reactive correction。

---

## 19.4 Cross-Embodiment 仍然困难

即使输入输出 shape 被统一：

```text
不同机器人
仍然拥有不同运动学、控制器和动力学
```

所以跨机器人训练并不是“padding 后一起训练”这么简单。

---

## 19.5 π₀ 不是 World Model

π₀ 主要学习：

$$
o_t
\rightarrow
A_t
$$

它不是显式学习：

$$
(s_t,a_t)
\rightarrow
s_{t+1}
$$

因此不能直接把它等同于一个显式预测世界未来状态的 World Model。

---

# 20. 本章内容在 VLA 知识体系中的位置

可以把当前学习进度放回完整链路：

```text
Robot Policy
↓
Transformer
↓
VLM
↓
Robot Learning / BC
↓
Distribution Shift
↓
Robot Dataset
↓
Action Representation
↓
Action Chunking
↓
ACT
↓
Diffusion
↓
Diffusion Policy
↓
Flow Matching
↓
RT-1 / RT-2
↓
Open X-Embodiment
↓
OpenVLA
↓
π₀
↓
π₀.5
↓
VLA Fine-tuning / Agent + VLA / VLA + RL
```

π₀ 是一个重要汇合点。

它把此前多个知识模块真正组合到一起：

```text
VLM
+
Cross-Embodiment Data
+
Action Chunking
+
Flow Matching
+
Continuous Robot Control
```

因此，学完 π₀ 后，再阅读现代 flow-based VLA，就应该开始从系统角度提问：

```text
Vision encoder 是什么？
Language backbone 是什么？
Robot state 如何输入？
Action space 是什么？
Action chunk 长度是多少？
Flow Matching 如何训练？
Cross-Embodiment 如何统一？
Inference 需要多少步？
Closed-loop 如何实现？
Latency 如何处理？
```

---

# 21. 研究视角：阅读 π₀ 类论文应该重点看什么

## 21.1 Action Representation

首先确认：

```text
joint position?
joint velocity?
EEF pose?
delta pose?
gripper?
```

动作表示会直接影响：

- 数据统一；
- normalization；
- cross-embodiment；
- 低层 controller。

---

## 21.2 Action Generation

确认是：

```text
autoregressive token
regression
diffusion
flow matching
```

不同生成方法意味着完全不同的训练目标和推理开销。

---

## 21.3 Chunk Horizon

需要区分：

```text
prediction horizon
execution horizon
replanning frequency
```

论文中只说 H = 50 并不足够，还要看实际执行多少步后重新推理。

---

## 21.4 Robot State

确认：

- state 包含哪些变量；
- 如何 normalization；
- 如何 embedding；
- 是否不同机器人使用相同 state schema。

---

## 21.5 Cross-Embodiment

重点不是只看“用了多少机器人”。

更要检查：

```text
action space 如何统一
state space 如何统一
camera 缺失怎么处理
dataset mixture 怎么做
不同 embodiment 是否有标识
不同数据质量是否平衡
```

---

## 21.6 Inference Cost

Flow-based policy 的一个关键工程问题是：

```text
一次 action chunk
需要多少次网络前向？
```

这会直接影响：

- policy frequency；
- latency；
- stale observation；
- real-time control。

---

# 22. 综合思考

## 综合题 1

为什么不能把 π₀ 简单总结成：

```text
一个更大的 OpenVLA
```

### 关键思路

因为 π₀ 在 action generation mechanism 上发生了结构性变化：

```text
离散 action token
↓
Continuous Action Chunk + Flow Matching
```

同时引入 Action Expert 负责机器人动作生成。

---

## 综合题 2

为什么 π₀ 仍然需要 robot state？只用 RGB + language 不可以吗？

### 关键思路

视觉主要描述外部环境。

Robot state 描述机器人自身配置。

同一视觉场景中，不同当前关节状态可能需要完全不同的下一步动作。

---

## 综合题 3

Flow Matching 的监督信号为什么不需要人工标注？

### 关键思路

因为：

```text
Expert Action A*
来自 demonstration

Gaussian Noise ε
由训练程序采样
```

所以：

$$
A^*
-
\epsilon
$$

可以直接计算。

---

## 综合题 4

如果 H = 50、D = 18，模型做 10 次 Flow integration，最终生成了多少个机器人动作？

### 解析

最终生成：

```text
50 个 action steps
```

每个 action step 是 18 维。

10 次只是生成过程内部的 integration steps。

---

## 综合题 5

为什么 π₀ 使用 Action Chunk 以后仍然必须 closed-loop？

### 关键思路

因为环境会变化，模型预测也会有误差。

Action Chunk 只能减少推理频率并增加时间一致性，不能消除实时反馈需求。

---

## 综合题 6

Cross-Embodiment 中 Zero Padding 解决了什么，又没有解决什么？

### 解析

解决：

```text
tensor dimension mismatch
```

没有解决：

```text
action semantics
kinematics
dynamics
controller mismatch
```

---

## 综合题 7

从研究路线看，RT-2 → OpenVLA → π₀ 可以如何概括？

### 参考心智模型

```text
RT-2
证明 VLM knowledge 可以帮助机器人

OpenVLA
把 VLA 开放化、工程化

π₀
让 VLM 与连续生成式动作模型更自然地结合
```

---

# 23. 本章总结

π₀ 的核心价值，不只是“又一个更大的机器人模型”，而是重新组织了 VLA 中语义理解与连续控制的关系。

第一，π₀ 保留预训练 VLM，使机器人继续受益于互联网尺度视觉语言知识。

第二，它不再要求所有机器人动作都适应离散语言 token 的生成方式，而是引入 Action Expert，直接针对连续机器人动作建模。

第三，它使用 Flow Matching 学习：

```text
Noisy Action
→
Continuous Action
```

的 velocity field，而不是直接预测最终动作或逐 token 生成动作。

第四，它一次联合生成整个 Continuous Action Chunk，从而建模动作之间的时间相关性。

第五，它通过多机器人、多任务数据尝试获得 Cross-Embodiment manipulation prior，但 tensor shape 的统一并不意味着动作语义和动力学已经统一。

第六，π₀ 仍然是 learned robot policy，因此不会自动消除 distribution shift、latency、stale action、cross-embodiment mismatch 等真实机器人问题。

最终应形成这样的心智模型：

```text
Image + Language
↓
VLM Backbone
↓
Semantic Representation
↕
Action Expert
↑
Robot State + Noisy Action + Flow Time
↓
Velocity Field
↓
Flow Matching Integration
↓
Continuous Action Chunk
↓
Robot Controller
↓
Robot
↓
New Observation
↓
Replan
```

如果能够自然解释这条数据流，并清楚区分：

```text
VLM semantic representation
Action Expert
Flow velocity
Robot physical velocity
Flow time
Robot time
Action Chunk
Cross-Embodiment padding
```

就说明已经建立了 π₀ 的核心知识框架。

---

# 24. 关键词

**Vision-Language Backbone — 视觉语言骨干网络 — 利用预训练视觉语言知识理解图像、语言和任务语义。**

**Action Expert — 动作专家 — 专门处理 robot state、noisy action 和连续动作生成的 Transformer 专家模块。**

**Flow Matching — 流匹配 — 学习一个 velocity field，把 Gaussian noise 连续映射到真实动作分布。**

**Velocity Field — 速度场 — 描述生成空间中样本应如何移动，不等于机器人真实关节速度。**

**Continuous Action — 连续动作 — 不经过离散 action token quantization 的机器人控制向量。**

**Action Chunk — 动作块 — 一次预测未来多个时间步动作形成的连续序列。**

**Flow Time — 流时间 — 生成模型内部从 noise 到 data 的连续时间变量。**

**Robot Time — 机器人时间 — Action Chunk 中真实控制序列的时间索引。**

**Proprioception — 本体状态 — 描述机器人自身关节、夹爪等当前状态的信息。**

**Cross-Embodiment Learning — 跨本体学习 — 使用多种不同机器人平台的数据训练统一策略。**

**Zero Padding — 零填充 — 将不同维度的 state/action 补齐到统一 tensor shape 的工程手段。**

**Distribution Shift — 分布偏移 — 部署时策略访问到的状态分布与专家数据分布不同。**

**Stale Action — 过时动作 — Action Chunk 后半段动作仍基于旧 observation，环境变化后可能失效。**

**Receding Horizon — 滚动时域 — 执行部分动作后重新观测和规划，使 Action Chunk 仍处于闭环控制中。**

---

# 25. 本章最终检查

阅读一篇新的 flow-based VLA 论文时，应能够自然提出：

```text
1. VLM backbone 是什么？
2. Image 如何编码？
3. Language 如何输入？
4. Robot state 如何表示？
5. Action space 是什么？
6. Action 是 continuous 还是 tokenized？
7. 是否预测 Action Chunk？
8. Chunk horizon 是多少？
9. Flow Matching 的训练目标是什么？
10. Flow inference 需要多少步？
11. Action Expert 是否独立参数化？
12. 不同机器人 action dimension 如何统一？
13. Cross-Embodiment 是否只是 padding，还是还有更深层设计？
14. 实际执行多少步后重新规划？
15. Latency 和 stale action 如何处理？
16. 模型仍然会出现哪些 distribution shift？
17. 论文真正的贡献是在语义、动作生成、数据规模还是实时执行？
```

能够围绕这些问题分析模型，就意味着已经不仅是“知道 π₀ 是什么”，而是开始具备阅读现代 VLA 架构与实验设计的能力。
