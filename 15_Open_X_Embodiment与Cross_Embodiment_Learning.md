# 15｜Open X-Embodiment 与 Cross-Embodiment Learning

## 1. 本章导言

在前面的 VLA 学习中，一个核心矛盾逐渐变得清晰：机器人策略希望像语言模型、视觉语言模型那样通过大规模数据获得泛化能力，但机器人数据天然比互联网图文数据更昂贵、更异构，也更难统一。

单一机器人平台通常只能覆盖有限的任务、场景、相机视角和动作分布。如果一个模型只在某一种机械臂、某一种控制接口和某一套数据采集流程上训练，它很容易把“任务本身的规律”与“这台机器人的具体运动方式”绑定在一起。于是，一旦换成另一种机器人，哪怕任务语义完全相同，模型也可能立即失效。

Cross-Embodiment Learning 试图解决的正是这个问题：

> 能否把不同机器人、不同实验室、不同传感器配置、不同动作空间的数据放到一个统一学习框架中，让模型从多种 embodiment 中提取可以共享的任务与操作知识？

Open X-Embodiment 是这一方向中的关键代表。它的重要性不在于“已经把所有机器人统一成同一种控制方式”，而在于推动机器人学习从单一平台的数据孤岛走向大规模、异构、多机器人联合训练。

本章将围绕以下主线展开：

```text
单一机器人数据规模不足
↓
不同机器人数据能否联合训练？
↓
什么是 Embodiment？
↓
为什么 Observation / Action 很难统一？
↓
Normalization 能解决什么，不能解决什么？
↓
如何进行 Dataset Mixture？
↓
Cross-Embodiment 到底共享什么？
↓
为什么会出现 Negative Transfer？
↓
如何判断真正的 Cross-Embodiment Generalization？
↓
Open X-Embodiment 对后续 VLA 的意义
```

学完本章后，应能够独立分析一篇跨机器人 VLA 论文中的数据表示、动作空间、数据混合策略、泛化层次与实验结论，而不是只看到“多机器人数据”就默认模型已经实现了通用机器人控制。

---

## 2. 问题从哪里来：机器人数据为什么难以规模化？

### 2.1 单一机器人数据的局限

假设只使用一台 Franka 机械臂进行训练。

数据可能包含：

- 抓杯子；
- 放盘子；
- 打开抽屉；
- 移动积木；
- Pick-and-Place。

模型最终学习到的是类似：

$$
\pi(a \mid o, l, e_{\text{Franka}})
$$

其中：

- o 表示机器人观测；
- l 表示语言指令；
- a 表示机器人动作；
- e 表示 embodiment 信息。

问题在于，这个策略虽然可能在 Franka 上表现很好，但它看到的“机器人世界”仍然非常窄。

它没有经历：

- 移动底盘参与 manipulation；
- 双臂协同；
- 不同相机布置；
- 不同夹爪；
- 不同关节数；
- 不同控制空间；
- 同一任务的多种执行方式。

因此，模型很容易把：

```text
“抓杯子”这个任务概念
```

和：

```text
“Franka 应该怎样运动”
```

混为一体。

Cross-Embodiment Learning 希望模型逐渐把这两者解耦。

---

### 2.2 为什么互联网图文数据更容易统一？

互联网视觉语言数据常常可以抽象为：

```text
Image + Text
↓
Visual Tokens + Text Tokens
↓
Multimodal Model
```

不同图片虽然尺寸不同，不同语言虽然长度不同，但经过视觉编码器和 tokenizer 后，通常都可以进入统一的 token 空间。

而机器人数据通常是一条 trajectory：

$$
\tau = (o_1, a_1, o_2, a_2, \dots, o_T, a_T)
$$

其中每一个 o 和 a 都与具体机器人平台强相关。

不同数据集可能同时存在：

- 相机数量不同；
- 图像视角不同；
- proprioception 维度不同；
- action 维度不同；
- action 物理语义不同；
- 控制频率不同；
- timestamp 对齐方式不同；
- gripper 定义不同；
- 成功与失败标注方式不同。

因此，机器人数据的难点不是“把文件放在一起”，而是：

> 如何把不同平台的数据变成一个模型可以共同学习、同时又不丢失具体 embodiment 语义的训练体系。

---

## 3. 核心概念：什么是 Embodiment？

Embodiment 不能只理解为“机器人外形”或“机器人型号”。

在机器人学习中，它通常至少包含以下几类因素：

1. 机器人本体结构；
2. 关节自由度；
3. 运动学与动力学约束；
4. 传感器配置；
5. 相机位置与数量；
6. 末端执行器形式；
7. 底层控制接口；
8. 可执行动作空间。

因此，embodiment 更接近：

> 一个机器人“如何感知世界、如何拥有身体、如何通过身体对世界施加动作”的整体配置。

### 3.1 三种机器人执行同一个任务

考虑指令：

```text
Pick up the cup.
```

#### Franka 固定机械臂

可能使用 7 DoF 机械臂和夹爪。

动作可以表示为：

$$
a_t =
[
\Delta x,
\Delta y,
\Delta z,
\Delta r_x,
\Delta r_y,
\Delta r_z,
g
]
$$

这里表示：

- 三维末端平移增量；
- 三维末端旋转增量；
- gripper 状态。

#### 移动机械臂

除了机械臂，还存在移动底盘。

其动作可能包含：

$$
a_t =
[
v_x,
v_y,
\omega,
\Delta x,
\Delta y,
\Delta z,
\dots,
g
]
$$

机器人可能先移动底盘，再调整机械臂，最后完成抓取。

#### 双臂机器人

动作可能写成：

$$
a_t =
[
a_t^{\text{left}},
a_t^{\text{right}}
]
$$

它甚至可能使用一只手固定物体，另一只手完成操作。

因此：

```text
任务语义相同
≠
执行动作相同
```

这就是 Cross-Embodiment Learning 的根本困难之一。

---

## 4. 阶段思考 A：相同任务为什么不能直接共享同一个 Action？

### 思考问题

1. Franka 和移动机械臂都执行“抓起杯子”，为什么不能认为它们的 policy 输出应该完全相同？
2. 两个机器人如果 action 都是 7 维，能否认为它们使用相同的 action space？
3. Cross-Embodiment 真正希望不同机器人共享的是什么？

### 解析

第一问检查的是“任务语义”与“底层动作”的区分。

即使语言指令相同：

```text
pick up the cup
```

不同机器人完成任务时需要考虑不同的：

- 运动学结构；
- 可达空间；
- 底盘；
- 双臂协同；
- gripper；
- 控制器接口。

因此，VLA 最终输出仍然必须适配具体机器人。

第二问更容易产生误解。

两个机器人 action 都是 7 维，并不能说明它们的动作语义相同。例如：

```text
Robot A:
[x, y, z, roll, pitch, yaw, gripper]
```

而另一个机器人可能是：

```text
Robot B:
[q1, q2, q3, q4, q5, q6, q7]
```

两者维度都是 7，但一个是 Cartesian action，另一个是 joint-space action。

所以：

```text
相同维度
≠
相同 Action Space
```

第三问是整个章节最重要的心智模型：

> Cross-Embodiment 希望优先共享的是视觉语义、任务理解、物体关系和可迁移的操作知识，而不是要求所有机器人直接共享完全一致的低层电机命令。

可以概括为：

```text
Shared Knowledge
≠
Shared Motor Command
```

---

## 5. Cross-Embodiment 希望共享什么？

不同机器人虽然身体不同，但它们工作在相同的物理世界中。

例如指令：

```text
Pick up the red cup.
```

无论机器人是什么型号，都需要理解：

```text
识别 red cup
↓
建立目标与场景关系
↓
接近目标
↓
对准
↓
抓取
↓
抬起
```

这些知识具有较强的跨机器人共性。

而：

- 第三个关节旋转多少；
- 是否先移动底盘；
- 左手还是右手；
- 使用关节角还是 EEF delta；

这些属于 embodiment-specific execution。

于是可以建立一个抽象：

```text
Observation + Language
↓
Shared Semantic / Manipulation Representation
↓
Embodiment-Specific Action
```

数学上，可以把共享表示写成：

$$
z = f_\theta(o, l, e)
$$

其中：

- o：机器人观测；
- l：语言任务；
- e：embodiment 信息；
- fθ：共享主干模型；
- z：共享隐表示。

然后再映射到具体动作：

$$
a^{(r)} = g_{\phi_r}(z)
$$

其中：

- r 表示具体机器人；
- gφᵣ 表示 robot-specific action mapping；
- a⁽ʳ⁾ 表示该机器人的动作。

这不是 Open X-Embodiment 唯一的具体结构，而是一种理解“共享知识 + 具体执行”的通用建模方式。

---

## 6. Open X-Embodiment 为什么出现？

机器人领域长期存在大量数据孤岛：

```text
Lab A
↓
Franka Dataset
↓
Policy A
```

```text
Lab B
↓
Mobile Manipulator Dataset
↓
Policy B
```

```text
Lab C
↓
Dual-arm Dataset
↓
Policy C
```

每个实验室独立采集、独立训练、独立评测。

这些数据难以互相利用，导致：

- 单个数据集规模有限；
- 任务覆盖有限；
- embodiment 单一；
- 数据采集成本重复；
- 泛化能力受限。

Open X-Embodiment 的核心意义，可以概括为：

```text
Franka Data ───────┐
Mobile Robot Data ─┤
Dual-arm Data ─────┼──→ Heterogeneous Dataset Mixture
Other Robots ──────┘
                            ↓
                  Generalist Robot Policy
```

它推动机器人学习从：

```text
单实验室
单机器人
单数据集
```

向：

```text
多实验室
多机器人
多任务
大规模异构数据
```

发展。

它并没有简单地把所有 action tensor 直接拼接起来，而是建立一种让多来源机器人数据能够被共同处理、共同训练和共同研究的基础。

---

## 7. 异构数据统一的四个层次

Cross-Embodiment 中的“统一”不是一个单一步骤。

可以分成四个层次理解。

### 7.1 Level 1：统一数据组织格式

第一层先解决：

> 数据应该以什么统一形式存储、读取和对齐？

例如一个标准 episode 可以抽象为：

```json
{
  "observation": {
    "images": "...",
    "robot_state": "..."
  },
  "action": "...",
  "language_instruction": "...",
  "timestamps": "...",
  "metadata": {
    "robot_type": "..."
  }
}
```

统一数据结构的意义是：

- 让 dataloader 可以读取不同数据集；
- 明确 image、state、action、language 的位置；
- 统一 episode / trajectory 组织方式；
- 支持后续 preprocessing。

但必须强调：

> 统一数据字段不等于统一动作空间。

---

### 7.2 Level 2：统一数值尺度

第二层是 Normalization。

不同机器人即使使用相似 action representation，其动作范围也可能不同。

例如：

```text
Robot A:
Δx ∈ [-0.05, 0.05] m

Robot B:
Δx ∈ [-0.20, 0.20] m
```

为了减少不同数据集的数值尺度差异，可以进行归一化。

---

### 7.3 Level 3：统一或适配动作语义

如果两个机器人动作维度和物理意义不同，仅做 normalization 是不够的。

需要进一步考虑：

- 转换到公共动作表示；
- robot-specific action head；
- action tokenizer；
- action mask；
- embodiment conditioning；
- 专用 decoder。

---

### 7.4 Level 4：跨数据集联合训练

最后才进入 Dataset Mixture：

```text
Dataset A
Dataset B
Dataset C
...
↓
Sampling Strategy
↓
Shared Policy Training
```

这里要解决：

- 不同数据量如何采样；
- 哪些数据应提高权重；
- 如何避免大数据集支配训练；
- 如何避免小数据集过拟合；
- 是否存在负迁移。

---

## 8. 数学机制：Action Normalization

### 8.1 为什么需要 Normalization？

假设两个机器人都在执行“向右移动一半最大范围”。

Robot A：

```text
最大范围：±0.05 m
实际动作：+0.025 m
```

Robot B：

```text
最大范围：±0.20 m
实际动作：+0.10 m
```

两个物理动作数值不同，但它们在各自动作范围中的相对位置相同。

一种典型 Min-Max Normalization 可以写为：

$$
a_{\text{norm}}
=
2
\frac
{
a-a_{\min}
}
{
a_{\max}-a_{\min}
}
-1
$$

其中：

- a：原始动作值；
- a_min：该维动作的最小值；
- a_max：该维动作的最大值；
- a_norm：归一化后的动作。

归一化后，动作通常进入统一的有限数值区间，例如：

```text
[-1, 1]
```

这样能够降低不同机器人动作量纲和幅值差异对训练的影响。

---

### 8.2 推理时的反归一化

模型输出的 a_norm 一般不能直接发送给机器人。

需要恢复到目标机器人的实际动作空间。

反归一化为：

$$
a
=
\frac
{
a_{\text{norm}}+1
}
{2}
(a_{\max}-a_{\min})
+
a_{\min}
$$

因此，完整流程是：

```text
原始 Robot Action
↓
Normalize
↓
Policy Training
↓
Normalized Model Output
↓
Denormalize
↓
Robot Controller
```

训练与部署必须使用相互匹配的 normalization 定义和统计参数。

---

## 9. 阶段思考 B：Normalization 到底解决了什么？

### 思考问题

如果两个机器人：

```text
Robot A:
action[0] = x 方向位移

Robot B:
action[0] = 第一关节角
```

两者都被归一化到：

```text
[-1, 1]
```

是否意味着它们已经拥有统一 action space？

### 解析与纠偏

答案是否定的。

Normalization 解决的是：

```text
数值尺度差异
```

它不能解决：

```text
动作物理语义差异
```

即：

```text
Normalization
≠
Action Space Unification
```

这是 Cross-Embodiment 中最重要的误区之一。

即使：

```text
Robot A action[0] = 0.5
Robot B action[0] = 0.5
```

两个 0.5 的含义仍然可能完全不同。

因此，动作空间问题必须继续通过 action representation、robot conditioning、专用 action head 或 tokenizer 等机制处理。

---

## 10. 不同 Action Space 如何处理？

### 10.1 方法一：公共 Action Representation

一种自然思路是：

> 尽可能把不同机器人转换到相似的任务空间动作表示。

例如对机械臂使用：

$$
a=
[
\Delta x,
\Delta y,
\Delta z,
\Delta r_x,
\Delta r_y,
\Delta r_z,
g
]
$$

这样，不同关节结构的机械臂可以尽量共享：

```text
EEF Delta Pose + Gripper
```

这种方式有几个优点：

- 更接近任务空间；
- 减少关节数差异；
- 对不同机械臂更容易统一；
- 有利于任务语义与动作表示对齐。

但它仍有明显限制。

如果机器人包含：

- 移动底盘；
- 双臂；
- 灵巧手；
- humanoid；

简单的 7D EEF action 往往不足以描述所有行为。

因此：

> 公共动作表示是有效工具，但不是万能统一方案。

---

### 10.2 方法二：Shared Backbone + Robot-Specific Action Head

另一种典型思路是：

```text
Image
Language
Robot State
↓
Shared Backbone
↓
Shared Representation z
↓
Robot-Specific Action Head
↓
Robot Action
```

数学上：

$$
z=f_\theta(o,l)
$$

然后针对机器人 r：

$$
a^{(r)}=g_{\phi_r}(z)
$$

共享部分负责学习：

- 物体语义；
- 场景理解；
- 任务结构；
- 操作模式。

专用 action head 负责处理：

- 动作维度；
- 控制接口；
- robot-specific mapping。

这种架构的核心思想是：

```text
共享理解能力
+
保留执行差异
```

---

### 10.3 方法三：Tokenized / Conditional Action

另一类思路是把动作编码为 token 或标准化表示。

例如：

```text
Robot Type: Franka
Action Tokens: [...]
```

或：

```text
Robot Type: Mobile Manipulator
Action Tokens: [...]
```

模型可以使用统一 Transformer 预测 token，但在解码时仍然需要知道：

- 当前机器人是谁；
- token 对应哪种动作语义；
- 应该使用哪套 normalization；
- 如何恢复成控制命令。

因此：

```text
Action Tokenization
≠
Automatically Unified Physical Action
```

Tokenization 只是改变表示方式，不能自动消除物理差异。

---

## 11. Dataset Mixture：不同机器人如何共同训练？

假设有三个数据集：

| 数据集 | 机器人 | 轨迹规模 |
|---|---|---:|
| D₁ | Franka | 100000 |
| D₂ | Mobile Manipulator | 10000 |
| D₃ | Dual-arm | 2000 |

如果直接把所有数据放在一起随机抽样，训练样本会被最大数据集主导。

这可能导致模型最终：

```text
非常擅长 Franka
+
稍微会一点其他机器人
```

因此需要控制 Dataset Mixture。

---

### 11.1 数学表达

设第 i 个数据集为 Dᵢ。

每次训练时，按照概率 pᵢ 选择数据集。

满足：

$$
\sum_{i=1}^{N} p_i = 1
$$

整体训练目标可以写成：

$$
\mathcal L(\theta)
=
\sum_{i=1}^{N}
p_i
\mathbb E_{(o,a,l)\sim D_i}
\left[
\ell
\left(
\pi_\theta(o,l),a
\right)
\right]
$$

其中：

- N：数据集数量；
- pᵢ：数据集 i 的采样权重；
- Dᵢ：第 i 个数据集；
- o：observation；
- a：action；
- l：language instruction；
- πθ：共享策略；
- ℓ：训练损失。

具体 loss 取决于动作表示方式。

例如：

- 连续回归可以使用回归损失；
- Action Token 可以使用 Cross Entropy；
- Diffusion / Flow Policy 使用对应生成式目标。

---

### 11.2 数据量与采样概率不是同一件事

最简单的策略是：

$$
p_i
\propto
|D_i|
$$

这意味着严格按数据规模采样。

但也可以进行平滑：

$$
p_i
\propto
|D_i|^\alpha
$$

其中：

```text
0 < α < 1
```

当 α 较小时，较小数据集会得到更高的相对采样概率。

其目标不是简单追求“平均”，而是避免：

```text
大数据集完全支配训练
```

同时也必须防止：

```text
小数据集被过度采样而过拟合
```

因此 Dataset Mixture 本身就是一个重要设计变量。

---

## 12. 阶段思考 C：Dataset Mixture 的常见误区

### 思考问题 1

如果 Franka 有 100 万条数据，双臂机器人只有 1 万条数据，直接混合后随机训练，会发生什么？

### 解析

最可能的结果是模型的大部分梯度都来自 Franka。

于是虽然名义上是 multi-embodiment training，实际训练分布仍然严重偏向 Franka。

因此：

```text
Dataset Merge
≠
Balanced Multi-Embodiment Learning
```

---

### 思考问题 2

如果提高小数据集采样概率 pᵢ，是否一定提升对应机器人的性能？

### 解析

不一定。

提高 pᵢ 可能：

- 增强该 embodiment 的训练信号；
- 改善被大数据集淹没的问题；

但也可能：

- 放大小数据集噪声；
- 导致过拟合；
- 强化低质量示范；
- 引发负迁移。

所以 pᵢ 是一个需要实验验证的超参数，而不是“越均衡越好”。

---

## 13. Cross-Embodiment Training 为什么可能提升泛化？

Cross-Embodiment 的核心直觉可以概括为：

> 通过 embodiment 的变化，迫使模型学习跨机器人不变的任务结构。

假设模型看过：

```text
Franka:
抓杯子 → 放进盒子
```

又看过：

```text
UR5:
抓杯子 → 放进盒子
```

又看过：

```text
Mobile Manipulator:
移动 → 抓杯子 → 放进盒子
```

变化的是：

```text
Robot Morphology
Action Representation
Camera View
Motion Trajectory
```

稳定的是：

```text
Cup
Box
Pick
Move
Place
Task Order
Object Relation
```

这可以理解成：

```text
通过变化学习不变量
```

于是模型更有可能学习：

```text
Task Semantics
Object Semantics
Manipulation Structure
```

而不是只记住：

```text
Franka 的固定动作模式
```

---

## 14. Cross-Embodiment Generalization 的层次

“Cross-Embodiment”这个词很容易被过度解读。

必须区分至少三个层次。

### 14.1 多机器人联合训练

训练集中包含：

```text
R₁, R₂, R₃
```

模型能够同时控制这些机器人。

这是：

```text
Multi-Embodiment Training
```

它证明的是一个模型可以处理多个已见 embodiment。

但不能自动证明模型能泛化到新机器人。

---

### 14.2 Seen Embodiment 上的新任务

例如：

```text
Franka 在训练中出现过
```

但测试的是新的 language task。

这主要考察：

```text
Task Generalization
```

而不是严格意义上的 Embodiment Generalization。

---

### 14.3 Unseen Embodiment

训练：

```text
R₁, R₂, R₃
```

测试：

```text
R₄
```

且：

```text
R₄ 不属于训练机器人
```

如果模型能够零样本或少量数据快速适配 R₄，这才是更强意义上的：

```text
Cross-Embodiment Generalization
```

因此阅读论文时必须问：

> 测试机器人在训练数据中是否出现过？

这是判断论文结论强弱的关键问题。

---

## 15. 为什么 Zero-Shot 新机器人仍然非常困难？

VLA 最终不是只输出“语义理解”，而是必须产生真实动作。

例如模型知道：

```text
杯子在右边
↓
应该靠近杯子
```

这属于高层语义。

但到了执行层：

```text
Franka:
输出 EEF delta
```

```text
Mobile Manipulator:
可能先输出 base motion
```

```text
Dual-arm:
还要选择左手或右手
```

因此，跨机器人共享能力大致存在一个层次关系：

```text
Language / Task
↓
Semantic Plan
↓
Manipulation Skill
↓
EEF Motion
↓
Joint Motion
↓
Torque / Motor Command
```

越接近上层：

```text
越容易跨 embodiment
```

越接近底层：

```text
越依赖具体 embodiment
```

因此可以建立一个非常重要的结论：

```text
Task Understanding Generalization
≠
Motor Control Generalization
```

一个模型可能已经理解“应该做什么”，但依然无法正确控制全新的机器人身体。

---

## 16. 完整机器人实例：Franka、移动机械臂与双臂机器人共同学习 Pick-and-Place

考虑统一任务：

```text
把红色盒子从篮子 A 放到篮子 B。
```

### 16.1 共享任务结构

三个机器人都需要理解：

```text
识别红色盒子
↓
识别篮子 A 与 B
↓
定位目标物体
↓
接近目标
↓
抓取
↓
移动
↓
放置
↓
释放
```

这些属于跨 embodiment 可共享的信息。

---

### 16.2 Franka

可能输出：

$$
a_t^{F}
=
[
\Delta x,
\Delta y,
\Delta z,
\Delta r_x,
\Delta r_y,
\Delta r_z,
g
]
$$

---

### 16.3 移动机械臂

如果目标距离超出机械臂工作空间，先执行：

$$
a_t^{M,\text{base}}
=
[
v_x,
v_y,
\omega
]
$$

接近目标后，再输出机械臂动作。

因此其完整行为包含：

```text
Navigation
+
Manipulation
```

---

### 16.4 双臂机器人

策略还必须决定：

```text
使用左臂？
使用右臂？
双臂协同？
```

于是它的动作结构可能明显不同。

---

### 16.5 训练时真正希望共享的内容

模型最好能够把三种数据中的共同结构提取出来：

```text
red box
↓
graspable object
↓
basket A is source
↓
basket B is target
↓
pick precedes place
```

而具体执行部分则需要适配：

```text
Franka Control Space
Mobile Manipulator Control Space
Dual-arm Control Space
```

因此，Cross-Embodiment Learning 的目标并不是让三台机器人产生相同动作，而是：

> 让它们尽可能共享“为什么这样做、做什么、对象是谁、顺序是什么”，同时保留“如何用自己的身体完成”的差异。

---

## 17. Observation Heterogeneity

Action 并不是唯一问题。

不同机器人 observation 也可能完全不同。

例如：

```text
Robot A:
front RGB
```

```text
Robot B:
front RGB
wrist RGB
```

```text
Robot C:
head RGB
left wrist RGB
right wrist RGB
depth
```

本体状态也可能不同：

```text
7D joint state
10D mobile + arm state
dual-arm state
```

一种通用思路是把不同模态编码到共享 latent space。

视觉：

$$
z_{\text{vision}}
=
f_{\text{vision}}(I)
$$

机器人状态：

$$
z_{\text{state}}
=
f_{\text{state}}(s)
$$

语言：

$$
z_{\text{text}}
=
f_{\text{text}}(l)
$$

再组合：

$$
z
=
[
z_{\text{vision}},
z_{\text{text}},
z_{\text{state}}
]
$$

但这并没有完全解决问题。

仍然存在：

- camera viewpoint shift；
- missing modality；
- sensor configuration shift；
- wrist camera 有无差异；
- state dimension 差异。

因此 Cross-Embodiment 不只是 action 问题，也是 observation distribution 问题。

---

## 18. Negative Transfer：更多数据为什么可能更差？

一个常见误区是：

```text
More Robot Data
=
Always Better
```

实际上并不成立。

不同数据集质量和定义可能不一致。

### 18.1 最简单的语义冲突

例如：

```text
Dataset A:
gripper = 1 表示 close
```

而：

```text
Dataset B:
gripper = 1 表示 open
```

如果数据预处理没有统一，模型会收到互相冲突的监督信号。

---

### 18.2 更深层的策略冲突

即使动作语义已经统一，也可能存在行为差异。

例如：

```text
Robot A:
通常从正上方抓杯子
```

而：

```text
Robot B:
由于结构限制必须侧抓
```

如果模型没有足够的 embodiment information，它可能学习一个模糊的平均策略。

这种情况叫：

```text
Negative Transfer
```

即：

> 加入额外数据后，模型在某些任务或机器人上的表现反而下降。

---

## 19. Embodiment Conditioning

为了避免模型把不同机器人的动作混在一起，可以显式提供 embodiment 信息。

策略写成：

$$
\pi(a \mid o,l,e)
$$

其中 e 可以非常简单，例如：

```text
robot_id = "Franka"
```

也可以更加结构化，例如：

```text
DoF
Joint Limits
Kinematics
Gripper Type
Robot Morphology
```

进一步还可以使用：

```text
URDF
Link Geometry
Morphology Token
```

于是问题从：

```text
“你是 Robot 17”
```

升级为：

```text
“你的身体结构是什么？”
```

这引出了一个重要研究方向：

```text
Robot Morphology Representation
```

如果模型真正理解 robot morphology，就可能比简单记 robot ID 更有希望泛化到新 embodiment。

---

## 20. 容易混淆的概念

### 20.1 Unified Dataset Format vs Unified Action Space

**Unified Dataset Format**

解决：

```text
数据怎么存
怎么读
字段怎么组织
```

**Unified Action Space**

解决：

```text
动作维度与物理语义如何兼容
```

二者完全不同。

---

### 20.2 Action Normalization vs Action Unification

Normalization：

```text
统一数值尺度
```

Action Unification：

```text
统一或适配动作物理语义
```

Normalization 不能自动让 Joint Action 和 EEF Action 变成同一种动作。

---

### 20.3 Multi-Embodiment Training vs Cross-Embodiment Generalization

Multi-Embodiment Training：

```text
训练和测试都是已见机器人
```

更强意义的 Cross-Embodiment Generalization：

```text
测试机器人在训练中没有出现
```

阅读论文时必须确认测试设置。

---

### 20.4 Semantic Generalization vs Motor Generalization

模型可能正确理解：

```text
应该抓哪个物体
应该放到哪里
```

但仍然无法：

```text
用新机器人的身体正确执行
```

所以：

```text
理解任务成功
≠
底层控制成功
```

---

### 20.5 Action Tokenization vs Physical Action Alignment

Action Tokenization 只是：

```text
把动作转成模型可预测的离散或统一表示
```

它不自动解决：

```text
不同机器人的动作物理意义不一致
```

---

## 21. 与传统控制和 Robot Learning 的联系

Cross-Embodiment Learning 主要作用在：

```text
High-Level Task Understanding
↓
Learned Representation / VLA
↓
Action Representation
↓
Robot-Specific Mapping
↓
Controller
↓
Actuator
```

它并不意味着 VLA 直接取代：

- IK；
- Operational Space Controller；
- Joint Controller；
- Low-Level Motor Control。

在很多系统中，VLA 输出仍然是：

```text
EEF Target
Joint Target
Velocity Target
Action Chunk
```

然后由底层控制器执行。

所以本章研究的是：

> 如何让“策略学习层”利用多机器人数据获得更通用的能力，而不是让所有机器人的低层控制器完全相同。

---

## 22. Open X-Embodiment 在 VLA 发展链中的位置

可以把这一阶段的发展理解为：

```text
RT-1
↓
单一机器人体系上的大规模 generalist policy
```

```text
RT-2
↓
Web Vision-Language Knowledge
+
Robot Data
```

```text
Open X-Embodiment
↓
Many Robots
+
Many Datasets
+
Heterogeneous Data
```

后续：

```text
OpenVLA
π0
π0.5
```

进一步探索：

- 大规模 VLA；
- continuous action；
- cross-embodiment；
- generative action modeling；
- new robot adaptation。

因此 Open X-Embodiment 是从“单机器人 generalist policy”迈向“robot foundation model”的关键桥梁。

---

## 23. 主要研究瓶颈

### 23.1 Action Space Heterogeneity

最核心的问题之一仍然是：

```text
有没有足够通用的 Action Representation？
```

不同机器人可能使用：

- Joint Position；
- Joint Velocity；
- Torque；
- EEF Pose；
- Cartesian Delta；
- Base Velocity；
- Multi-arm Action。

越接近底层动作，越依赖具体 embodiment。

---

### 23.2 Observation Heterogeneity

包括：

- camera 数量不同；
- camera viewpoint 不同；
- wrist camera 缺失；
- depth 有无；
- proprioception 维度不同。

模型需要面对非常强的 observation distribution shift。

---

### 23.3 Dataset Quality 不一致

不同实验室可能使用：

- expert teleoperation；
- policy rollout；
- human intervention；
- 成功轨迹；
- 混合成功失败；
- 不同控制频率；
- 不同 timestamp 同步。

所以：

```text
D₁, D₂, ..., Dₙ
```

并不是自然同分布的数据。

---

### 23.4 Data Imbalance

不同数据集规模可能相差几个数量级。

如果不设计 Dataset Mixture，最大数据源会主导优化。

---

### 23.5 Negative Transfer

数据越多并不保证性能越好。

跨数据集定义冲突、任务策略冲突、动作语义冲突，都可能导致负迁移。

---

### 23.6 Embodiment Representation

简单的 robot ID 能够帮助区分机器人，但它不一定具有可泛化性。

真正值得研究的问题是：

> 模型能否理解一个此前未见机器人的 morphology 和控制能力？

---

### 23.7 Evaluation

评价 Cross-Embodiment 模型时，不能只看平均成功率。

必须确认：

- 测试机器人是否在训练中出现；
- 测试任务是否出现；
- 是否需要新机器人 demonstration；
- adaptation 需要多少数据；
- 是 zero-shot、few-shot 还是 full fine-tuning；
- 提升来自更多数据还是来自同平台数据量增加。

因此 Cross-Embodiment Benchmark 本身也是研究难点。

---

## 24. 研究视角：阅读论文时应主动追问什么？

以后阅读 Cross-Embodiment 或大规模 VLA 论文，应自然检查以下问题。

### 数据层

```text
使用了多少个 dataset？
多少种机器人？
每个机器人数据量是多少？
是否存在严重不平衡？
Dataset Mixture 权重如何设置？
```

### Observation 层

```text
相机数量是否一致？
Proprioception 如何处理？
缺失模态怎么办？
```

### Action 层

```text
使用 Joint 还是 Cartesian？
Absolute 还是 Delta？
不同 robot 如何对齐？
Normalization 如何做？
是否使用 Action Tokenizer？
```

### 模型层

```text
是否输入 robot ID？
是否存在 morphology embedding？
是否共享 action head？
是否存在 robot-specific decoder？
```

### Evaluation 层

```text
测试机器人训练时见过吗？
任务见过吗？
是真正 unseen embodiment 吗？
需要多少 adaptation data？
```

### 泛化层

必须区分：

```text
Task Generalization
Object Generalization
Scene Generalization
Embodiment Generalization
```

不要把所有成功都统称为：

```text
Generalization
```

---

## 25. 综合思考

### 问题 1

如果一个模型能同时控制 Franka、UR5 和 WidowX，但三个机器人都在训练集中出现过，这是否足以证明它具有强 Cross-Embodiment Generalization？

### 解析

不能。

它首先证明的是：

```text
Multi-Embodiment Learning
```

如果要证明更强的 Cross-Embodiment Generalization，需要测试：

```text
训练中没有出现过的新机器人
```

或者至少证明在新机器人上只需要极少数据即可快速适配。

---

### 问题 2

如果一个新双臂机器人测试失败，是否说明模型没有学到跨机器人知识？

### 解析

不一定。

模型可能已经学到了：

- 目标物体识别；
- 抓取与放置的语义顺序；
- 场景关系；
- 任务意图。

但它无法把这些知识映射到新机器人的：

- 双臂选择；
- joint configuration；
- control interface。

因此可能是：

```text
Semantic Transfer 成功
Motor Adaptation 失败
```

---

### 问题 3

如果加入一个新的机器人数据集后，所有机器人平均成功率下降，应优先检查什么？

### 解析

可以从以下方向分析：

```text
Action Semantics 是否统一？
↓
Normalization 是否一致？
↓
Dataset Quality 是否下降？
↓
Sampling Weight 是否不合理？
↓
是否出现 Negative Transfer？
↓
Embodiment Conditioning 是否足够？
```

不能简单得出：

```text
Cross-Embodiment 无效
```

因为问题也可能出在数据处理和 mixture design。

---

### 问题 4

为什么 Cross-Embodiment 越靠近任务语义层越容易共享，越靠近低层控制越困难？

### 解析

任务语义具有较强的平台不变性，例如：

```text
抓杯子
放进篮子
先抓后放
```

而底层控制直接由机器人本体决定：

```text
Joint Limits
Kinematics
Actuator
Controller
Morphology
```

因此可以形成一个重要连续谱：

```text
High-Level Semantics
        ↓
更容易共享

Low-Level Motor Control
        ↓
更依赖 Embodiment
```

---

## 26. 本章核心心智模型

整章最终应形成下面这套结构。

```text
不同机器人
↓
Observation 不同
Action 不同
Morphology 不同
Dataset Quality 不同
↓
统一数据组织
↓
Normalization
↓
Action Representation / Embodiment Conditioning
↓
Dataset Mixture
↓
Shared Representation Learning
↓
Robot-Specific Execution
↓
Multi-Embodiment Policy
↓
进一步追求 Unseen Robot Generalization
```

最重要的不是记住某一种具体技术，而是知道：

> Cross-Embodiment 的本质，是寻找“可以跨机器人共享的知识”与“必须保留 embodiment-specific 的执行信息”之间的边界。

---

## 27. 本章总结

### 结论一：Embodiment 是机器人完整的“身体—感知—控制”配置

它不仅包含机器人型号，还包含 morphology、sensor、DoF、action space 与控制接口。

---

### 结论二：Cross-Embodiment 共享的不是完全相同的底层动作

真正希望共享的是：

```text
视觉语义
任务理解
物体关系
操作结构
部分 skill representation
```

而底层执行仍然高度依赖机器人本体。

---

### 结论三：Normalization 不是 Action Space Unification

Normalization 只解决：

```text
数值尺度
```

无法自动解决：

```text
动作物理语义
```

这是本章必须牢固掌握的区别。

---

### 结论四：Dataset Mixture 决定不同机器人在训练中获得多少“话语权”

单纯 merge 数据并不能保证真正的 multi-embodiment learning。

采样概率、数据质量、数据规模和负迁移都需要控制。

---

### 结论五：Multi-Embodiment Training 不等于真正的新机器人泛化

如果测试机器人在训练中出现过，实验主要说明：

```text
一个模型可以控制多个已见机器人
```

真正更强的 Cross-Embodiment Generalization 关注：

```text
Unseen Robot
Zero-shot / Few-shot Adaptation
```

---

### 结论六：Open X-Embodiment 的价值在于推动机器人数据规模化

其历史意义不是“彻底解决了所有动作空间差异”，而是证明：

```text
Many Robots
+
Many Labs
+
Many Tasks
+
Heterogeneous Data
```

可以共同构建更通用的机器人策略基础。

它为后续 OpenVLA、π0 等更大规模 VLA 路线提供了重要的数据与研究范式基础。

---

## 28. 关键词

**Embodiment — 具身形态 — 机器人身体结构、传感器和控制接口的整体配置**

**Cross-Embodiment Learning — 跨具身学习 — 利用多种机器人数据学习可迁移策略与表示**

**Open X-Embodiment — 开放多机器人数据体系 — 汇集多机器人、多实验室异构数据用于大规模机器人学习**

**Heterogeneous Robot Dataset — 异构机器人数据集 — Observation、Action、频率和机器人结构均可能不同的数据集合**

**Action Normalization — 动作归一化 — 将不同动作数值尺度映射到统一范围**

**Action Space — 动作空间 — 策略能够输出的动作变量及其物理语义**

**Dataset Mixture — 数据集混合 — 按指定采样策略联合使用多个机器人数据集**

**Sampling Weight — 采样权重 — 控制不同数据集在训练中被抽取概率的参数**

**Embodiment Conditioning — 具身条件信息 — 显式告诉模型当前机器人结构或身份的信息**

**Robot Morphology — 机器人形态 — 关节、连杆、自由度、执行器和几何结构**

**Negative Transfer — 负迁移 — 加入其他数据后反而导致目标任务性能下降**

**Multi-Embodiment Training — 多机器人联合训练 — 同一个模型在多个已见机器人数据上训练**

**Cross-Embodiment Generalization — 跨机器人泛化 — 将能力迁移到新的机器人 embodiment**

**Task Generalization — 任务泛化 — 在已知或未知机器人上完成训练中未覆盖的新任务**

**Semantic Generalization — 语义泛化 — 对物体、语言、任务关系等高层知识进行迁移**

**Motor Generalization — 运动控制泛化 — 将策略迁移到不同机器人身体与动作接口**

**Action Tokenization — 动作 Token 化 — 将动作编码成模型可预测的离散 token 表示**

---

## 29. 本章在《VLA 与现代机器人学习基础》中的位置

```text
Robot Policy
↓
Transformer
↓
VLM
↓
Robot Learning / BC
↓
Distribution Shift / DAgger
↓
Robot Dataset
↓
Action Representation
↓
Action Chunking
↓
ACT
↓
Diffusion Policy
↓
Flow Matching
↓
RT-1
↓
RT-2
↓
Open X-Embodiment
↓
OpenVLA
↓
π0 / π0.5
↓
Fine-tuning / Agent / RL
```

RT-2 回答了：

```text
VLM 的知识如何进入机器人策略？
```

本章进一步回答：

```text
机器人数据本身如何跨平台规模化？
```

而接下来的 OpenVLA、π0 等模型会继续回答：

```text
如何在这种大规模异构机器人数据基础上训练真正的现代 VLA？
```
