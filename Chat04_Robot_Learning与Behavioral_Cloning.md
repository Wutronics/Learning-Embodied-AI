# 04｜Robot Learning 与 Behavioral Cloning：从专家示范到机器人策略

## 1. 本章导言

现代 Vision-Language-Action Model 最终需要解决的核心问题，并不是单纯“看懂图像”或者“理解语言”，而是：

> 给定机器人当前能够获得的信息，应该产生什么动作？

前面的 Transformer、Vision Transformer 和 Vision-Language Model 主要解决感知、表示和多模态信息融合问题。从本章开始，学习重点正式进入 Robot Learning：

> 机器人策略究竟如何从数据中学习动作？

其中最基础的方法就是行为克隆（Behavioral Cloning，BC）。

BC 的思想并不复杂：

> 让专家操作机器人，记录专家在不同 observation 下采取的 action，然后训练一个 policy 模仿这些行为。

但是，这个看似简单的过程实际上包含了后续整个现代机器人学习体系的许多基础概念：

- demonstration；
- trajectory；
- state；
- observation；
- action；
- policy；
- probabilistic policy；
- maximum likelihood；
- MSE；
- Cross Entropy；
- Supervised Fine-Tuning；
- Dataset 与 DataLoader；
- batch、epoch 与 training step；
- rollout；
- distribution shift。

理解这些概念以后，再阅读 ACT、Diffusion Policy、OpenVLA、π0 等模型时，复杂的网络结构将不再遮挡最底层的训练逻辑。

无论网络有多大、视觉编码器有多复杂，一个机器人模仿学习系统的基本问题仍然可以抽象为：

$$
\text{Observation}
\rightarrow
\text{Policy}
\rightarrow
\text{Action}
$$

而 Behavioral Cloning 的核心任务，就是利用专家数据学习这个映射。

---

# 2. 问题从哪里来：机器人如何“跟专家学”？

考虑一个最简单的 Pick-and-Place 任务：

> 机械臂抓起桌上的红色杯子并放入指定区域。

一种直接获得机器人策略的方法，是由人类专家通过遥操作设备控制机器人完成任务。

例如一次操作过程可能为：

```text
t = 0
机器人看到杯子位于右前方
专家控制机械臂向右前方移动

t = 1
末端靠近杯子
专家继续微调位置

t = 2
夹爪与杯子对齐
专家控制夹爪下降

t = 3
夹爪到达抓取位置
专家关闭夹爪

t = 4
杯子被抓住
专家控制机械臂向上移动
```

每一个时刻都可以记录：

```text
机器人看到什么
+
机器人自身状态
+
专家采取了什么动作
```

于是得到：

$$
(o_0,a_0^*),
(o_1,a_1^*),
(o_2,a_2^*),
\ldots,
(o_T,a_T^*)
$$

其中：

- oₜ 表示时刻 t 的 observation；
- aₜ* 表示专家在时刻 t 给出的 action；
- 星号通常强调这是 expert action 或 ground-truth action。

训练一个 policy：

$$
\pi_\theta
$$

希望它满足：

$$
\pi_\theta(o_t)
\approx
a_t^*
$$

这就是 Behavioral Cloning 最基本的出发点。

从机器学习角度看，它具有非常典型的 supervised learning 结构：

```text
Observation
    ↓
Policy
    ↓
Predicted Action
    ↓
与 Expert Action 比较
    ↓
Loss
    ↓
参数更新
```

但机器人学习与普通监督学习的区别，会在部署阶段逐渐显现。本章后半部分会回到这个问题。

---

# 3. Demonstration 与 Trajectory

## 3.1 Demonstration：专家演示

Demonstration 可以理解为：

> 专家向机器人展示的一次任务执行过程。

例如人类通过遥操作设备完成：

```text
靠近杯子
↓
对准杯子
↓
下降
↓
闭合夹爪
↓
抬起
↓
移动
↓
放入目标区域
```

这一整段执行过程可以称为一次 demonstration。

如果论文中写：

```text
500 demonstrations
```

通常表示采集了大约 500 次任务演示，而不是仅仅记录了 500 个控制动作。

一条 demonstration 内部往往包含几十、几百甚至更多 timestep。

---

## 3.2 Trajectory：时间序列轨迹

Trajectory 更强调数据在时间上的组织形式。

最简单可以表示为：

$$
\tau
=
(o_0,a_0,
o_1,a_1,
\ldots,
o_T,a_T)
$$

如果显式使用系统 state，则可能写成：

$$
\tau
=
(s_0,a_0,
s_1,a_1,
\ldots,
s_T,a_T)
$$

在强化学习中，还可能包含 reward：

$$
\tau
=
(s_0,a_0,r_0,
s_1,a_1,r_1,
\ldots)
$$

所以 trajectory 本质上强调：

> 系统随时间演化形成的一整串数据。

---

## 3.3 Demonstration 与 Trajectory 的关系

两者并不是简单的“包含关系”。

更加准确的心智模型是：

```text
Trajectory
│
├── Expert Trajectory
│      ↓
│  Demonstration
│
└── Policy Trajectory
       ↓
    Robot Rollout
```

如果一条 trajectory 是由专家操作机器人得到的，那么它可以作为 demonstration。

如果同样一条 trajectory 是机器人自己执行 policy 得到的，则它依然是一条 trajectory，但通常称为 policy rollout，而不是 expert demonstration。

因此：

> Trajectory 强调数据结构和时间序列。

> Demonstration 强调这条轨迹由专家提供，用来向机器人展示正确行为。

这一差异之后理解 DAgger、online data collection 和 reinforcement learning 时会非常重要。

---

# 4. 阶段思考 A：机器人数据中的基本对象

### 问题 1

Demonstration 与 trajectory 是什么关系？

### 解析

只说“trajectory 包含 demonstration”并不准确。

Trajectory 是更一般的时间序列概念。

专家产生的 trajectory 可以作为 demonstration；机器人自己 rollout 得到的 trajectory 则未必属于 demonstration。

因此应从“数据来源”而不是简单集合关系理解两者。

---

### 问题 2

专家遥操作机械臂完成一次：

```text
抓杯子
→
抬起
→
移动
→
放下
```

这是一个 action、一个 timestep，还是一条 trajectory？

### 解析

它是一条 trajectory。

其中每一个控制时刻对应一个 timestep，每个 timestep 内可能包含一个 observation 和一个 action。

---

### 问题 3

为什么 system state 和 observation 不能认为是同一个概念？

这一问题需要先进一步理解 state、observation 和 action。

---

# 5. State、Observation 与 Action

这是机器人学习中最容易混淆，也最重要的一组概念。

---

## 5.1 State：世界真实处于什么状态

State 表示：

> 描述当前系统状态所需的完整信息。

对于一个机械臂抓杯子的系统，完整 state 可能包含：

```text
机器人全部关节角
机器人全部关节速度
末端执行器位置
末端执行器姿态
夹爪状态

杯子的真实三维位置
杯子的真实姿态
杯子的速度

桌子位置
障碍物位置
接触状态
摩擦信息
……
```

抽象写成：

$$
s_t
$$

问题在于：

> 真实机器人通常无法直接获得完整 state。

例如杯子的真实世界坐标可能是：

```text
x = 0.5321 m
y = 0.1836 m
z = 0.0417 m
```

机器人并不会天然知道这些精确值。

它可能只能通过 RGB 摄像头间接判断杯子在哪里。

因此 state 描述的是：

> 世界真正是什么样。

而 observation 描述的是：

> 机器人实际上能够知道什么。

---

# 6. Observation：机器人能够观测到什么

机器人 policy 的输入通常不是完整 state，而是 observation。

例如：

$$
o_t
=
(I_t,q_t,g_t)
$$

其中：

- Iₜ：摄像头图像；
- qₜ：机器人关节位置；
- gₜ：夹爪状态。

在 VLA 中，还可能加入语言：

$$
o_t
=
(I_t,l,s_t^{robot})
$$

其中：

- Iₜ：视觉 observation；
- l：language instruction；
- sₜ robot：机器人 proprioception。

一个具体输入可能是：

```text
RGB Image:
224 × 224 × 3

Language:
"Pick up the red cup"

Joint Position:
7 dimensions

Gripper:
1 dimension
```

这些信息一起构成 policy 真正能够使用的 observation。

---

# 7. Proprioception：机器人对自身状态的感知

机器人论文中经常出现：

> Robot State

这个词需要谨慎理解。

论文所说的 robot state，很多时候实际上是：

> Proprioceptive Observation

例如：

```text
Joint Position
Joint Velocity
End-Effector Pose
Gripper State
```

这些量从物理意义上属于机器人真实状态的一部分。

但是因为机器人可以通过编码器、驱动器反馈或者其他传感器直接读取它们，所以当这些信息作为 policy 输入时，它们也属于 observation。

因此：

> 同一个物理变量可以是系统 state 的组成部分，同时又作为机器人能够观测到的 proprioceptive observation 输入模型。

真正需要区分的是：

```text
完整环境状态
vs
机器人实际可获得的信息
```

---

# 8. 从控制理论理解 State 与 Observation

对于具有控制背景的读者，可以将这一关系类比到状态空间系统。

传统系统中：

$$
x_{t+1}
=
f(x_t,u_t)
$$

观测为：

$$
y_t
=
h(x_t)
$$

其中：

- xₜ 表示真实系统状态；
- yₜ 表示测量输出；
- h 表示传感或观测过程。

机器人学习中可以建立类似心智模型：

```text
完整真实状态 sₜ
        ↓
传感器 / 摄像头 / 编码器
        ↓
Observation oₜ
```

因此通常：

$$
s_t
\neq
o_t
$$

而可以抽象认为：

$$
o_t
=
h(s_t)
$$

现实机器人中的 h 往往远比传统线性测量矩阵复杂。

例如 RGB camera 就是一种高维、非线性的 observation mechanism。

---

# 9. Action：策略究竟输出什么

Policy 最终产生：

$$
a_t
=
\pi_\theta(o_t)
$$

Action 可以采用很多形式，例如：

```text
Joint Position
Joint Velocity
Joint Torque

End-Effector Position
End-Effector Pose
Cartesian Velocity
Cartesian Delta

Gripper Command
```

例如一种机械臂 action 可以写成：

$$
a_t
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

其中：

- Δx、Δy、Δz 表示末端位置增量；
- Δr 表示末端姿态变化；
- g 表示夹爪指令。

Action Representation 本身将在后续专门学习。本章暂时只需要理解：

> BC 的监督标签就是 expert action。

---

# 10. 阶段思考 B：State / Observation / Action 辨析

### 问题 1

RGB camera image 属于 state、observation 还是 action？

### 解析

属于 observation。

它是摄像头对真实世界状态进行测量之后产生的信息。

---

### 问题 2

Joint position 属于 state 还是 observation？

### 解析

这个问题不能机械回答。

从物理系统角度，它是 robot state 的组成部分。

但当机器人通过编码器读取 joint position，并把它输入 policy 时，它也是 proprioceptive observation。

所以阅读论文时，应判断作者是在谈：

```text
物理状态定义
```

还是：

```text
模型实际输入
```

---

### 问题 3

完整 environment state 还可能包含什么？

例如：

```text
目标物体真实 6D pose
物体速度
接触状态
障碍物状态
摩擦条件
环境中其他物体的状态
```

这些信息不一定能够直接被机器人获得。

---

### 问题 4

为什么通常有：

$$
s_t
\neq
o_t
$$

### 解析

因为 observation 通常只是完整 state 经过传感器得到的一部分信息。

现实机器人往往处于 partial observability 条件下。

---

# 11. Behavioral Cloning 的核心目标

现在假设已经获得数据：

$$
D
=
\{(o_i,a_i^*)\}_{i=1}^{N}
$$

其中：

- oᵢ 是第 i 个 observation；
- aᵢ* 是对应的 expert action；
- N 是训练样本数量。

训练一个 policy：

$$
\pi_\theta
$$

希望预测：

$$
\hat a_i
=
\pi_\theta(o_i)
$$

并满足：

$$
\hat a_i
\approx
a_i^*
$$

真正的问题变成：

> 怎样定义“预测动作和专家动作足够接近”？

这就是 loss function 的作用。

---

# 12. 连续动作：最简单的 MSE Behavioral Cloning

假设机器人 action 是：

$$
a
=
[
\Delta x,
\Delta y,
\Delta z
]
$$

专家给出：

$$
a^*
=
[
0.02,
-0.01,
-0.03
]
$$

模型预测：

$$
\hat a
=
[
0.018,
-0.015,
-0.025
]
$$

一种直接方法是计算平方误差：

$$
L
=
\|
\hat a-a^*
\|_2^2
$$

对于整个数据集：

$$
L_{BC}
=
\frac{1}{N}
\sum_{i=1}^{N}
\|
\pi_\theta(o_i)-a_i^*
\|_2^2
$$

这就是最简单的 continuous-action Behavioral Cloning。

训练过程为：

```text
Observation
    ↓
Policy
    ↓
Predicted Action
    ↓
Expert Action
    ↓
MSE
    ↓
Backpropagation
    ↓
Update Parameters
```

---

# 13. 更一般的 BC：学习条件动作分布

MSE 并不是 BC 最一般的形式。

现代 robot learning 中更常写：

$$
\pi_\theta(a|o)
$$

它表示：

> 给定 observation o，policy 对不同 action 的概率分布。

例如：

```text
当前 observation：
夹爪已经靠近杯子

Policy：

继续向前         0.70
向左调整         0.10
向右调整         0.05
关闭夹爪         0.15
```

Behavioral Cloning 希望：

> 专家真正执行的 action 在模型概率分布中具有尽可能高的概率。

因此希望最大化：

$$
\prod_i
\pi_\theta(a_i^*|o_i)
$$

但大量概率相乘不便于优化，而且容易出现数值问题，因此通常取 logarithm：

$$
\sum_i
\log
\pi_\theta(a_i^*|o_i)
$$

训练中通常采用最小化 loss 的形式，因此变为：

$$
L_{BC}
=
-
\mathbb E_{(o,a^*)\sim D}
\left[
\log
\pi_\theta(a^*|o)
\right]
$$

这就是 BC 更一般的 maximum likelihood formulation。

直觉仍然很简单：

> 专家执行了哪个 action，就让模型认为这个 action 更可能。

---

# 14. 为什么 Maximum Likelihood 会得到 MSE？

这是本章重要的数学联系之一。

假设 continuous action 的 policy 被建模为 Gaussian：

$$
\pi_\theta(a|o)
=
\mathcal N
\left(
\mu_\theta(o),
\sigma^2I
\right)
$$

其中：

- μθ(o) 是网络根据 observation 输出的动作均值；
- σ² 表示固定 variance；
- I 是单位矩阵。

Gaussian likelihood 可以写成：

$$
\pi_\theta(a^*|o)
=
\frac{1}
{(2\pi\sigma^2)^{d/2}}
\exp
\left(
-
\frac{
\|a^*-\mu_\theta(o)\|^2
}
{2\sigma^2}
\right)
$$

取 negative log：

$$
-
\log
\pi_\theta(a^*|o)
=
C
+
\frac{1}{2\sigma^2}
\|
a^*-\mu_\theta(o)
\|^2
$$

其中 C 与模型预测无关。

如果 σ² 固定，那么优化过程中：

```text
C 不影响最优参数
1 / 2σ² 只是一个固定比例
```

因此最小化 negative log likelihood，等价于最小化：

$$
\|
a^*-\mu_\theta(o)
\|^2
$$

也就是 MSE。

因此正确关系是：

```text
Behavioral Cloning
↓
学习 p(a | o)
↓
Maximum Likelihood
↓
Negative Log Likelihood
↓
假设固定方差 Gaussian
↓
MSE
```

---

# 15. 阶段纠偏：MSE 与 Maximum Likelihood 的关系

一种容易出现的理解是：

> Gaussian 最大概率点和 MSE 最小值恰好落在同一个位置，所以二者差不多。

这个理解方向接近，但不够准确。

真正的关系是：

> 在固定方差 Gaussian policy 假设下，negative log likelihood 经过数学展开后，本身就等于一个常数项加上缩放后的 squared error。

因此两者不是“结果恰好类似”，而是在这一建模假设下具有明确的数学等价关系。

这一区别在以后遇到更加复杂的 action distribution 时非常重要。

---

# 16. 离散 Action：Cross Entropy

如果 action 是离散的，例如夹爪只有：

```text
OPEN
CLOSE
```

模型可能输出：

```text
P(OPEN | o)  = 0.2
P(CLOSE | o) = 0.8
```

如果专家动作是：

```text
CLOSE
```

则 loss 为：

$$
L
=
-\log 0.8
$$

这就是典型的 Cross Entropy / Negative Log Likelihood。

因此可以形成基本对应：

| Action 表示 | 模型输出 | 常见训练目标 |
|---|---|---|
| Continuous Position / Velocity | Continuous Value | MSE / Gaussian NLL |
| Discrete Action | Class Probability | Cross Entropy |
| Action Token | Token Probability | Cross Entropy |
| Complex Continuous Distribution | Distribution / Generative Process | 更复杂生成式目标 |

最后一类将在 Diffusion Policy 和 Flow Matching 中继续学习。

---

# 17. 为什么要使用概率形式，而不是简单回归？

机器人动作并不总是只有唯一正确答案。

假设机械臂需要绕过一个障碍物去抓杯子：

```text
           Obstacle
              █

Robot                    Cup
```

可能存在两条正确路径：

```text
左侧绕行
```

或者：

```text
右侧绕行
```

因此对于同一个 observation：

$$
p(a|o)
$$

可能拥有多个高概率区域。

这被称为：

> Multimodal Action Distribution。

这也是为什么机器人策略不能永远只被理解为：

```text
Observation
→
唯一正确 Action
```

更一般的理解应该是：

```text
Observation
→
Conditional Action Distribution
→
从中产生合理 Action
```

---

# 18. MSE 的 Mode Averaging 问题

考虑一个极简例子。

同一个 observation 下，专家一半时间选择：

$$
a=-1
$$

表示向左绕。

另一半选择：

$$
a=+1
$$

表示向右绕。

如果使用普通 squared error：

$$
L(a)
=
(a+1)^2
+
(a-1)^2
$$

展开：

$$
L(a)
=
2a^2+2
$$

求最小值：

$$
a=0
$$

但是：

```text
-1 → 左绕
+1 → 右绕
0  → 直走
```

如果中间正好是障碍物，那么：

> MSE 学出来的平均动作可能是专家从未执行过、甚至会导致失败的动作。

这就是：

> Mode Averaging。

更准确地说：

> MSE 的最优预测倾向于条件均值。

因此，MSE 并不是“只要存在多个答案就完全不能训练”，而是：

> 当真实 action distribution 是 multimodal 时，单峰回归目标可能把多个合理 mode 平均到一个不合理位置。

这一问题是后续 Diffusion Policy、Flow Matching 等生成式动作模型的重要动机之一。

---

# 19. 阶段思考 C：BC 的概率建模

### 问题 1

πθ(a|o) 表示什么？

### 解析

表示：

> 给定 observation o 后，policy 对 action a 的条件概率分布。

它不是单纯“一个动作值”。

---

### 问题 2

为什么 BC 可以写成：

$$
-
\log
\pi_\theta(a^*|o)
$$

### 解析

因为 Behavioral Cloning 希望提高专家动作的 likelihood。

对于整个 dataset，就是希望专家 action 的联合 likelihood 最大。

取 log 后将概率乘积变为 log likelihood 求和，再将 maximization 改写为 minimization，就得到 negative log likelihood。

---

### 问题 3

为什么 Gaussian continuous policy 会自然导出 MSE？

### 解析

关键不是“Gaussian 的峰值和 MSE 最小点相同”，而是：

> 固定 variance Gaussian 的 negative log likelihood 经过展开后，正比于 squared error。

因此这是明确的数学等价关系。

---

### 问题 4

OPEN / CLOSE 类型 action 为什么更自然使用 Cross Entropy？

### 解析

因为它本质上是离散分类变量。

模型输出的是每个类别的 probability，而不是连续数值距离。

---

### 问题 5

为什么 multimodal action distribution 会给普通 MSE 带来问题？

### 解析

MSE 倾向预测条件平均值。

但机器人任务中的多个有效策略可能属于完全不同的行为 mode。

平均两个合法动作不一定仍然合法。

---

# 20. Behavioral Cloning 与 Supervised Fine-Tuning

在现代 VLA 论文中，经常同时出现两个词：

```text
Behavioral Cloning
Supervised Fine-Tuning
```

二者并不是完全不同的学习原理。

关键区别是观察角度。

---

## 20.1 Robot Learning 视角

如果数据是：

$$
(o,a^*)
$$

并且目标是让 policy 模仿 expert action，那么称为：

> Behavioral Cloning。

强调的是：

```text
专家行为
↓
模仿
↓
Robot Policy
```

---

## 20.2 Foundation Model 视角

如果已经有一个 pretrained model，现在使用带 target 的 supervised data 继续训练，则通常称为：

> Supervised Fine-Tuning，SFT。

强调的是：

```text
Pretrained Model
↓
Supervised Dataset
↓
Fine-Tuning
```

---

## 20.3 二者的关系

在许多 VLA 训练场景中：

$$
\text{VLA SFT}
\approx
\text{BC on Robot Demonstrations}
$$

也就是说：

> 从 robot learning 角度，它是在做 BC。

> 从 large-model training 角度，它是在做 SFT。

因此不能把它们理解成两个互不相关的算法。

它们更多是在不同研究传统中，对同一个监督式模仿训练过程采用不同语言描述。

---

# 21. Action Token 与 VLA 中的监督训练

某些 VLA 不直接回归 continuous action，而是将动作离散化。

例如原动作：

$$
[
0.12,
-0.04,
0.31,
\ldots
]
$$

经过 action tokenizer 后可能变为：

```text
<ACTION_51>
<ACTION_103>
<ACTION_87>
```

此时模型输入可能是：

```text
RGB Image
+
Language Instruction
+
Robot State
```

输出：

```text
ACTION TOKEN
```

例如：

```text
ACTION_51    0.70
ACTION_22    0.20
ACTION_13    0.10
```

专家 target 为：

```text
ACTION_51
```

那么训练目标就可以使用 Cross Entropy。

从形式上看，这和 language model 的 token prediction 非常相似。

但是从 robot learning 角度，它仍然是在：

> 模仿专家 action。

因此前面的 Transformer、VLM 与本章 Robot Learning 开始真正连接起来：

```text
Image
↓
Visual Encoder
↓
Visual Tokens

Language
↓
Language Tokens

Robot State
↓
State Representation

        ↓
Multimodal Policy
        ↓
Robot Action
```

---

# 22. 从 Demonstration 到 Dataset

现在进入完整训练流程。

假设采集了 M 条 demonstration：

$$
D
=
\{
\tau_1,
\tau_2,
\ldots,
\tau_M
\}
$$

第 j 条 trajectory 可以写为：

$$
\tau_j
=
\{
(o_0,a_0^*),
(o_1,a_1^*),
\ldots,
(o_{T_j},a_{T_j}^*)
\}
$$

需要特别区分两个数据规模：

```text
Trajectory Number
vs
Timestep Number
```

例如：

```text
1000 demonstrations

每条约 200 timesteps
```

则大约包含：

```text
200000 observation-action pairs
```

因此论文中：

```text
1000 trajectories
```

和：

```text
200000 action steps
```

描述的是不同层次的数据规模。

---

# 23. Dataset 与 DataLoader

## 23.1 Dataset

Dataset 表示完整的数据集合。

例如：

```text
200000 training samples
```

每个 sample 可能包含：

```text
Image
Language
Robot State
Expert Action
```

---

## 23.2 DataLoader

DataLoader 并不是另一个数据集。

它是：

> 负责从 Dataset 中按照训练需要读取和组织数据的机制。

典型功能包括：

```text
Shuffle
Batching
Sampling
Parallel Loading
```

例如：

```text
Dataset
200000 samples
        ↓
DataLoader
        ↓
Batch 1: 32 samples
Batch 2: 32 samples
Batch 3: 32 samples
...
```

因此：

> Dataset 回答“数据是什么”。

> DataLoader 回答“训练时怎么把这些数据送给模型”。

---

# 24. Batch

如果 batch size 为：

```text
B = 32
```

那么一次 training step 中可能得到：

$$
\{
(o_i,a_i^*)
\}_{i=1}^{32}
$$

模型不会一次加载整个百万级 dataset。

而是不断处理 mini-batch。

例如：

```text
Batch 1
↓
Forward
↓
Loss
↓
Backward
↓
Update

Batch 2
↓
Forward
↓
Loss
↓
Backward
↓
Update
```

---

# 25. Observation 如何进入模型

原始 observation 可能为：

```text
RGB Image:
224 × 224 × 3

Language:
"Pick up the red cup"

Joint State:
7

Gripper:
1
```

这些信息不能直接以原始形式送进统一 Transformer。

通常需要 preprocessing / encoding：

```text
RGB Image
↓
Vision Encoder
↓
Visual Tokens


Language
↓
Tokenizer
↓
Language Tokens


Robot State
↓
State Encoder
↓
State Embedding
```

之后再进入 multimodal policy。

因此前面学习的 Transformer 和 VLM 并不是独立知识，而是在这里成为 Robot Policy 的表示基础。

---

# 26. Forward Pass：Policy 预测动作

最简单的 continuous-action BC：

$$
\hat a
=
\pi_\theta(o)
$$

假设 batch size 为 32，action dimension 为 7。

则可能有：

```text
Observation Batch
↓
Policy
↓
Predicted Action

Shape:
32 × 7
```

Dataset 同时给出 expert action：

```text
Expert Action Shape:
32 × 7
```

于是可以直接计算 batch loss。

---

# 27. Loss

对于 MSE：

$$
L
=
\frac{1}{B}
\sum_{i=1}^{B}
\|
\hat a_i-a_i^*
\|^2
$$

其中：

- B 为 batch size；
- âᵢ 是 policy prediction；
- aᵢ* 是 expert action。

Loss 回答的问题是：

> 当前 policy 的预测和 expert behavior 相差多少？

但仅仅知道 loss 还不够。

还需要知道：

> 模型参数应该向什么方向变化？

---

# 28. Backpropagation

训练中计算：

$$
\nabla_\theta L
$$

它表示：

> Loss 对 policy parameters 的 gradient。

直觉上是在回答：

```text
哪个参数增大一点会让 loss 上升？
哪个参数减小一点会让 loss 下降？
每个参数对当前误差有多大贡献？
```

PyTorch 中通常对应：

```python
loss.backward()
```

这一操作的核心结果是：

> 得到 gradients。

它本身并不直接完成参数更新。

---

# 29. Optimizer

Optimizer 根据 gradient 更新参数。

最简单的 gradient descent：

$$
\theta_{new}
=
\theta_{old}
-
\eta
\nabla_\theta L
$$

其中：

- θ 表示模型参数；
- η 表示 learning rate；
- ∇θL 表示 loss gradient。

实际深度学习中常用：

```text
Adam
AdamW
```

虽然更新规则更加复杂，但基本思想没有改变：

> 根据梯度信息调整参数，使 loss 逐渐下降。

因此三个概念要严格区分：

```text
Loss
↓
衡量预测有多错

Backpropagation
↓
计算每个参数对应的梯度

Optimizer
↓
根据梯度真正修改参数
```

---

# 30. Epoch 与 Training Step

假设：

```text
Dataset Size = 100000
Batch Size   = 100
```

那么完整遍历 Dataset 一次，大约需要：

```text
1000 training steps
```

这次完整遍历称为：

> 1 epoch。

因此：

```text
Sample
↓
一个训练样本

Batch
↓
一组 samples

Training Step
↓
一次 batch forward + backward + update

Epoch
↓
整个 training dataset 被遍历一次
```

---

# 31. 一个最简 PyTorch Behavioral Cloning 训练循环

```python
policy = Policy()
optimizer = Adam(policy.parameters())

for epoch in range(num_epochs):

    for observation, expert_action in dataloader:

        predicted_action = policy(observation)

        loss = mse_loss(
            predicted_action,
            expert_action
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
```

逐步对应：

```python
predicted_action = policy(observation)
```

代表：

$$
\hat a
=
\pi_\theta(o)
$$

---

```python
loss = mse_loss(
    predicted_action,
    expert_action
)
```

代表：

$$
L
=
\|
\hat a-a^*
\|^2
$$

---

```python
loss.backward()
```

代表计算：

$$
\nabla_\theta L
$$

---

```python
optimizer.step()
```

代表：

$$
\theta
\rightarrow
\theta'
$$

这一训练骨架非常重要。

未来 ACT、Diffusion Policy、OpenVLA 和其他 VLA 的 architecture、loss 和 data structure 会复杂很多，但大量训练代码仍然可以从这个框架理解。

---

# 32. 阶段思考 D：完整训练流水线

### 问题 1

Dataset 与 DataLoader 有什么区别？

### 解析

Dataset 是完整数据集合。

DataLoader 是训练过程中读取、shuffle 和组成 mini-batch 的机制。

将 DataLoader 理解为“每轮训练的数据”不够准确，因为它并不是数据本身，而是数据加载器。

---

### 问题 2

1000 条 trajectory，每条平均 200 timestep，大约有多少 observation-action pairs？

答案约为：

```text
200000 pairs
```

这里检查的是：

> trajectory-level dataset size 和 timestep-level dataset size 不应混淆。

---

### 问题 3

loss.backward() 与 optimizer.step() 有什么区别？

### 解析

```text
loss.backward()
↓
Gradient Computation
```

而：

```text
optimizer.step()
↓
Parameter Update
```

反向传播并不等价于优化器更新参数。

---

### 问题 4

Epoch 与 training step 有什么区别？

### 解析

Training step 通常对应一个 batch 的一次参数更新。

Epoch 表示整个 dataset 被遍历一遍。

---

# 33. 完整机器人实例：从遥操作数据到抓取 Policy

现在用一个完整 Pick-and-Place 例子把本章所有概念串起来。

任务：

> 将桌面上的红色盒子抓起并放入右侧篮子。

---

## 33.1 数据采集

专家通过遥操作完成 500 次任务。

一条 trajectory 包含：

```text
RGB Camera
Joint Position
Gripper State
Language Instruction
Expert Cartesian Delta Action
```

例如：

```text
Observation:

Image:
640 × 480 RGB

Instruction:
"Move the red box to the right basket"

Robot State:
q = [q₁, q₂, ..., q₇]
gripper = 1


Expert Action:

Δx = +0.018
Δy = -0.004
Δz = -0.021

Δroll  = 0
Δpitch = 0.006
Δyaw   = -0.002

gripper = open
```

---

## 33.2 构成 Dataset

500 条 trajectory，每条例如 300 timestep：

```text
约 150000 timestep samples
```

每个样本可以抽象为：

$$
(o_i,a_i^*)
$$

---

## 33.3 DataLoader

设置：

```text
Batch Size = 32
Shuffle = True
```

每次取出：

```text
32 images
32 robot states
32 language instructions
32 expert actions
```

---

## 33.4 Policy Forward

视觉信息经过 visual encoder：

```text
Image
↓
Visual Tokens
```

语言经过 tokenizer：

```text
Instruction
↓
Language Tokens
```

机器人状态经过 embedding：

```text
Joint State
↓
State Features
```

最终输入 policy：

```text
Visual Information
+
Language Information
+
Robot State
↓
Policy
↓
Predicted Action
```

---

## 33.5 Loss

如果 action 为 continuous Cartesian delta：

$$
L
=
\|
\hat a-a^*
\|^2
$$

或者使用概率建模：

$$
L
=
-
\log
\pi_\theta(a^*|o)
$$

---

## 33.6 Backpropagation 与 Optimization

模型计算：

$$
\nabla_\theta L
$$

Optimizer 更新：

$$
\theta
\leftarrow
\theta
-
\eta
\nabla_\theta L
$$

经过大量 batch 和多个 epoch 后得到训练完成的 policy。

---

# 34. Policy Rollout

训练完成后，需要真正让机器人自己执行。

这叫：

> Policy Rollout。

过程为：

```text
当前 Observation
↓
Policy
↓
Action
↓
Robot Executes
↓
Environment Changes
↓
New Observation
↓
Policy
↓
Next Action
↓
...
```

数学形式为：

$$
o_0
\xrightarrow{\pi}
a_0
\rightarrow
o_1
\xrightarrow{\pi}
a_1
\rightarrow
o_2
\xrightarrow{\pi}
a_2
\rightarrow
\cdots
$$

这里出现一个 Behavioral Cloning 中非常重要的边界：

> 普通 BC 的 rollout 本身不会让 policy 自动变好。

Rollout 是部署和 evaluation。

如果机器人失败：

```text
BC 不会因为一次失败
自动获得正确 action label。
```

除非额外引入：

```text
Expert Relabeling
DAgger
Reward
RL
Human Intervention
Online Learning
```

否则 vanilla BC 的训练仍然只依赖原始 expert dataset。

---

# 35. 阶段纠偏：BC 并不依赖“不断 Rollout 来优化”

一种容易产生的错误流程是：

```text
Demonstration
↓
Dataset
↓
Pretrained Model
↓
Rollout
↓
不断得到更好的 Policy
```

这里包含两个问题。

第一：

> Behavioral Cloning 本身不要求必须从 pretrained model 开始。

可以使用 pretrained backbone，也可以从随机初始化的 policy 开始训练。

第二：

> Rollout 本身并不会产生参数更新。

正确的 BC 主流程是：

```text
Expert Demonstration
↓
Dataset
↓
Policy Prediction
↓
Compare with Expert Action
↓
Loss
↓
Gradient
↓
Optimizer
↓
Updated Policy
```

训练完成以后：

```text
Policy
↓
Rollout
↓
Evaluation
```

如果希望 rollout 中的新数据继续影响策略，就已经开始进入 DAgger、online learning 或 reinforcement learning 的范围。

---

# 36. Robot Learning 与普通监督学习的根本差异

到这里会出现一个极其重要的问题：

> 如果 BC 本质上是 supervised learning，为什么机器人 imitation learning 往往比普通 classification 更容易失败？

因为机器人 policy 的输出会改变未来输入。

普通图片分类：

```text
Image₁
↓
Classifier
↓
Prediction

Image₂
```

即使 Image₁ 分类错误，也不会因为这个错误自动改变 Image₂。

但是机器人系统是：

```text
Observationₜ
↓
Policy
↓
Actionₜ
↓
Environment
↓
Observationₜ₊₁
```

因此：

> 当前 prediction error 会影响未来 state。

这使 robot learning 天然成为一个闭环序列决策问题。

---

# 37. Behavioral Cloning 的核心局限：训练分布与执行分布不同

训练数据来自 expert：

$$
o
\sim
p_E(o)
$$

但是 rollout 时 observation 来自机器人自己的 policy：

$$
o
\sim
p_{\pi}(o)
$$

一般而言：

$$
p_E(o)
\neq
p_{\pi}(o)
$$

这就是 Distribution Shift 的基本来源。

---

# 38. 一个具体的抓取失败过程

专家理想轨迹：

```text
Step 0
夹爪正对杯子

Step 1
向前

Step 2
对准

Step 3
下降

Step 4
抓取
```

假设 learned policy 在第一个动作中只有 1 mm 误差。

于是机器人并没有到达 expert 本来会到达的位置。

下一时刻：

```text
Expert State
≠
Policy State
```

于是：

```text
Expert Observation
≠
Policy Observation
```

机器人进入了一个训练数据较少覆盖的区域。

如果新的 observation 更陌生，prediction error 可能进一步变大：

```text
1 mm error
↓
Different State
↓
Less Familiar Observation
↓
Larger Error
↓
Even More Different State
↓
Failure
```

这种现象称为：

> Compounding Error。

---

# 39. Distribution Shift 与 Compounding Error 的关系

二者不是同一个概念。

Distribution Shift 指：

> Policy rollout 遇到的 observation distribution 与 expert dataset 中的 observation distribution 不同。

Compounding Error 指：

> 一个时间步产生的小误差不断影响未来 state，使后续误差沿 trajectory 累积并可能越来越严重。

二者具有紧密因果联系：

```text
Single-Step Prediction Error
↓
Robot Reaches Different State
↓
Observation Distribution Shifts
↓
Policy Encounters Less Familiar States
↓
Prediction Error Increases
↓
Further State Deviation
↓
Compounding Error
```

---

# 40. 为什么“完美专家数据”也可能存在问题？

假设 expert 永远能够精准对准杯子。

那么 dataset 中大量数据可能是：

```text
已经基本对准杯子
↓
应该怎样完成抓取
```

但真实 policy rollout 可能出现：

```text
夹爪偏左 3 cm
```

此时机器人真正需要学习的是：

> 已经偏离轨迹之后，怎样 recovery？

问题在于：

> 一个始终执行完美的专家，可能从来没有访问过这些失败附近的 state。

于是 training dataset 中缺乏：

```text
Recovery State
→
Corrective Action
```

这形成一个非常重要的 robot learning 现象：

> Expert demonstrations 往往大量覆盖“成功轨迹附近”的状态，却不一定覆盖 learned policy 实际会犯错到达的状态。

这正是后续 DAgger 要解决的问题之一。

---

# 41. 与传统控制系统的联系

BC / VLA policy 并不等价于整个机器人控制系统。

典型机器人系统可以抽象为：

```text
High-Level Task
↓
Language / Task Command
↓
Learned Policy / VLA
↓
Action Target
↓
Robot Controller
↓
Actuator
↓
Physical Robot
```

本章讨论的 BC 主要工作在：

```text
Observation
↓
Learned Policy
↓
Action Representation
```

这一层。

例如 policy 输出：

```text
End-Effector Delta Pose
```

底层可能还需要：

```text
Inverse Kinematics
↓
Joint Controller
↓
Motor Command
```

因此：

> Behavioral Cloning 学习的是策略层的 action generation，不一定直接替代底层控制器。

对于具有控制背景的研究者，理解这一区分尤其重要。

---

# 42. 容易混淆的概念

## 42.1 Demonstration vs Trajectory

Trajectory：

> 任意来源的时序轨迹。

Demonstration：

> 专家产生、用于学习的轨迹。

---

## 42.2 State vs Observation

State：

> 世界完整真实状态。

Observation：

> 机器人能够测量或获得的信息。

通常：

$$
o_t
=
h(s_t)
$$

---

## 42.3 Robot State vs Environment State

Robot state 在论文中常常主要指：

```text
Joint Position
Joint Velocity
EEF Pose
Gripper
```

Environment state 则可能包含：

```text
Objects
Obstacles
Contacts
Other Agents
Environment Dynamics
```

---

## 42.4 MSE vs Behavioral Cloning

MSE 不是 BC 本身。

BC 是：

> 模仿 expert policy 的训练范式。

MSE 只是其中一种可能的 loss。

---

## 42.5 Maximum Likelihood vs MSE

在固定方差 Gaussian policy 假设下：

```text
Maximum Likelihood
↓
Negative Log Likelihood
↓
Squared Error
```

这是数学关系，不是单纯经验对应。

---

## 42.6 Behavioral Cloning vs SFT

BC：

> Robot Learning 语境。

SFT：

> Foundation Model / Large Model 语境。

在 robot demonstration 上对 VLA 做 supervised fine-tuning 时，两者往往描述同一个训练过程的不同侧面。

---

## 42.7 Dataset vs DataLoader

Dataset：

> 数据内容。

DataLoader：

> 数据读取和 batch 组织机制。

---

## 42.8 Backpropagation vs Optimizer

Backpropagation：

> 计算 gradient。

Optimizer：

> 根据 gradient 更新参数。

---

## 42.9 Training vs Rollout

Training：

> 使用 expert target 计算 loss 并更新参数。

Rollout：

> Policy 自己控制机器人产生 trajectory。

普通 BC 的 rollout 本身不会自动训练模型。

---

# 43. 本章在 VLA 知识链中的位置

完整学习链可以写成：

```text
Robot Policy
↓
Transformer
↓
Vision-Language Model
↓
Robot Learning / Behavioral Cloning
↓
Action Representation
↓
Action Chunking
↓
Diffusion / Flow Matching
↓
Vision-Language-Action Model
↓
Fine-Tuning / Agent / RL
```

本章第一次把前面的“模型如何理解输入”和后面的“机器人如何产生动作”真正连接起来。

前面学习：

```text
Image
→
Visual Representation

Language
→
Language Representation
```

本章开始回答：

```text
这些 Representation
如何最终监督出
Robot Action？
```

因此 Behavioral Cloning 是从 VLM 走向 VLA 的核心基础之一。

---

# 44. 研究视角：阅读 Robot Learning / VLA 论文时应该问什么？

以后看到一个 imitation-learning 或 VLA 系统，不能只看模型名字。

首先应该检查以下设计。

### 数据来自哪里？

```text
Human Teleoperation?
Scripted Expert?
Existing Robot Dataset?
Cross-Embodiment Dataset?
```

### 一共有多少数据？

需要区分：

```text
Episodes
Trajectories
Timesteps
Hours
```

### Observation 包含什么？

例如：

```text
One RGB Camera?
Multiple Cameras?
Depth?
Proprioception?
Language?
History?
```

### Action Representation 是什么？

例如：

```text
Joint Position
EEF Pose
Delta Pose
Velocity
Action Token
Action Chunk
```

### Policy 建模什么分布？

例如：

```text
Deterministic Regression
Gaussian Policy
Categorical Distribution
Diffusion
Flow Matching
```

### Training Objective 是什么？

例如：

```text
MSE
Cross Entropy
Negative Log Likelihood
Generative Objective
```

### Evaluation 是什么？

仅看 offline action prediction loss 远远不够。

Robot policy 最终还需要看：

```text
Rollout Success Rate
Task Completion
Failure Modes
Robustness
Recovery Ability
```

这也是为什么机器人论文里：

> Validation Loss 很低

并不能自动推出：

> Real-Robot Success Rate 很高。

---

# 45. 从 Offline Accuracy 到 Closed-Loop Success

这是本章最值得形成的研究心智模型之一。

在普通 supervised learning 中，模型经常通过：

```text
Validation Accuracy
Validation Loss
```

评价效果。

但是 Robot Policy 的真实目标是：

```text
Closed-Loop Task Success
```

因为一个单步预测看起来很小的误差，可能改变未来 state。

因此：

```text
Low Offline MSE
```

不等于：

```text
High Rollout Success Rate
```

一个机器人 policy 可能在 expert distribution 上预测非常准确，但只要 rollout 后稍微离开 expert trajectory，就可能迅速失败。

所以评价 Robot Learning 模型时必须区分：

```text
Offline Prediction Quality
```

和：

```text
Closed-Loop Control Performance
```

---

# 46. 综合思考

## 问题 1

为什么说 Behavioral Cloning 在形式上类似 supervised learning，但在机器人系统中又比普通监督学习困难？

### 解析

训练阶段确实是：

```text
Input Observation
+
Expert Action Label
```

但机器人部署是 closed-loop。

当前 prediction 会改变未来 state，因此错误会反馈进未来输入。

这使机器人 policy 面临 distribution shift 和 compounding error。

---

## 问题 2

如果 validation action MSE 极低，但真机 success rate 很低，应该优先检查什么？

首先不应该简单得出：

> 模型容量不够。

更应该检查：

```text
Train / Deployment Distribution Shift
Observation Mismatch
Camera Pose Mismatch
Action Normalization
Action Interface
State Distribution Coverage
Recovery Data
Closed-Loop Error Accumulation
```

其中本章直接建立的理论核心是：

> Expert distribution 与 policy rollout distribution 不一致。

---

## 问题 3

如果同一 observation 下存在多个合理动作，为什么简单 deterministic regression 可能存在问题？

因为它可能学习条件平均值。

如果多个 mode 在 action space 中相距很远，平均值甚至可能不属于任何合理行为模式。

---

## 问题 4

为什么不能说“rollout 越多，BC policy 自然会越来越好”？

因为 vanilla BC 的 rollout 没有自动产生监督信号。

只有当 rollout data 被：

```text
Expert Relabel
Reward
Human Feedback
Online Objective
```

重新纳入训练时，policy 才会因为这些新数据发生改变。

---

## 问题 5

为什么优秀专家的数据仍然可能不足以训练鲁棒 policy？

因为 expert 可能几乎不进入错误状态。

因此 dataset 主要覆盖：

```text
Successful States
```

而缺少：

```text
Off-Trajectory States
Recovery States
```

learned policy 却恰恰可能频繁访问这些状态。

---

# 47. 本章核心逻辑总结

Behavioral Cloning 可以用一条完整链条理解：

```text
Expert
↓
Demonstration
↓
Trajectory
↓
Observation-Action Pairs
↓
Dataset
↓
DataLoader
↓
Policy
↓
Predicted Action
↓
Compare with Expert Action
↓
Loss
↓
Backpropagation
↓
Gradient
↓
Optimizer
↓
Updated Policy
↓
Rollout
```

其中最核心的数学目标是：

$$
\theta^*
=
\arg\min_\theta
\mathbb E_{(o,a^*)\sim D}
\left[
-
\log
\pi_\theta(a^*|o)
\right]
$$

这个公式可以读成：

> 找到一组 policy parameters，使专家 demonstration 中真正采取的 action 在模型条件动作分布中拥有尽可能高的 probability。

当 policy 是固定 variance Gaussian 时，这一目标自然退化为 continuous action 中常见的 MSE。

当 action 是离散类别或 token 时，可以使用 Cross Entropy。

当动作分布是 multimodal 时，简单 regression 可能产生 mode averaging，这进一步推动了更加复杂的 generative policy。

而真正把 Robot Learning 与普通 supervised learning 区分开的，是：

> Policy output 会改变未来 observation。

因此：

```text
单步误差
↓
State 偏移
↓
Observation Distribution Shift
↓
更大的 Policy Error
↓
Compounding Error
```

这构成 Behavioral Cloning 最核心的局限之一。

---

# 48. 最终应形成的心智模型

完成本章后，不应该只记住：

> BC 就是用 MSE 模仿专家。

更加完整的心智模型应该是：

> Behavioral Cloning 是一种从 expert demonstration 中学习 conditional action policy 的 supervised imitation-learning 方法。

它首先需要明确：

```text
机器人到底观察什么？
```

然后明确：

```text
专家到底输出什么 action？
```

再通过：

```text
Dataset
→
Policy
→
Loss
→
Gradient
→
Optimizer
```

学习：

$$
\pi_\theta(a|o)
$$

但是训练数据主要来自 expert distribution，而机器人实际执行时产生自己的 state distribution。

因此：

> Offline imitation learning 与 closed-loop deployment 之间天然存在 gap。

理解这个 gap，是进入 DAgger、online imitation learning、VLA post-training 和 reinforcement learning 的起点。

---

# 49. 关键词

**Demonstration — 专家示范 — 专家完成任务产生、用于机器人学习的行为数据。**

**Trajectory — 轨迹 — 系统随时间演化形成的 observation / state / action 序列。**

**State — 状态 — 描述系统真实情况所需的完整变量集合。**

**Observation — 观测 — 机器人实际能够从传感器等来源获得的信息。**

**Proprioception — 本体感知 — 机器人对自身关节、速度、末端和夹爪等状态的感知。**

**Action — 动作 — Policy 输出并交给机器人执行接口的控制目标或指令。**

**Policy — 策略 — 从 observation 到 action 或 action distribution 的映射。**

**Behavioral Cloning — 行为克隆 — 使用 expert demonstrations 监督训练机器人 policy。**

**Maximum Likelihood — 最大似然 — 提高 expert action 在 policy distribution 下概率的训练原则。**

**Negative Log Likelihood — 负对数似然 — Maximum Likelihood 常用的最小化 loss 形式。**

**MSE — 均方误差 — Continuous action regression 中常见损失，在固定 variance Gaussian 假设下可由 maximum likelihood 导出。**

**Cross Entropy — 交叉熵 — Discrete action 或 action-token prediction 常见训练目标。**

**Multimodal Distribution — 多模态分布 — 同一 observation 下存在多个明显不同但都合理的 action mode。**

**Mode Averaging — 模态平均 — Regression 将多个合理 action mode 平均成可能并不合理的中间动作。**

**SFT — 监督微调 — 使用有 target 的 supervised dataset 对 pretrained model 继续训练。**

**Dataset — 数据集 — 完整的训练样本集合。**

**DataLoader — 数据加载器 — 负责读取、shuffle 和组织 mini-batch 的训练组件。**

**Batch — 批次 — 一次 training step 同时处理的一组样本。**

**Epoch — 训练轮次 — 完整遍历一次 training dataset。**

**Backpropagation — 反向传播 — 根据 loss 计算模型参数 gradient。**

**Optimizer — 优化器 — 使用 gradient 实际更新 model parameters。**

**Rollout — 策略执行轨迹 — Policy 自己与环境交互产生完整 trajectory 的过程。**

**Expert Distribution — 专家状态分布 — Expert demonstration 中频繁出现的 observation / state distribution。**

**Policy Distribution — 策略访问分布 — Learned policy rollout 实际产生的 observation / state distribution。**

**Distribution Shift — 分布偏移 — Training distribution 与 deployment distribution 不一致。**

**Compounding Error — 累积误差 — 单步误差改变未来状态后沿闭环 trajectory 不断积累和放大的现象。**

---

# 50. 本章结束后的知识接口

本章建立的是：

```text
Expert Data
↓
Behavioral Cloning
↓
Learned Policy
↓
Closed-Loop Rollout
```

接下来最自然的问题已经出现：

> 如果 learned policy 在 rollout 中访问了 expert dataset 从未覆盖过的状态，应该怎么办？

这将把 Robot Learning 推向下一层问题：

```text
Expert State Distribution
≠
Policy State Distribution
↓
Distribution Shift
↓
Compounding Error
↓
需要重新收集 / 标注 Policy 实际访问的状态
```

由此自然进入：

**Distribution Shift、Compounding Error 与 DAgger。**
