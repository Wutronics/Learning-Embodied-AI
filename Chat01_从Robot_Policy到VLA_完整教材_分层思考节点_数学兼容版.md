# 01｜从 Robot Policy 到 VLA：VLA 到底是什么？

## 1. 本章导言

视觉—语言—动作模型（Vision-Language-Action Model, VLA）是现代机器人学习的重要研究范式。它看起来融合了视觉模型、语言模型、Transformer、机器人控制和学习策略等多个领域，但从系统层面看，它解决的仍然是一个非常经典的问题：

> 机器人在当前环境中看到了什么、接到了什么任务、自己处于什么状态，以及接下来应该执行什么动作？

因此，理解 VLA 的第一步并不是研究某个具体模型有多少参数，而是建立一个稳定的系统级抽象：

```text
环境 / 机器人
    ↓
Observation
    ↓
Robot Policy
    ↓
Action
    ↓
Controller / Actuator
    ↓
环境发生变化
    ↓
New Observation
    ↺
```

这个框架将贯穿后续所有内容。Transformer、VLM、Behavioral Cloning、Action Chunking、Diffusion Policy、Flow Matching、OpenVLA、π0、Fine-tuning、Agent + VLA 和 VLA + RL，最终都可以重新放回这个框架中理解。

本章围绕六个问题展开：

1. Robot Policy 到底是什么；
2. LLM、VLM 与 VLA 的关系是什么；
3. VLA 的输入通常包括什么；
4. VLA 到底输出什么动作；
5. 为什么真实机器人必须 closed-loop 执行；
6. 为什么 VLA 不是一种固定网络架构。

本章的目标不是记住某个模型结构，而是建立一个以后阅读所有 VLA 论文都能复用的“第一层坐标系”。

---

## 2. 从传统控制到 Robot Policy

### 2.1 传统控制器在做什么

在经典控制系统中，控制器根据当前状态和目标计算控制输入。最简单的状态反馈形式可以写成：

$$
u_t = Kx_t
$$

其中：

- x_t 表示时刻 t 的系统状态；
- K 表示控制增益；
- u_t 表示控制器输出。

如果还需要显式考虑参考目标 r_t，也可以写成：

$$
u_t = f(x_t,r_t)
$$

例如机械臂当前末端位置为 p_t，目标位置为 p_d，一个非常简单的速度控制律可以写成：

$$
v_t = K(p_d-p_t)
$$

这里的核心逻辑是：

```text
当前状态
+
目标
↓
控制律
↓
控制输入
```

也就是说，控制器本质上是在做“根据当前信息决定下一步怎么动”。

---

### 2.2 Robot Policy 的基本形式

在机器人学习（Robot Learning）中，这种“根据当前可获得信息决定动作”的函数通常称为策略（Policy）。

最简单的确定性策略可以写成：

$$
a_t=\pi_\theta(o_t)
$$

其中：

- π_θ 表示由参数 θ 决定的策略；
- o_t 表示时刻 t 的观测；
- a_t 表示时刻 t 的机器人动作。

如果策略由神经网络实现，那么 θ 就是网络中的可学习参数。

因此，可以先建立一个最基本的直觉：

> Robot Policy 就是机器人从当前可获得信息到动作之间的决策映射。

与传统控制相比，差别主要在于这个映射如何得到。

传统控制通常由人工根据系统模型、稳定性分析和设计目标构造；Robot Learning 中的 policy 则可能通过示教数据、监督学习、模仿学习或强化学习得到。

---

### 2.3 Policy 不等于底层 Controller

Policy 和 Controller 很容易被混为一谈。

在一些低层强化学习任务中，policy 可以直接输出 torque，此时它已经非常接近底层控制器。但在大量现代 VLA 系统中，policy 输出的是更高层的运动目标，例如：

- 末端执行器目标位姿；
- 笛卡尔空间增量；
- 关节目标位置；
- 一段动作序列。

这些输出通常仍需要经过传统机器人控制链才能真正驱动电机。

典型链路如下：

```text
VLA / Learned Policy
        ↓
EEF Target / Joint Target
        ↓
IK / Operational-Space Controller
        ↓
Joint Command
        ↓
Joint Controller
        ↓
Motor / Actuator
```

因此，VLA 更适合被理解为机器人系统中的 learned policy，而不是“一个大模型直接替代所有控制器”。

---

## 3. 阶段思考 A：为什么说 VLA 仍然是 Robot Policy？

### 思考题 1

为什么说 VLA 本质上仍然是 Robot Policy？

### 解析

判断一个系统是不是 policy，关键不在于内部用了什么网络，而在于它最终是否实现了：

```text
当前信息
↓
动作决策
```

VLA 虽然内部可能包含 Vision Encoder、LLM、Transformer、Action Head，甚至 Diffusion 或 Flow Matching，但它的最终功能仍然是根据 observation 和任务条件产生 robot action。

因此，VLA 的核心抽象仍然可以写成：

$$
a_t \sim \pi_\theta(a_t\mid o_t,l)
$$

其中 l 是语言任务条件。

如果把 observation 进一步拆开，常见形式是：

$$
a_t \sim \pi_\theta(a_t\mid I_t,s_t,l)
$$

所以 VLA 并不是脱离 Robot Learning 的另一种系统，而是多模态基础模型进入 Robot Policy 之后形成的一类策略。

---

### 思考题 2

下面哪一种说法更准确？

A. VLA 是一种类似 ResNet 的固定神经网络架构。  
B. VLA 是一类把视觉、语言和机器人状态映射到机器人动作的 Policy 范式。

### 解析

B 更准确。

“VLA”描述的是功能范式，而不是唯一网络结构。不同 VLA 可以采用完全不同的：

- Vision Encoder；
- Language Backbone；
- 多模态融合方式；
- Action Representation；
- Training Objective；
- Action Generation Method。

因此，看到一个模型被称为 VLA，不能据此推断它内部一定长什么样。

---

## 4. 为什么 Policy 常写成概率分布

最简单的 policy 可以写成：

$$
a_t=\pi_\theta(o_t)
$$

这表示一个确定性映射：同一个输入对应一个动作。

但机器人学习中更一般的形式是：

$$
\pi_\theta(a_t\mid o_t)
$$

它表示：给定 observation 后，policy 描述可能动作的条件分布。

这样建模的原因是，机器人任务通常并不存在唯一正确动作。

例如机械臂抓取一个杯子时，可以：

- 从左侧接近；
- 从右侧接近；
- 从顶部抓取；
- 从侧面抓取。

只要最终任务成功，这些动作都可能是合理的。

因此，同一个 observation 对应的合理动作可能不止一个。动作分布可能具有 multimodal 特征。

本章只需要建立以下心智模型：

```text
同一个 Observation
↓
可能存在多个合理 Action
↓
Policy 可以描述动作分布，而不只是唯一答案
```

这一点会在后续 Diffusion 与 Flow Matching 中进一步展开，因为生成式策略正是为复杂动作分布提供了新的建模方式。

---

## 5. 从 LLM 到 VLM，再到 VLA

### 5.1 LLM：语言到语言

大语言模型（Large Language Model, LLM）主要处理语言序列。

最典型的形式是：

```text
Language
↓
LLM
↓
Text
```

例如输入：

> “解释什么是机械臂末端执行器。”

模型可以输出文字解释。

如果没有额外视觉输入，LLM 本身并不能直接知道机器人摄像头中当前出现了什么物体。

---

### 5.2 VLM：视觉与语言联合理解

视觉—语言模型（Vision-Language Model, VLM）能够联合处理图像和语言。

它可以抽象成：

$$
y=f_\theta(I,l)
$$

其中：

- I 表示图像；
- l 表示语言输入；
- y 表示文本结果或某种语义表示。

例如给模型一张桌面图片，再询问：

> “红色杯子在哪里？”

VLM 可以根据图像内容回答杯子的位置关系。

因此，VLM 的核心可以概括为：

```text
Vision
+
Language
↓
Semantic Understanding / Representation
```

它主要解决“看懂”和“理解”的问题。

---

### 5.3 VLA：从理解走向动作

VLA 不只需要理解环境，还必须生成机器人能够执行的动作。

常见抽象为：

$$
a_t \sim \pi_\theta(a_t\mid I_t,l,s_t)
$$

其中：

- I_t：当前视觉输入；
- l：语言指令；
- s_t：机器人本体状态；
- a_t：机器人动作；
- π_θ：机器人策略。

例如：

```text
Camera Image
+
"Pick up the red cup"
+
Robot State
        ↓
       VLA
        ↓
Cartesian Delta / Joint Target / Action Chunk
```

因此，VLM 与 VLA 的核心区别不在于是否使用 Transformer，也不在于模型是否足够大，而在于最终输出处于哪个语义层级。

VLM 的核心是：

```text
Vision + Language
↓
Understanding
```

VLA 的核心是：

```text
Vision + Language + Robot Context
↓
Robot Action
```

---

## 6. 阶段思考 B：VLM 与 VLA 的边界在哪里？

### 思考题 3

VLM 和 VLA 最本质的区别是什么？

### 解析

最关键的区别是输出。

如果模型看到桌面图像后输出：

> “红色杯子位于蓝色瓶子左侧。”

这仍然是视觉语言理解。

如果模型根据相同视觉和语言任务进一步输出：

```text
Δx = +0.02
Δy = -0.01
Δz = -0.03
gripper = open
```

那么输出已经进入机器人动作空间。

因此判断一个模型是不是 VLA，不能只看它有没有视觉和语言模块，而要看它是否真正承担了 robot policy 的角色。

---

### 思考题 4

为什么不能简单地把“有视觉输入的大语言模型”都叫作 VLA？

### 解析

因为视觉输入只说明模型具有多模态感知能力，并不能说明它已经具有机器人动作生成能力。

一个 VLM 可以非常擅长识别物体、解释场景和回答问题，但如果它没有定义机器人 action space，也没有输出可被控制链执行的机器人动作，它仍然不是完整意义上的 VLA policy。

---

## 7. State、Observation 与 Proprioception

### 7.1 Environment State：真实世界的完整状态

在控制理论和强化学习中，常用 state 表示系统真实状态。

假设用 x_t 表示时刻 t 的完整环境状态，它可能包括：

- 物体精确 6D 位姿；
- 物体速度；
- 质量；
- 摩擦系数；
- 接触状态；
- 机器人关节状态；
- 环境中其他动态变量。

理论上，如果拥有完整 state，就能够更完整地描述系统如何演化。

但真实机器人通常无法直接获得所有这些信息。

---

### 7.2 Observation：机器人真正能测到什么

机器人只能通过传感器获得有限信息，因此通常使用 observation。

可以抽象为：

$$
o_t=h(x_t)
$$

其中：

- x_t 是真实环境状态；
- h 表示传感器和观测过程；
- o_t 是机器人实际得到的观测。

例如，机器人可以获得：

- RGB 图像；
- Depth；
- Joint Position；
- Joint Velocity；
- Force / Torque Sensor；
- Gripper State。

但机器人通常无法直接知道：

- 杯子的真实摩擦系数；
- 桌面的精确材料参数；
- 所有物体的真实动力学参数。

因此，机器人决策通常基于 observation，而不是假设拥有完美 world state。

---

### 7.3 Robot State 与 Proprioception

机器人自己的内部状态通常称为 proprioception。

对机械臂而言，可以写成：

$$
s_t=[q_t,\dot q_t,g_t]
$$

其中：

- q_t：关节位置或关节角；
- q̇_t：关节速度；
- g_t：夹爪状态。

对于一个 7DoF 机械臂：

$$
q_t\in\mathbb{R}^7
$$

例如关节角可能是：

$$
q_t=
[0.12,-0.52,0.31,-1.48,0.05,1.02,0.73]
$$

Proprioception 的重要性在于：即使视觉画面完全相同，如果机械臂当前构型不同，下一步合理动作也可能不同。

因此一个更清晰的 VLA 写法是：

$$
o_t=(I_t,s_t)
$$

再将语言 l 作为任务条件：

$$
\pi_\theta(a_t\mid o_t,l)
$$

等价地：

$$
\pi_\theta(a_t\mid I_t,s_t,l)
$$

---

## 8. 阶段思考 C：机器人到底“知道”什么？

### 思考题 5

为什么真实机器人通常写 observation，而不是假设能够直接获得完整 state？

### 解析

因为完整 state 包含大量无法直接测量的变量。

机器人只能通过传感器获得世界的一部分投影。例如摄像头给出像素，编码器给出关节角，力传感器给出接触信息。这些都属于 observation。

所以：

```text
真实世界状态
↓
传感器
↓
Observation
↓
Policy
```

VLA 处理的通常不是“世界真相本身”，而是机器人能够感知到的有限信息。

---

### 思考题 6

什么是 proprioception？对于 7DoF 机械臂，它可能包含什么？

### 解析

Proprioception 可以理解为“机器人对自己身体状态的感知”。

典型信息包括：

- 7 个关节角；
- 7 个关节速度；
- 夹爪开合状态；
- 有时还包括末端位姿；
- 有时还包括力矩或力传感器读数。

它和视觉信息不同。视觉主要描述外部世界，proprioception 主要描述机器人自身。

---

## 9. VLA 的 Action Space：模型到底预测什么

VLA 并没有统一的动作输出格式。

这也是以后阅读论文时最需要优先确认的问题之一：

> 这个 VLA 的 action 到底是什么？

常见表示包括：

- Joint Position；
- Joint Velocity；
- Joint Torque；
- End-Effector Pose；
- Cartesian Delta；
- Cartesian Velocity；
- Gripper State；
- Action Token；
- Continuous Action Chunk。

---

### 9.1 Joint Position

如果以关节位置作为动作：

$$
a_t=q_t^{\text{target}}
$$

对于 7DoF 机械臂，输出可以是：

$$
a_t=[q_1,q_2,\ldots,q_7]
$$

含义是给出七个关节的目标位置。

---

### 9.2 Joint Velocity

如果 action 是关节速度：

$$
a_t=\dot q_t
$$

例如：

$$
a_t=[0.10,-0.05,0.02,0,0.10,-0.03,0]
$$

这表示当前控制周期内各个关节应该如何运动。

---

### 9.3 Joint Torque

如果 action 是关节力矩：

$$
a_t=\tau_t
$$

那么策略已经非常接近底层动力学控制。

这种动作表示通常对控制频率、机器人动力学和稳定性更敏感，因此不能把“VLA 输出 action”自动理解成“VLA 直接输出 motor PWM”。

---

### 9.4 End-Effector Pose

机械臂也可以在任务空间中输出末端执行器目标。

例如：

$$
a_t=
(p_t^{\text{target}},R_t^{\text{target}})
$$

其中：

- p_target 表示目标末端位置；
- R_target 表示目标末端姿态。

随后系统可以通过 IK 或 Operational-Space Controller 得到关节命令。

```text
EEF Target
    ↓
IK / Operational-Space Controller
    ↓
Joint Target
    ↓
Joint Controller
    ↓
Robot
```

---

### 9.5 Cartesian Delta

现代 Robot Learning 中经常使用相对动作，而不是绝对目标位置。

例如：

$$
a_t=
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

一个具体输出可能是：

$$
a_t=
[
0.02,
0,
-0.01,
0,
0,
0.03,
1
]
$$

它可以表示：

- x 方向移动 2 cm；
- y 方向不变；
- z 方向下降 1 cm；
- 某个旋转维度变化 0.03 rad；
- 夹爪执行相应动作。

这种表示的关键思想是：

```text
不是“去绝对坐标哪里”
而是
“相对当前位置怎么变化”
```

---

### 9.6 Gripper Action

对于抓取任务，夹爪通常也是 action 的一部分。

例如：

```text
0 → open
1 → close
```

也可能使用连续值表示夹爪宽度或夹爪速度。

因此，对一个 6DoF 末端位姿增量加 1 个夹爪维度，常见 action dimension 可以是 7。

---

### 9.7 Action Token

连续动作也可以先离散化，再编码成 token。

例如某个动作维度：

$$
\Delta x\in[-0.05,0.05]
$$

如果将其划分成若干个区间，每个区间对应一个离散 token，就可以把动作预测改写成类似语言模型的 token prediction。

基本流程是：

```text
Continuous Action
      ↓
Discretization
      ↓
Action Token
      ↓
Autoregressive Prediction
```

这种思路使语言模型式的生成框架可以直接处理机器人动作。

但它也会引入新的问题，例如量化精度和动作离散化误差。本章只建立概念，不提前展开。

---

## 10. 阶段思考 D：Action 到底处于哪一层？

### 思考题 7

为什么 VLA 通常不直接输出电机 PWM？请给出一条合理的控制链。

### 解析

VLA 往往工作在比电机驱动更高的决策层。

例如：

```text
VLA
↓
Cartesian Delta
↓
Target EEF Pose
↓
IK
↓
Joint Target
↓
Joint Position Controller
↓
Motor Driver
↓
Motor
```

这样做可以让 VLA 主要负责“做什么动作”，而把高频稳定控制交给传统控制器。

这与现代机器人系统的分层思想一致。

---

### 思考题 8

如果一篇论文只写“model outputs actions”，为什么不能立刻判断它与另一篇 VLA 方法是否可直接比较？

### 解析

因为 action 可能位于完全不同的控制层级。

一个方法可能输出：

- Joint Torque；

另一个方法可能输出：

- EEF Delta；

还有的方法可能输出：

- Action Token；
- Action Chunk。

这些 action 的物理含义、频率、控制难度和跨机器人兼容性完全不同。

因此阅读论文时必须继续追问：

```text
Action 是什么？
↓
在哪个坐标系？
↓
绝对还是相对？
↓
单步还是序列？
↓
之后还经过哪些 Controller？
```

---

## 11. Single Action 与 Action Chunk

最简单的策略一次只输出一个动作：

$$
o_t\rightarrow a_t
$$

现代机器人策略也经常一次输出未来多个时间步：

$$
o_t
\rightarrow
[a_t,a_{t+1},\ldots,a_{t+H-1}]
$$

其中 H 表示动作预测长度。

这段动作序列称为 Action Chunk。

例如抓取杯子的一个 chunk 可能包含：

```text
向前接近
↓
向下移动
↓
继续对准
↓
夹爪闭合
↓
向上抬起
```

一次预测多个动作有两个直观优势：

第一，连续机器人动作存在强时间相关性，一次生成一段动作可以更自然地建模局部运动结构。

第二，大型 VLA 推理通常比较昂贵，如果每个底层控制周期都完整运行一次大模型，计算成本可能过高。

但 Action Chunk 也带来一个关键风险：

> 当环境在执行过程中发生变化时，较早预测的后续动作可能已经过时。

因此，Action Chunk 并不意味着整个任务开环执行。实际系统通常会执行部分动作后重新观察，再重新规划。

---

## 12. Closed-loop：为什么真实机器人不能只做一次前向推理

### 12.1 一次性预测的局限

假设机器人在 t = 0 时刻观察到杯子位置，然后一次性生成整个任务的动作：

$$
a_{0:T}=\pi(I_0,l)
$$

随后完全不再获取新观测。

这接近 open-loop execution。

真实机器人中却存在很多不确定因素：

- 摄像头估计误差；
- 机械臂跟踪误差；
- 物体滑动；
- 摩擦变化；
- 接触不确定性；
- 环境中其他物体移动；
- 人类干预。

因此，即使最初规划完全正确，真实执行轨迹也可能逐渐偏离预期。

---

### 12.2 Closed-loop 的数学表达

设真实环境状态为 x_t，系统动力学可以抽象为：

$$
x_{t+1}=f(x_t,a_t,w_t)
$$

其中：

- x_t：当前真实状态；
- a_t：机器人动作；
- w_t：扰动与未建模因素；
- f：机器人与环境的动力学演化。

执行动作后，传感器得到新的 observation：

$$
o_{t+1}=h(x_{t+1})
$$

policy 再基于新的 observation 输出动作：

$$
a_{t+1}
\sim
\pi_\theta(a\mid o_{t+1},l)
$$

于是形成：

```text
Observation_t
     ↓
   Policy
     ↓
 Action_t
     ↓
Robot / Environment
     ↓
Observation_{t+1}
     ↓
   Policy
     ↓
 Action_{t+1}
```

这就是 closed-loop execution。

---

### 12.3 与反馈控制的联系

从控制背景看，这个思想并不陌生。

传统反馈控制的核心是：

```text
测量
↓
计算误差
↓
控制
↓
系统变化
↓
重新测量
```

VLA 只是把反馈信息从低维状态扩展到了高维多模态 observation，例如：

- Image；
- Language Condition；
- Joint State；
- Gripper State；
- History。

因此，可以把 VLA 理解为一种高维、多模态、学习得到的闭环决策策略。

但要注意：

> VLA 的闭环频率不一定等于底层控制器的闭环频率。

底层 Joint Controller 可以以高频运行，而 VLA 可能只以较低频率重新推理。这种多频率闭环结构会在后续实时推理与 latency 中进一步讨论。

---

## 13. 阶段思考 E：Action Chunk 会破坏 Closed-loop 吗？

### 思考题 9

如果 VLA 一次输出：

$$
[a_t,a_{t+1},\ldots,a_{t+15}]
$$

这叫什么？

为什么即使这样，机器人仍然需要 closed-loop？

### 解析

这叫 Action Chunk。

它只表示“策略一次预测多个未来动作”，并不代表机器人需要把整个 chunk 无条件执行到底。

更合理的方式是：

```text
Observe
↓
Predict Chunk
↓
Execute Part of Chunk
↓
Observe Again
↓
Replan
```

因此，Action Chunk 与 Closed-loop 并不矛盾。

真正需要研究的是：

- Chunk Horizon 多长；
- 每次执行多少步；
- 多久重新规划；
- 如果新 observation 与旧 chunk 冲突，如何处理。

这些问题会自然连接到 Receding Horizon 和 Real-Time VLA。

---

## 14. Pick-and-Place：完整机器人实例

考虑任务：

> Pick up the red cup and place it in the left tray.

---

### 14.1 初始观测

摄像头得到图像 I_0。

机器人读取 proprioception s_0，例如：

- 7DoF joint position；
- joint velocity；
- gripper open；
- 当前 EEF pose。

语言指令为 l：

```text
"Pick up the red cup and place it in the left tray."
```

策略输入：

$$
(I_0,s_0,l)
$$

---

### 14.2 第一次动作生成

VLA 根据当前观测输出：

$$
a_0=
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

例如当前动作表示为：

```text
向右前方接近杯子
+
保持夹爪打开
```

VLA 输出之后进入控制链：

```text
VLA
↓
EEF Delta
↓
Target EEF Pose
↓
IK / Controller
↓
Joint Command
↓
Robot
```

---

### 14.3 执行后的重新观测

由于机械误差或视觉误差，机器人实际位置可能与预计略有差异。

于是系统重新获取：

$$
I_1
$$

以及：

$$
s_1
$$

VLA 根据最新 observation 再次计算：

$$
a_1
\sim
\pi_\theta(a\mid I_1,s_1,l)
$$

如果视觉发现夹爪偏左，就修正横向位置。

如果已经位于杯子上方，就执行下降。

---

### 14.4 抓取

当夹爪对准杯子后，VLA 输出：

```text
gripper = close
```

机器人执行夹爪闭合。

随后仍然需要重新观察，而不是默认“闭合就一定抓成功”。

视觉或状态信息可能表明：

- 杯子已经抓起；
- 杯子滑落；
- 夹爪位置不正确；
- 杯子发生旋转。

这说明 closed-loop 不只是运动纠偏，也承担任务执行反馈。

---

### 14.5 搬运与放置

抓起杯子后，机器人继续：

```text
Observation
↓
VLA
↓
向左托盘移动
↓
Observation
↓
VLA
↓
调整位置
↓
Observation
↓
释放夹爪
↓
Observation
↓
确认结果
```

因此，一个完整 VLA Manipulation 任务应理解为连续的 policy rollout，而不是一次静态图像理解。

---

## 15. VLA 在机器人系统中的层级位置

为了避免把 VLA 误解成“万能机器人控制器”，可以把真实系统拆成几个层级。

### 15.1 High-level Task Planning

负责回答：

> 接下来应该完成哪个子任务？

例如：

```text
找到杯子
↓
接近杯子
↓
抓取
↓
搬运
↓
放置
```

未来的 Agent + VLA 系统中，这一层可能由 LLM / VLM Planner 或 Agent 完成。

---

### 15.2 Learned Policy / VLA

VLA 主要负责：

> 当前 observation 下，应该产生什么机器人 action？

例如输出：

- EEF Delta；
- Joint Target；
- Action Token；
- Action Chunk。

---

### 15.3 Action / Trajectory Generation

有些 VLA 本身直接产生连续动作。

有些系统则只产生目标，再由 trajectory planner 规划路径。

因此，“VLA 与 trajectory planner 的边界”并不固定。

---

### 15.4 Controller

Controller 负责把目标变成稳定可执行的控制命令，例如：

- Joint Position Control；
- Velocity Control；
- Impedance Control；
- Operational-Space Control。

---

### 15.5 Actuator

执行器是最终产生物理运动的：

- Motor；
- Driver；
- Mechanical Transmission；
- Gripper Mechanism。

完整链路可以概括为：

```text
Language / Task
      ↓
Perception
      ↓
VLA / Learned Policy
      ↓
Action / Target / Chunk
      ↓
IK / Trajectory / Controller
      ↓
Joint Command
      ↓
Actuator
      ↓
Physical World
```

---

## 16. 为什么 VLA 不是固定网络架构

VLA 是按功能定义的，不是按网络结构定义的。

只要一个模型主要完成：

```text
Vision
+
Language
+
Robot Context
↓
Robot Action
```

就可以属于 VLA 范畴。

一种 VLA 可能是：

```text
Image
 ↓
Vision Encoder
       \
Language → Multimodal Transformer → Action Token
       /
Robot State
```

另一种可能是：

```text
Image + Language
       ↓
Vision-Language Backbone
       ↓
Action Expert
       ↓
Continuous Action Generator
       ↓
Action Chunk
```

还有的系统可能：

- 直接回归连续动作；
- 把动作离散成 token；
- 使用 Diffusion；
- 使用 Flow Matching；
- 冻结 backbone；
- 只训练 action head；
- 端到端全部 fine-tune。

因此，阅读 VLA 论文时不应首先问：

> “它是不是 Transformer？”

而应该问：

```text
输入是什么？
↓
视觉如何编码？
↓
语言如何进入？
↓
Robot State 如何进入？
↓
模态如何融合？
↓
Action Space 是什么？
↓
如何训练？
↓
如何推理？
↓
如何闭环执行？
```

---

## 17. 容易混淆的概念

### 17.1 VLM 与 VLA

VLM 主要完成视觉语言理解。

VLA 必须进一步产生机器人动作。

判断依据不是“有没有视觉和语言”，而是“最终输出是否进入 robot action space”。

---

### 17.2 Policy 与 Controller

Policy 更强调：

> 应该做什么。

Controller 更强调：

> 怎样稳定、准确地把目标执行出来。

如果 policy 直接输出 torque，两者会靠得很近；如果 VLA 输出 EEF Delta，两者分层就非常明显。

---

### 17.3 Environment State 与 Robot State

Environment State 描述整个环境真实状态。

Robot State 只描述机器人自身内部状态。

例如：

```text
杯子 6D Pose → Environment State
机器人 Joint Angle → Robot State / Proprioception
```

---

### 17.4 State 与 Observation

State 是系统真实状态。

Observation 是机器人实际能够测到的信息。

真实机器人通常只能根据 observation 决策。

---

### 17.5 Action 与 Motor Command

VLA action 不等于 motor PWM。

Action 可以位于更高层，例如：

- EEF Pose；
- Cartesian Delta；
- Joint Target；
- Action Token；
- Action Chunk。

---

### 17.6 Action Chunk 与 Open-loop

Action Chunk 表示一次预测多个未来动作。

Open-loop 表示执行过程中不根据新反馈修正。

两者不是同一个概念。

Action Chunk 完全可以在 closed-loop 系统中使用。

---

## 18. 本章内容在整个 VLA 知识链中的位置

本章建立的是 VLA 的最外层系统框架：

```text
Robot Policy
↓
Multimodal Observation
↓
Vision + Language + Robot State
↓
VLA
↓
Action Representation
↓
Robot Control Interface
↓
Closed-loop Execution
```

后续章节会逐步填充这张图。

### Transformer

回答：

> 图像 token、语言 token、机器人状态之间如何建立复杂依赖关系？

### VLM

回答：

> 图像和语言如何形成统一语义表示？

### Behavioral Cloning

回答：

> Policy 如何从 expert demonstration 中学习？

### Action Representation

回答：

> VLA 究竟应该预测 Joint、EEF、Delta 还是其他动作？

### Action Chunking

回答：

> 为什么一次预测多个动作？如何重新规划？

### Diffusion / Flow Matching

回答：

> 如何生成复杂、多模态的连续动作分布？

### VLA Fine-tuning

回答：

> 已有基础 VLA 如何适配新 robot 或新 task？

### Agent + VLA

回答：

> 更高层 Agent 应该什么时候调用 VLA Skill？

### VLA + RL

回答：

> 如何利用环境 reward 和真实交互继续提升 policy？

因此，本章不是某个具体 VLA 模型的介绍，而是在建立以后所有章节都要复用的坐标系。

---

## 19. 局限与研究视角

理解“VLA 是多模态 Robot Policy”只是第一步。

真实研究中还存在很多关键问题。

### 19.1 Action Representation 不统一

不同 robot 可能使用：

- Joint Space；
- Cartesian Space；
- Absolute Action；
- Relative Action；
- Velocity；
- Torque。

这会直接影响 cross-embodiment learning。

---

### 19.2 Offline Prediction 准确不等于 Closed-loop 成功

模型在 dataset 上预测 expert action 很准，并不能自动保证真实 rollout 成功。

因为 policy 自己的动作会改变后续 observation。

真实机器人执行会形成：

```text
Policy
↓
Action
↓
New State
↓
New Observation
↓
Policy
```

这与普通 i.i.d. supervised learning 不同。

后续 BC 与 DAgger 会专门处理这个问题。

---

### 19.3 大 VLA 与实时控制之间存在频率矛盾

大型模型推理可能很慢，而机器人控制往往需要高频运行。

因此需要区分：

- Camera Frequency；
- VLA Inference Frequency；
- Robot Control Frequency。

这会自然连接到 latency、stale observation、asynchronous inference 等问题。

---

### 19.4 视觉语言能力不等于机器人控制能力

模型“知道杯子是什么”不等于“能够稳定抓住杯子”。

机器人执行还涉及：

- Geometry；
- Kinematics；
- Dynamics；
- Contact；
- Timing；
- Control Error；
- Distribution Shift。

因此，VLM 能力只是 VLA 能力的一部分。

---

## 20. 综合思考

### 综合题 1

给出并解释下面这个 VLA policy：

$$
\pi_\theta(a_t\mid I_t,l,s_t)
$$

### 解析

π_θ 表示参数化机器人策略。

a_t 表示当前要执行的机器人动作。

I_t 表示当前视觉 observation。

l 表示语言任务条件。

s_t 表示机器人自身状态。

整个表达式表示：

> 在当前视觉、语言任务和机器人本体状态条件下，策略对机器人动作进行预测或建模。

---

### 综合题 2

为什么真实机器人不能只做一次 VLA forward，然后执行完整任务？

### 解析

因为机器人与环境之间存在扰动和不确定性。

一次预测只基于旧 observation。

动作执行后，真实环境已经改变，因此后续动作必须基于新 observation 重新计算。

完整闭环是：

```text
Observe
↓
Predict
↓
Execute
↓
Observe Again
↓
Correct / Replan
```

---

### 综合题 3

如果一个 VLA 输出 EEF Delta，它与机器人真正转动电机之间通常还缺哪些环节？

### 解析

典型链路：

```text
EEF Delta
↓
Target EEF Pose
↓
IK / Operational-Space Controller
↓
Joint Target
↓
Joint Controller
↓
Motor Driver
↓
Motor
```

这说明 policy 与 actuator 之间通常存在明确的机器人控制栈。

---

### 综合题 4

拿到一篇新的 VLA 论文时，第一轮阅读最应该问哪些问题？

### 解析

建议优先检查：

1. Observation 是什么；
2. Language 如何编码；
3. Visual Encoder 是什么；
4. Robot State 是否输入；
5. Action Space 是什么；
6. Action 是 Joint、EEF、Velocity 还是 Delta；
7. Single Action 还是 Action Chunk；
8. Action 是 token、regression、Diffusion 还是 Flow；
9. 输出之后如何进入 Controller；
10. Closed-loop 如何实现；
11. 多久重新推理；
12. 论文真正测试的是哪一种 generalization。

这套问题比单纯关注参数量更适合作为研究者阅读 VLA 论文的第一步。

---

## 21. 本章总结

本章的核心不是记住很多术语，而是建立一个统一心智模型。

第一，Robot Policy 是从当前机器人可获得信息到动作的决策映射：

$$
a_t=\pi_\theta(o_t)
$$

更一般地，它可以表示动作条件分布：

$$
\pi_\theta(a_t\mid o_t)
$$

第二，VLA 本质上仍然是 Robot Policy，只是输入扩展到了视觉、语言和机器人本体状态：

$$
a_t
\sim
\pi_\theta(a_t\mid I_t,l,s_t)
$$

第三，VLM 与 VLA 的根本区别在于输出是否进入机器人动作空间。

第四，VLA 的 action 没有统一形式。它可以是 Joint Position、EEF Pose、Cartesian Delta、Action Token 或 Action Chunk。

第五，VLA 通常不直接替代整个机器人控制系统。它的输出仍然需要经过 IK、Controller 和 Actuator 等模块。

第六，真实机器人必须 closed-loop 执行。动作改变环境以后，系统必须重新观察，再根据新的 observation 修正下一步行为。

最终应形成如下核心框架：

```text
Vision
Language
Robot State
    ↓
   VLA
(Robot Policy)
    ↓
Robot Action
    ↓
Controller
    ↓
Physical Robot
    ↓
New Observation
    ↺
```

后续所有关于 Transformer、VLM、BC、Action Representation、Action Chunking、Diffusion、Flow Matching、Fine-tuning、Agent 和 RL 的内容，都可以放回这张图中定位。

---

## 22. 关键词

**Robot Policy — 机器人策略 — 根据当前 observation 和任务条件产生 robot action 的决策映射或条件分布。**

**Observation — 观测 — 机器人通过摄像头、编码器、力传感器等实际获得的信息。**

**State — 状态 — 描述系统真实情况的变量集合，真实机器人通常无法完整获得。**

**Proprioception — 本体感觉 — 机器人对自身关节、速度、夹爪等内部状态的感知。**

**Vision-Language Model (VLM) — 视觉语言模型 — 联合处理视觉和语言并形成语义理解或表示的模型。**

**Vision-Language-Action Model (VLA) — 视觉语言动作模型 — 根据视觉、语言和机器人状态产生机器人动作的多模态策略。**

**Action Space — 动作空间 — Policy 所允许输出的机器人动作表示及其取值范围。**

**Joint Position — 关节位置动作 — 直接给出各关节的目标位置。**

**Joint Velocity — 关节速度动作 — 给出各关节目标速度。**

**Joint Torque — 关节力矩动作 — 直接给出关节力矩，控制层级更低。**

**End-Effector Pose — 末端执行器位姿 — 以机械臂末端的位置和姿态作为动作目标。**

**Cartesian Delta — 笛卡尔增量动作 — 预测末端相对当前位置的平移与旋转变化。**

**Gripper State — 夹爪状态 — 表示夹爪打开、闭合或连续开合程度。**

**Action Token — 动作 Token — 将机器人动作离散化或编码为类似语言 token 的符号。**

**Action Chunk — 动作块 — 一次预测连续多个未来时间步的动作序列。**

**Closed-loop — 闭环 — 动作执行后重新获取 observation，并基于新反馈继续修正策略输出。**

**Open-loop — 开环 — 根据初始信息生成动作后，在执行过程中不根据新反馈修正。**

**Inverse Kinematics (IK) — 逆运动学 — 根据目标末端位姿求解对应关节配置。**

**Controller — 控制器 — 将 policy 或 planner 给出的目标转化为稳定可执行的底层控制命令。**

**Actuator — 执行器 — 最终产生真实物理运动的电机、驱动器及相关机械机构。**
