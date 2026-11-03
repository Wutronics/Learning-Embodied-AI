# Chat 19｜VLA Fine-tuning、Representation Adaptation 与 SpatialVLA

> 《VLA 与现代机器人学习基础》

---

## 1. 本章导言

在完成 Robot Policy、Transformer、VLM、Behavioral Cloning、Action Representation、Action Chunking、Diffusion Policy、Flow Matching、RT-1、RT-2、OpenVLA、π0 与 π0.5 等基础之后，VLA 学习开始从“理解一个预训练模型是什么”进入“如何把预训练模型真正迁移到自己的机器人和任务上”。

这一阶段的核心问题是：

> 一个已经在大规模机器人数据上预训练的 VLA，为什么换到新的机器人、新的相机、新的控制接口或新的任务之后，仍然经常不能直接工作？

最直观的答案是：可以继续训练模型，即 Fine-tuning。

但仅仅停留在“拿新数据做 LoRA”这一层，会把 VLA adaptation 理解得过于狭窄。真实机器人迁移中，失败原因可能来自多个层次：

```text
数据采集 / 部署链路
↓
Observation Distribution
↓
Action / State Representation
↓
Robot Embodiment
↓
Parameter Adaptation
↓
Closed-loop Evaluation
```

因此，本章不把 Fine-tuning 简化成某一种训练技巧，而是建立更完整的研究框架：

```text
Distribution Shift Diagnosis
↓
Representation Alignment
↓
Parameter Adaptation
↓
Knowledge Retention
↓
Data Efficiency
↓
Closed-loop Evaluation
```

SpatialVLA 是本章的重要贯穿案例。它的重要性并不在于“又提出了一种微调方法”，而在于提出一个非常关键的研究视角：

> 当新机器人适配困难时，问题不一定只是参数没有调好，也可能是输入或输出的表示空间本身不利于迁移。

这使 Fine-tuning 的研究对象从单纯的 parameter adaptation 扩展到 representation adaptation。

本章最后还会把这些思想迁移到一个实际问题：在 Fourier 机器人上部署 π0.5 完成药盒 Pick-and-Place，只有单路 RGB 视觉输入，已有约 500 条真实示教，LoRA 和 Full Fine-tuning 效果都不理想。这个案例可以帮助理解：当两种参数微调方式都失败时，下一步不应只是继续调训练超参数，而应该回到数据、接口、表示和分布偏移层面重新诊断。

---

## 2. 问题从哪里来：为什么预训练 VLA 仍然需要适配

一个 VLA 可以抽象为机器人策略：

$$
a_t
=
\pi_\theta
\left(
o_t,
l,
s_t
\right)
$$

其中：

- o_t：当前视觉 observation；
- l：语言指令；
- s_t：机器人 proprioceptive state；
- a_t：机器人动作；
- θ：模型参数。

如果模型在预训练阶段已经见过大量机器人数据，很容易产生一个直觉：

> 既然模型已经学了这么多数据，为什么换一台机器人还需要再训练？

问题在于，策略真正面对的条件远比公式中显式写出的变量复杂。实际部署时还隐含依赖：

- embodiment；
- camera pose；
- gripper；
- control interface；
- workspace geometry；
- action scale；
- observation preprocessing；
- normalization；
- task distribution；
- environment distribution。

因此，训练环境中的联合分布可以写成：

$$
p_{\text{train}}
\left(
o,
s,
a,
l
\right)
$$

而目标机器人上的分布为：

$$
p_{\text{target}}
\left(
o,
s,
a,
l
\right)
$$

只要二者存在明显差异，就会形成 distribution shift：

$$
p_{\text{train}}
\neq
p_{\text{target}}
$$

Fine-tuning 的本质，就是利用目标机器人或目标任务上的新数据，让策略在新的数据分布附近重新获得较高性能。

但需要特别强调：

> Fine-tuning 并不自动解决 representation mismatch。

如果 observation、state、action 的定义本身就和预训练时期不一致，直接更新模型参数可能是在要求神经网络“自己吸收接口差异”。

---

## 3. Task Adaptation 与 Robot Adaptation

新任务适配和新机器人适配经常同时被称为 Fine-tuning，但两者并不等价。

### 3.1 新任务适配

假设机器人、相机、控制接口均保持不变：

```text
原任务：Pick cup
新任务：Open drawer
```

此时变化主要来自行为目标：

$$
p_{\text{task-old}}
\rightarrow
p_{\text{task-new}}
$$

模型仍然面对相似的：

- visual domain；
- state definition；
- action semantics；
- kinematic constraints。

这种情况主要属于 Task Adaptation。

### 3.2 新机器人适配

如果从一种机器人切换到另一种机器人：

```text
WidowX
↓
Franka
```

即使语言指令仍然是：

```text
Pick up the red cup.
```

也可能同时变化：

- 机械臂几何；
- 工作空间；
- 相机安装方式；
- 末端夹爪；
- action scale；
- state dimension；
- controller；
- action latency。

此时变化不仅是 policy behavior，而是 observation、state、action 都可能发生 shift。

因此可以认为：

```text
Task Adaptation
主要改变“要做什么”

Robot / Embodiment Adaptation
同时可能改变“看到什么、自己是什么状态、动作是什么含义、如何执行”
```

这也是为什么新机器人适配通常比单纯的新任务适配更加困难。

---

## 思考节点 A｜“VLA 已经预训练过，为什么还不能直接上新机器人？”

### 问题

如果一个 VLA 已经在超过百万条机器人 demonstration 上预训练，为什么换一台机器人后仍然可能必须 Fine-tune？

### 解析

只回答“机器人本体变了”方向正确，但还不够完整。

真正的因果链是：

```text
机器人本体变化
↓
几何、相机、控制器、动作尺度、状态定义变化
↓
Observation / State / Action distribution shift
↓
预训练策略面对新的输入输出关系
↓
性能下降
↓
需要 adaptation
```

因此，Fine-tuning 不是因为预训练模型“不够大”，而是因为预训练和部署环境之间存在统计与接口差异。

---

# 4. Fine-tuning、SFT、Behavioral Cloning 与 Post-training

这几个术语在 VLA 文献中经常同时出现，需要先建立层级关系。

## 4.1 Fine-tuning

Fine-tuning 是最宽泛的概念之一：

> 在已经训练好的参数基础上，使用新的数据继续优化模型。

设预训练参数为 θ₀，目标机器人数据为 D_target：

$$
\theta^*
=
\arg\min_{\theta}
\mathbb{E}_{(o,l,s,a)\sim D_{\text{target}}}
\left[
\mathcal{L}
\left(
\pi_\theta(o,l,s),
a
\right)
\right]
$$

Fine-tuning 本身并不限定：

- 使用什么 loss；
- 更新多少参数；
- 是否只训练某个模块；
- 是否使用 RL。

---

## 4.2 Supervised Fine-Tuning

Supervised Fine-Tuning，简称 SFT，指：

> 使用带有目标输出标签的数据进行有监督微调。

对于机器人：

```text
observation + language + state
→
expert action
```

专家动作就是监督标签。

如果模型输出连续动作，可以使用回归损失：

$$
\mathcal{L}_{\text{MSE}}
=
\left\|
a_t^{\text{pred}}
-
a_t^{\text{expert}}
\right\|_2^2
$$

如果 action 被 token 化，则可能使用 Cross Entropy：

$$
\mathcal{L}_{\text{CE}}
=
-
\log
p_\theta
\left(
a_t^{\text{expert}}
\mid
o_t,
l,
s_t
\right)
$$

---

## 4.3 SFT 与 Behavioral Cloning

在机器人学习语境下，SFT 与 Behavioral Cloning 本质上常常描述同一个训练逻辑：

```text
专家 demonstration
↓
监督模型模仿 expert action
```

因此：

```text
LLM / VLM 社区
常称为 SFT

Robot Learning 社区
常称为 Behavioral Cloning
```

区别更多来自研究社区的语言，而不是优化本质。

---

## 4.4 Post-training

Post-training 是更宽泛的上层概念。

一条常见 VLA 训练链可以表示为：

```text
Foundation Pretraining
↓
Robot Pretraining
↓
Task / Robot Fine-tuning
↓
Post-training
```

Post-training 可能包含：

- SFT；
- LoRA；
- Full Fine-tuning；
- RL；
- Preference / reward-based optimization；
- representation adaptation；
- distillation；
- continual adaptation。

因此不能把：

```text
Post-training = LoRA
```

也不能把：

```text
Fine-tuning = LoRA
```

LoRA 只是 parameter adaptation 中的一种实现方式。

---

## 思考节点 B｜Fine-tuning 是否等于 LoRA？

### 问题

LoRA 与 Fine-tuning 是同一个层级的概念吗？

### 解析

不是。

正确层级为：

```text
Fine-tuning / Adaptation
├── Full Fine-tuning
├── Partial Fine-tuning
├── LoRA / PEFT
├── Action Head-only
└── 其它参数适配方式
```

Fine-tuning 描述的是“继续适配模型”这件事。

LoRA 描述的是“以低秩参数更新方式完成适配”。

如果把 Fine-tuning 等同于 LoRA，就会看不到更重要的问题：

- representation 是否匹配；
- state/action interface 是否一致；
- 哪些模块应该冻结；
- 是否存在 catastrophic forgetting；
- 是否需要新 robot-specific head；
- 数据量是否足够。

---

# 5. Parameter Adaptation：模型到底更新哪些参数

假设一个 VLA 由以下模块构成：

```text
Image
↓
Vision Encoder
↓
Projector / Multimodal Fusion
↓
VLM / LLM Backbone
↓
Action Module
↓
Robot Action
```

将参数写为：

$$
\theta
=
\{
\theta_v,
\theta_p,
\theta_b,
\theta_a
\}
$$

其中：

- θ_v：Vision Encoder；
- θ_p：Projector / multimodal adapter；
- θ_b：VLM / LLM backbone；
- θ_a：Action module。

不同 Fine-tuning 方法本质上就是选择不同的 trainable parameter set。

---

## 5.1 Full Fine-tuning

Full Fine-tuning 允许所有参数更新：

$$
\theta
\leftarrow
\theta
-
\eta
\nabla_\theta
\mathcal{L}
$$

优点：

- adaptation capacity 最大；
- 可以让视觉、语言、动作映射整体重构；
- 适合 target distribution 与 pretraining distribution 差异较大且数据充分的情况。

缺点：

- 显存和计算成本高；
- 小数据时容易 overfit；
- 更容易产生 catastrophic forgetting；
- 对训练稳定性要求更高。

---

## 5.2 Action Head-only

另一种极端是冻结 backbone：

```text
Vision Encoder       Frozen
Projector            Frozen
VLM Backbone         Frozen
Action Head          Trainable
```

它隐含的假设是：

> 模型已经能够正确理解视觉和语言，只是还不熟悉目标机器人的动作接口。

这种方案适用于：

```text
semantic understanding 基本没有变化
↓
主要变化发生在 action interface
```

例如同样的 RGB 观察和任务语义，但 target robot 使用新的 joint / Cartesian 控制接口。

局限是：如果视觉域、相机视角或空间映射也显著改变，仅训练 action head 无法修复早期表示错误。

---

## 5.3 Partial Fine-tuning

Partial Fine-tuning 处于两者之间。

例如：

```text
Vision Encoder    Frozen
LLM Backbone      Partial
Action Module     Trainable
```

或：

```text
Vision Encoder 的最后若干层可训练
Multimodal Projector 可训练
Action Module 可训练
```

其核心思想是：

> 只修改真正需要适配的模块，尽量保留预训练知识。

---

# 6. LoRA：低秩参数适配到底做了什么

LoRA 的核心不是“在网络最后增加一个小层”，而是在已有线性映射上学习一个低秩修正量。

设原线性层为：

$$
y
=
Wx
$$

其中：

$$
W
\in
\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

Full Fine-tuning 会直接更新整个 W。

LoRA 冻结 W，并学习：

$$
\Delta W
=
BA
$$

其中：

$$
A
\in
\mathbb{R}^{r\times d_{\text{in}}}
$$

$$
B
\in
\mathbb{R}^{d_{\text{out}}\times r}
$$

且：

```text
r ≪ d_in
r ≪ d_out
```

模型实际计算：

$$
y
=
Wx
+
\frac{\alpha}{r}
BAx
$$

因此，LoRA 实际上让：

$$
W'
=
W
+
\frac{\alpha}{r}
BA
$$

### 参数量直觉

如果：

```text
W = 4096 × 4096
```

Full Fine-tuning 需要训练约 1677 万个参数。

如果：

```text
r = 8
```

LoRA 只需要训练：

```text
8 × 4096
+
4096 × 8
≈
6.5 万
```

参数量大幅下降。

---

## 6.1 为什么低秩修正可以影响整个 Transformer

Transformer 中有大量线性映射：

```text
Q projection
K projection
V projection
Output projection
MLP projection
```

例如：

$$
Q
=
XW_Q
$$

应用 LoRA 后：

$$
Q
=
X
\left(
W_Q
+
B_QA_Q
\right)
$$

因此，虽然 LoRA 参数数量少，但它并不是只改变最终输出层，而可以在 Transformer 内部多个位置持续影响 feature transformation。

可以把它理解为：

```text
Pretrained model
保留原主干知识

+
Low-rank correction
学习任务 / 机器人相关修正
```

---

## 思考节点 C｜为什么 LoRA 能少训练很多参数

### 问题

如果 W 是一个巨大矩阵，为什么 BA 可以用远少于 Full Fine-tuning 的参数改变模型行为？

### 解析

关键不在于 BA 的矩阵尺寸最终是否与 W 一样。乘积 BA 的结果确实与 W 同 shape。

真正减少的是“独立可学习自由度”。

Full Fine-tuning：

```text
ΔW 的所有元素都可以独立变化
```

LoRA：

```text
ΔW 被约束为 rank ≤ r 的低秩结构
```

也就是说，模型只能沿一个较低维的子空间修改已有能力。

因此：

```text
可训练自由度下降
↓
显存 / 参数成本下降
↓
小数据过拟合风险通常降低
↓
但适配自由度也受到限制
```

LoRA 不是“更高级的 Full FT”，而是在 adaptation capacity 与效率之间做 trade-off。

---

# 7. Representation Adaptation：为什么“调参数”仍然可能不够

到这里会出现一个关键认知升级。

传统 Fine-tuning 默认：

```text
输入 representation 合理
输出 representation 合理
↓
只是模型参数还没有适应新任务
```

于是主要研究：

```text
如何更新 θ
```

但新机器人上的失败可能来自：

```text
问题根本没有被表示成模型最容易迁移的形式
```

例如：

- 不同机器人 action scale 差异巨大；
- 不同机器人的 Cartesian action 分布不同；
- camera pose 改变导致同一物体的 2D appearance-to-action 映射变化；
- 同一个 token 在不同 embodiment 下具有不同物理含义。

这时除了：

$$
\theta
\rightarrow
\theta'
$$

还可以改变：

$$
R_{\text{old}}
\rightarrow
R_{\text{new}}
$$

其中 R 表示 observation、state 或 action representation。

---

## 7.1 一个简单类比：单位变化

假设模型原来预测身高，训练标签全部是 cm：

```text
175
```

新的数据集却全部使用 inch：

```text
69
```

一种办法是：

```text
保持单位混乱
↓
让模型通过 Fine-tuning 自己学会换算
```

另一种办法是：

```text
先统一单位
↓
再做参数适配
```

后者就是 representation / interface alignment 的思想。

机器人上的 action representation mismatch 往往比“cm 和 inch”更加复杂。

---

## 思考节点 D｜Parameter Adaptation 与 Representation Adaptation

### 问题

如果换了新机器人，直接重新划分 action representation，属于 parameter adaptation 还是 representation adaptation？

### 解析

属于 representation adaptation。

两者的核心区别：

```text
Parameter Adaptation
研究“模型怎么学”

Representation Adaptation
研究“问题应该怎样表示给模型”
```

例如：

```text
LoRA
Full FT
Action Head-only
```

主要是 parameter adaptation。

而：

```text
重新设计 action grid
重新定义 canonical action space
显式加入 3D position encoding
```

主要属于 representation adaptation。

二者并不互斥。

最重要的组合思想是：

```text
先把新问题映射到更合适的表示空间
↓
再用 LoRA / Full FT 调整剩余参数差异
```

---

# 8. SpatialVLA：从参数微调扩展到空间表示适配

SpatialVLA 的核心研究问题可以概括为：

> 能否通过显式空间 representation，让 VLA 的 observation 和 action 更适合跨机器人预训练与 downstream adaptation？

它并不是简单增加一个 depth estimator，而是从输入和输出两端同时重新设计 representation。

整体逻辑：

```text
RGB Image
├── SigLIP
│   ↓
│   2D semantic visual features
│
└── Monocular Depth
    ↓
    3D camera-centric position

两者融合
↓
3D-aware visual tokens
↓
PaLiGemma2 / VLM
↓
Spatial Action Tokens
↓
Translation
Rotation
Gripper
```

---

# 9. SpatialVLA 输入端：Ego3D Position Encoding

## 9.1 普通视觉 token 的局限

视觉编码器可以从 RGB 中提取语义特征：

$$
X
=
f_{\text{vision}}
\left(
I
\right)
$$

其中 X 可以表示：

```text
这个 patch 很像红色盒子
这个 patch 很像桌面
这个 patch 很像夹爪
```

但仅依赖 2D visual feature，不会天然显式提供：

```text
这个物体离相机多远
这个物体在 camera frame 的 3D 位置
```

对于精确 manipulation，这些几何信息非常重要。

---

## 9.2 深度到 3D 坐标

从 RGB 图像估计深度：

$$
D
=
f_{\text{depth}}
\left(
I
\right)
$$

对于像素坐标 u、v 和对应深度 D，可以通过相机内参完成 back-projection：

$$
(u,v,D)
\rightarrow
(x,y,z)
$$

从而得到 3D position map：

$$
P
\in
\mathbb{R}^{3\times h\times w}
$$

P 表示每个视觉位置在 camera coordinate system 中对应的 3D 空间位置。

---

## 9.3 空间位置编码

3D position 不能直接与视觉 feature 相加，因此需要编码：

$$
P'
=
\operatorname{MLP}
\left(
\gamma(P)
\right)
$$

其中：

- P：原始 3D 坐标；
- γ：位置编码；
- MLP：将空间位置映射到视觉 embedding dimension；
- P'：可与视觉语义特征融合的 3D positional feature。

最终：

$$
O_{3D}
=
X
+
P'
$$

于是视觉 token 同时包含：

```text
是什么
+
在哪里
```

这就是 Ego3D Position Encoding 的直觉。

---

## 9.4 为什么使用 camera-centric，而不是 robot base frame

如果所有 3D 位置都表示在 robot base frame，需要知道精确的：

$$
T_{\text{camera}\rightarrow\text{base}}
$$

换机器人、换相机安装位置，通常需要重新标定 extrinsic。

SpatialVLA 使用 camera-centric representation 的关键动机之一是：

```text
不强依赖每台机器人特定的 camera-to-base calibration
↓
降低 embodiment-specific 坐标系依赖
↓
更适合 cross-embodiment representation
```

因此它强调的是 egocentric spatial representation。

---

## 思考节点 E｜X、P、P'、O₃D 到底有什么区别

### 问题

在 Ego3D 中，X、P、P'、O₃D 分别代表什么？

### 解析

正确数据流是：

```text
RGB
↓
Vision Encoder
↓
X
= 2D visual semantic feature

RGB
↓
Depth + Camera Geometry
↓
P
= 3D spatial coordinate

P
↓
Position Encoding + MLP
↓
P'
= 可与视觉 feature 融合的 spatial embedding

X + P'
↓
O₃D
= 3D-aware visual representation
```

容易出现的误区是把 P 和 P' 都理解成“深度”。

实际上：

- P 是几何坐标；
- P' 是经过网络映射后的 feature representation。

---

# 10. SpatialVLA 输出端：Adaptive Action Grids

## 10.1 从 7D Action 到 Spatial Token

典型机械臂动作可以表示为：

$$
a
=
[
\Delta x,
\Delta y,
\Delta z,
r_x,
r_y,
r_z,
g
]
$$

SpatialVLA 将动作分解为：

```text
Translation
Rotation
Gripper
```

其中 translation 部分：

$$
a_{\text{trans}}
=
[
\Delta x,
\Delta y,
\Delta z
]
$$

可以进一步转换为空间方向与距离：

$$
r
=
\sqrt{
\Delta x^2
+
\Delta y^2
+
\Delta z^2
}
$$

以及两个方向角。

因此，一次 7D action 可以被编码为：

```text
Translation spatial token
+
Rotation spatial token
+
Gripper token
```

即：

```text
7 个连续动作维度
↓
3 个空间动作 token
```

这与“每个动作维度独立量化成一个 token”的设计不同。

---

## 10.2 Adaptive 的含义

如果动作范围简单均匀离散：

```text
-10  -5  0  5  10
```

那么所有区间获得相同表示分辨率。

但机器人动作通常并不是均匀分布。

例如：

```text
绝大多数 Δx 都集中在 0 附近
很少出现极大位移
```

如果仍然均匀分配 token，会浪费大量 action vocabulary。

SpatialVLA 的思想是根据真实 action distribution 分配 grid resolution。

可以写成：

$$
a
\sim
\mathcal{N}
\left(
\mu,
\Sigma
\right)
$$

然后选择分割点，使每个 bin 的概率质量近似相同：

$$
P
\left(
a_i
<
a
<
a_{i+1}
\right)
\approx
\frac{1}{M}
$$

结果：

```text
高概率动作区域
→ grid 更密

低概率动作区域
→ grid 更稀
```

有限 token 被集中用于最常出现的机器人动作。

---

## 思考节点 F｜为什么 Adaptive Grid 不是普通量化

### 问题

“Equal probability bins” 与普通 uniform binning 的本质区别是什么？

### 解析

Uniform binning：

```text
按数值区间等距离切分
```

Adaptive / distribution-aware binning：

```text
按照数据概率质量切分
```

因此：

```text
数据越密集
↓
数值区间越窄
↓
局部分辨率越高
```

它强调的不是“动作范围更大”，而是：

> 将 representation capacity 分配到数据最需要精度的区域。

---

# 11. Spatial Embedding Adaptation：新机器人如何继承旧动作知识

这里是 SpatialVLA 与本章 Fine-tuning 主题结合最紧密的部分。

假设预训练 action grid 为：

$$
G_{\text{old}}
$$

新机器人 action distribution 发生明显变化：

$$
p_{\text{old}}(a)
\neq
p_{\text{new}}(a)
$$

SpatialVLA 不直接保留旧 grid，而是根据新机器人数据重新构造：

$$
G_{\text{new}}
$$

问题随之出现：

> 新 grid 中的 action token embedding 应该怎样初始化？

如果随机初始化：

$$
E_{\text{new}}
\sim
\text{Random}
$$

那么预训练过程中学习到的大量 spatial action knowledge 会被浪费。

因此使用旧 grid 中空间邻近的 embedding 做插值：

$$
\mathbf{e}_{\text{new}}^i
=
\sum_{j=1}^{K}
w_j
\mathbf{e}_j
$$

其中：

- e_newᶦ：新 action token embedding；
- e_j：邻近旧 action token embedding；
- w_j：根据空间距离计算的插值权重；
- K：参与插值的邻近旧 grid point 数量。

并满足：

$$
\sum_{j=1}^{K}
w_j
=
1
$$

直觉：

```text
新动作：
向右前方移动 3 cm

旧 grid 中：
向右前方 2 cm
向右前方 4 cm

↓
新 embedding
≈
两个相邻旧 embedding 的加权组合
```

这样，新 token 从一开始就具有与其空间动作接近的 pretrained prior。

---

## 思考节点 G｜为什么不能把新 action embedding 全部随机初始化

### 问题

如果 grid 已经根据新机器人重新划分，为什么不直接重新随机初始化 embedding，然后用新数据训练？

### 解析

这样做理论上可以，但会丢失预训练的 transfer advantage。

因果链：

```text
旧 action embedding
已通过大规模 robot data 学到空间动作语义
↓
新 grid 与旧 grid 在空间上仍有连续关系
↓
相近动作应具有相近初始化
↓
插值保留 spatial prior
↓
少量 downstream data 只需做局部修正
```

因此 Spatial Embedding Adaptation 的作用本质是：

> 提供更适合目标机器人 action distribution 的 representation，同时尽可能继承 pretrained spatial knowledge。

---

# 12. Representation Adaptation 与 LoRA 可以同时使用吗

可以，而且这正是非常重要的心智模型。

Representation Adaptation：

```text
改变问题怎样表示给模型
```

Parameter Adaptation：

```text
改变模型怎样处理这些表示
```

因此：

```text
新 Action Grid
+
Spatial Embedding Adaptation
+
LoRA
```

是完全合理的组合。

两条轴可以表示为：

```text
                    Parameter Adaptation
              LoRA                Full FT
                │                    │
旧 Representation ───────────────────
                │                    │
新 Representation ───────────────────
```

这意味着：

```text
Spatial Embedding Adaptation
≠
LoRA
```

二者解决不同问题。

---

# 13. SpatialVLA 的训练流程

## 13.1 Pretraining

预训练阶段大致结构：

```text
PaLiGemma2 / VLM backbone
+
Ego3D Position Encoding
+
Spatial Action Tokens
↓
Large-scale robot demonstrations
↓
Autoregressive action token prediction
```

动作经过 tokenizer 后成为目标 token：

$$
a_t^*
\rightarrow
\mathfrak{a}_t^*
$$

模型学习：

$$
p_\theta
\left(
\mathfrak{a}_t^*
\mid
O_{3D},
l
\right)
$$

训练目标仍然是监督模仿：

$$
\mathcal{L}
=
-
\log
p_\theta
\left(
\mathfrak{a}_t^*
\mid
O_{3D},
l
\right)
$$

因此从 Robot Learning 角度，本质仍然属于 BC / SFT。

SpatialVLA 的主要创新并不是提出新的 imitation loss，而是：

```text
Observation Representation
+
Action Representation
+
Downstream Representation Adaptation
```

---

## 13.2 Post-training

对新机器人：

```text
Target robot demonstrations
↓
统计新的 action distribution
↓
重新离散化 Action Grid
↓
Spatial Embedding Interpolation
↓
LoRA / Full Fine-tuning
↓
Target Robot Policy
```

因此其 downstream adaptation 可以写成：

```text
Representation Adaptation
+
Parameter Fine-tuning
```

---

# 14. 少量 Demonstration 为什么特别依赖好的初始化

设目标机器人只有 N 条 demonstrations。

当：

```text
N 很小
```

模型很难通过 target data 自己重新学习完整的 representation。

Full Fine-tuning 又会产生：

```text
过拟合
+
遗忘
+
训练不稳定
```

因此小数据 adaptation 的核心不是：

> 如何让模型从头重新学。

而是：

> 如何最大限度复用 pretrained prior，只用少量数据修正目标机器人真正不同的部分。

这也是 Spatial Embedding Adaptation 的重要意义。

可以用一个非常通用的关系理解：

```text
Downstream data 越少
↓
Initialization / Prior / Representation 越重要

Downstream data 越多
↓
模型越有能力直接从 target distribution 重新学习
```

---

# 15. Catastrophic Forgetting：为什么学会新任务会忘掉旧能力

设预训练模型在旧数据上的损失为：

$$
\mathcal{L}_{\text{old}}
\left(
\theta
\right)
$$

现在只优化新机器人数据：

$$
\theta^*
=
\arg\min_\theta
\mathcal{L}_{\text{new}}
\left(
\theta
\right)
$$

优化目标并没有要求：

$$
\mathcal{L}_{\text{old}}
\left(
\theta^*
\right)
$$

仍然保持较低。

因此完全可能出现：

$$
\mathcal{L}_{\text{new}}
\downarrow
$$

同时：

$$
\mathcal{L}_{\text{old}}
\uparrow
$$

这就是 Catastrophic Forgetting。

---

## 15.1 为什么 Full FT 风险更高

Full Fine-tuning 允许整个网络围绕新的窄分布更新：

```text
小规模 target data
+
大量 trainable parameters
↓
模型快速适应 target
↓
原通用 feature 可能被破坏
```

尤其当：

```text
target data 少
target distribution 窄
更新 epoch 多
```

风险更大。

---

## 15.2 为什么 LoRA 只能“降低风险”，不能保证不遗忘

LoRA 冻结 pretrained W，只学习：

$$
\Delta W
=
BA
$$

因此原始知识仍保存在 W 中。

但实际推理使用的是：

$$
W'
=
W
+
BA
$$

最终行为仍然发生改变。

所以：

```text
LoRA
通常比 Full FT 更保守
```

但不能推出：

```text
LoRA 不会 catastrophic forgetting
```

是否遗忘必须通过 old-task evaluation 验证。

---

## 15.3 常见知识保持方式

除了 LoRA，还可以：

### Freeze

只更新必要模块。

### Replay / Co-training

同时保留旧数据：

$$
D_{\text{train}}
=
D_{\text{new}}
\cup
D_{\text{old}}
$$

### Regularization / Distillation

约束新模型不要偏离旧模型过多：

$$
\mathcal{L}
=
\mathcal{L}_{\text{new}}
+
\lambda
D
\left(
\pi_\theta,
\pi_{\theta_0}
\right)
$$

这些方法共同处理一个核心矛盾：

> Adaptation 与 Knowledge Retention 之间的 trade-off。

---

## 思考节点 H｜冻结 text embedding 是否等于“解决遗忘”

### 问题

如果一篇 VLA 论文冻结部分 language embedding，是否可以直接说它解决了 catastrophic forgetting？

### 解析

不能。

一种设计“可能有助于保留知识”，并不等于论文已经证明“没有遗忘”。

真正验证 catastrophic forgetting 至少应该比较：

```text
Fine-tuning 前旧任务性能
↓
Fine-tuning 后旧任务性能
```

如果论文没有专门做 retention benchmark，则只能说：

```text
采用了有利于知识保持的设计
```

不能扩大成：

```text
解决了 catastrophic forgetting
```

这是科研阅读中非常重要的证据边界意识。

---

# 16. Data Efficiency：不是简单“少采数据”

Data Efficiency 的核心问题是：

> 达到相同性能需要多少 downstream robot data？

例如：

```text
方法 A：
500 demos → 80% success

方法 B：
50 demos → 80% success
```

方法 B 更 data-efficient。

在真实机器人中，这一点非常重要，因为 demonstration 的成本不仅是“录制一段 trajectory”。

还包括：

- 人工遥操作；
- Reset；
- 环境恢复；
- 设备磨损；
- 数据清洗；
- 失败 episode 处理；
- 相机与机器人标定；
- 人力时间。

因此理想的 Foundation VLA 应该逐渐接近：

```text
大规模通用预训练
+
极少量 embodiment-specific data
↓
高效适配新机器人
```

---

# 17. SpatialVLA Ablation 应该怎样读

研究者不能只看最终 benchmark，而要问：

> 性能提升究竟来自哪个模块？

这就是 Ablation Study 的作用。

---

## 17.1 Adaptive Action Grid 的作用

当 SpatialVLA 将 action representation 换回更普通的逐维离散方式后，部分任务性能明显下降。

这支持的结论不是：

```text
SpatialVLA 整体一定最优
```

而是更具体的：

> Action representation 本身显著影响 VLA manipulation performance。

因此，Fine-tuning 研究不应该只关注：

```text
训练多少参数
```

还应该关注：

```text
动作究竟怎样表示
```

---

## 17.2 Distribution-aware Grid 的作用

如果使用 uniform grid，而不是依据真实 action distribution 分配 resolution，表现会下降。

这支持：

```text
不仅需要 spatial grid
↓
grid 还应该匹配 robot action distribution
```

---

## 17.3 Grid Resolution

更高的 action grid resolution 可以提高动作表达精度。

但不会无限提升，因为更大的 vocabulary 同样带来：

- decoding cost；
- training difficulty；
- sparsity；
- vocabulary capacity trade-off。

因此应该理解为：

```text
Action precision
↔
Representation / inference cost
```

之间的平衡。

---

## 17.4 Ego3D 的作用

拿掉 Ego3D 后，一些 spatial / viewpoint variation 条件下性能下降。

它支持：

> 显式 3D-aware observation representation 有助于 manipulation robustness。

但它不能被扩大解释成：

> Ego3D 解决所有 distribution shift。

---

# 18. SpatialVLA 的真正 Contribution

如果只把 SpatialVLA 概括成：

```text
加入 depth
+
设计新的 action tokenizer
```

会低估它。

更准确的研究贡献是：

> 同时重新设计 VLA 的 observation representation 与 action representation，使预训练知识更容易以空间形式跨 embodiment 迁移，并在 downstream post-training 中进一步利用 representation adaptation 提高数据效率。

可以浓缩为：

```text
Observation Spatialization
+
Action Spatialization
+
Spatial Embedding Adaptation
```

这三个部分共同构成它最重要的研究逻辑。

---

# 19. SpatialVLA 的局限

## 19.1 Gaussian Action Modeling 不是普适假设

如果真实 action distribution：

- 多峰；
- 强偏态；
- 仅集中于少数方向；
- 含大量噪声；

单 Gaussian：

$$
\mathcal{N}
\left(
\mu,
\Sigma
\right)
$$

可能不能准确描述动作分布。

因此：

```text
Gaussian-based Grid
是一个有效工程假设
≠
最优 action representation
```

这为更灵活的：

- mixture model；
- learned codebook；
- latent action；
- continuous generative representation；

留下研究空间。

---

## 19.2 Autoregressive Action Token 的延迟

SpatialVLA 仍然需要顺序预测 action token。

这与：

```text
Diffusion Policy
Flow Matching
```

等连续生成方法不同。

因此一个自然研究问题是：

```text
Spatial Representation
+
Continuous Generative Policy
```

例如将空间表示思想与 flow matching VLA 结合。

---

## 19.3 3D Spatial Understanding 不等于 Long-horizon Planning

即使模型更准确地知道：

```text
物体在哪里
下一步应该往哪里移动
```

也不自动解决：

```text
长任务应该分几阶段
当前完成到哪一步
失败后是否需要 recovery
下一子任务是什么
```

因此：

```text
Spatial Representation
≠
Long-horizon reasoning
```

这会自然连接到后续 Agent + VLA、Memory、Planner 与 Failure Recovery。

---

# 20. 一个贯穿全章的完整机械臂实例

考虑一个新机器人执行：

```text
Pick the red medicine box and place it into basket B.
```

预训练模型来自其它机器人平台。

### 阶段 1：诊断 Distribution Shift

检查：

```text
Camera pose 是否变化
Robot geometry 是否变化
Gripper 是否变化
State definition 是否变化
Action definition 是否变化
Action scale 是否变化
Object / scene 是否变化
```

### 阶段 2：检查 Representation

如果预训练模型使用：

```text
Cartesian delta pose
```

目标机器人却记录：

```text
absolute joint position
```

首先需要解决 action representation mismatch。

如果 camera pose 大幅变化，也需要考虑：

```text
2D image-to-action mapping 是否出现明显 geometric shift
```

### 阶段 3：选择 Parameter Adaptation

如果 representation 基本匹配，数据较少：

```text
LoRA / Partial FT
```

如果数据充分且 domain shift 极大：

```text
Full FT
```

可以成为候选。

如果主要变化在 action interface：

```text
Action module-only
```

可能更合适。

### 阶段 4：检查 Knowledge Retention

不能只测新任务 success。

还要问：

```text
原有语言理解是否退化
旧 task 能力是否退化
OOD object 是否还能识别
```

### 阶段 5：评估 Data Efficiency

比较：

```text
50 demos
100 demos
250 demos
500 demos
```

而不是只报告：

```text
500 demos → 某成功率
```

这样才能知道方法是否真的提高了适配效率。

---

# 21. 与传统控制和 Robot Learning 的联系

VLA 并没有取代机器人控制器。

完整系统仍然可以表示为：

```text
Language / Task
↓
VLA Policy
↓
EEF / Joint Target / Action Chunk
↓
IK / Operational Space Controller / Joint Controller
↓
Actuator
↓
Robot
```

Fine-tuning 主要作用于：

```text
Learned Policy / Representation
```

而不是电机 PWM 层。

这意味着一个 VLA adaptation 失败，不一定是“模型不会”。

还可能是：

```text
模型输出正确
↓
Action denormalization 错
↓
Robot controller 收到错误量纲
↓
Closed-loop 失败
```

因此 VLA 真机研究必须把：

```text
Learning
+
Representation
+
Control Interface
+
Execution Pipeline
```

一起检查。

---

# 22. 真实案例：Fourier + π0.5 药盒 Pick-and-Place

现在将本章框架用于一个实际课题。

任务条件：

```text
Robot：
Fourier

Task：
药盒 / 小盒子 Pick-and-Place

Visual Input：
单路 RGB

Robot Data：
约 500 条真实 demonstration

Model：
π0.5

已尝试：
LoRA Fine-tuning
Full Fine-tuning

结果：
真机闭环表现不理想
```

最初路线大致为：

```text
π0.5 pretrained model
↓
500 demonstrations
↓
LoRA / Full FT
↓
Deploy
```

这条路线本身没有错误，但它隐含了一个很强的假设：

> Observation、State、Action、Normalization、Camera Geometry 与 Robot Interface 都已经适合 π0.5，剩余问题主要只是模型参数没有适配。

如果这个假设不成立，那么：

```text
LoRA bad
+
Full FT bad
```

并不能简单推出：

```text
π0.5 不适合 Fourier
```

反而可能说明：

> 真正瓶颈不在 parameter adaptation。

---

# 23. Fourier 项目中的四层诊断框架

更合理的诊断顺序是：

```text
Pipeline
↓
Distribution
↓
Representation
↓
Parameters
```

---

## 23.1 第一层：Pipeline Sanity

首先验证：

```text
Train / Inference state semantics 是否一致
Action semantics 是否一致
Normalization 是否一致
Denormalization 是否一致
Gripper encoding 是否一致
Action chunk 执行顺序是否一致
Control frequency 是否一致
```

例如：

训练时：

```text
gripper ∈ [0, 1]
```

部署接口却期待：

```text
gripper ∈ [-1, 1]
```

即使 VLA 学得很好，也会在真正抓取阶段失败。

因此必须逐维检查：

$$
s_{\text{train}}
\leftrightarrow
s_{\text{infer}}
$$

以及：

$$
a_{\text{train}}
\leftrightarrow
a_{\text{infer}}
$$

---

## 23.2 第二层：Distribution Sanity

如果训练数据采集时 camera geometry 与部署不同，例如：

```text
训练：
固定高度
固定角度

部署：
机器人高度变化
camera viewpoint 变化
```

那么同一个药盒在 image plane 中的位置和尺度都会变化。

即：

$$
p_{\text{train}}
\left(
I
\right)
\neq
p_{\text{deploy}}
\left(
I
\right)
$$

这种 shift 对语义识别可能影响不大，但对 precise manipulation 非常敏感。

```text
“这是药盒”
```

和：

```text
“夹爪应该沿哪个 3D 方向移动 2 cm”
```

不是同一种泛化能力。

因此：

```text
Semantic Generalization
≠
Visuomotor Geometric Generalization
```

---

## 23.3 第三层：Representation Sanity

需要检查：

```text
π0.5 预训练 action representation
与
Fourier target action representation
```

是否真正一致。

即使都是 Cartesian delta，也要检查：

- scale；
- coordinate frame；
- rotation convention；
- gripper convention；
- normalization；
- chunk horizon。

如果 Fourier 的 action distribution 与 pretrained embodiment 差异极大，模型可能需要的是：

```text
Action Representation Alignment
```

而不是单纯更大范围参数更新。

---

## 23.4 第四层：Parameter Adaptation

只有前三层相对可靠之后，再比较：

```text
LoRA
Full FT
Partial FT
Action Module-only
```

否则很可能发生：

```text
接口问题
↓
被误判为“模型容量问题”
↓
继续加大 Fine-tuning
↓
问题仍然存在
```

---

# 24. “500 条数据”为什么并不能直接说明数据足够

500 demonstrations 可能来自非常窄的分布：

```text
相机固定
物体位置接近
机器人初始状态接近
背景一致
抓取方式单一
```

从统计角度，它可能只是：

> 在同一个局部 manifold 上重复采样 500 次。

真正应该分析的是 coverage。

至少应查看：

```text
object position
object orientation
robot initial state
camera viewpoint
grasp approach
background
lighting
success / failure diversity
```

因此：

```text
Data Quantity
≠
Data Diversity
≠
Data Coverage
```

---

# 25. 为什么 LoRA 和 Full FT 都差，反而值得重视

如果只有 LoRA 表现差，可以提出一种解释：

```text
LoRA adaptation capacity 不足
```

因此尝试：

```text
Full FT
```

是合理的。

但如果：

```text
LoRA bad
+
Full FT bad
```

那么：

```text
“参数更新自由度不够”
```

这一解释就变弱了。

应该提高对以下问题的怀疑：

```text
Data Quality
Pipeline Bug
State / Action Mismatch
Normalization
Camera OOD
Representation Mismatch
```

这并不能数学上证明瓶颈一定不在参数，但从实验诊断逻辑上非常重要。

---

# 26. 从 SpatialVLA 得到的第一条研究启发：Action / State Representation Adaptation

可以统计 Fourier 数据中的每一维动作：

```text
mean
std
min
max
P1
P50
P99
```

并检查：

```text
Δx
Δy
Δz
rotation
gripper
```

是否具有异常尺度或高度偏置。

如果目标机器人 action distribution 与预训练 policy 使用的 action convention 差异明显，可以研究：

```text
Fourier-specific action
↓
Canonical Action Adapter
↓
π0.5 familiar action representation
```

可写成：

$$
a_{\text{canonical}}
=
f_\phi
\left(
a_{\text{Fourier}}
\right)
$$

执行时再做：

$$
a_{\text{Fourier}}
=
f_\phi^{-1}
\left(
a_{\text{canonical}}
\right)
$$

最简单的版本甚至可以只是：

$$
a'
=
D^{-1}
\left(
a-\mu
\right)
$$

这里真正借鉴 SpatialVLA 的不是 Adaptive Action Grid 的具体形式，而是：

> 不要要求 backbone 参数独自吸收所有 embodiment difference。

---

# 27. 第二条研究启发：RGB-only Spatial Adaptation

Fourier 项目没有 depth camera，但只有 RGB 并不意味着不能研究显式几何表示。

可以使用：

```text
RGB
↓
Monocular Depth
↓
Pseudo-depth
↓
3D / geometric feature
```

然后与 π0.5 的视觉特征融合。

这样可以形成一个明确的研究问题：

> 在 camera viewpoint / embodiment shift 下，显式 monocular geometric representation 是否能提高 RGB-only VLA 的 few-shot adaptation？

需要注意：

这不等于直接复制 SpatialVLA。

SpatialVLA 的 action generation 是 spatial token autoregressive 路线。

π0.5 则使用 continuous flow-based action generation。

因此更有价值的方向是：

```text
Spatial Representation
+
π0.5 Flow Policy
```

而不是机械复刻 SpatialVLA 全部模块。

---

# 28. 第三条研究启发：先验证 Failure Mode，再改模型

不应该一开始直接改 π0.5 architecture。

更合理的三层实验：

| 层次 | 实验目标 | 核心问题 |
|---|---|---|
| A. Pipeline Sanity | state/action/normalization/gripper/chunk | 模型输出是否被正确送到机器人 |
| B. Distribution Sanity | 完全复刻训练时 camera 与本体条件 | 去掉 OOD 后模型能否成功 |
| C. Research Adaptation | action adapter / augmentation / spatial adapter | representation adaptation 是否真正有增益 |

其中第二层最关键。

如果：

```text
原采集条件成功率很高
↓
camera height / angle 一变就崩
```

说明 viewpoint distribution shift 很可能是核心问题。

反之，如果：

```text
完全复刻训练条件仍然失败
```

优先级应回到：

```text
data / pipeline / action interface
```

而不是先研究几何泛化。

---

# 29. 真机抓取不能只看最终成功率

仅记录：

```text
Pick Success
```

很难定位问题。

应拆成阶段成功率：

```text
Reach
↓
Pre-grasp Alignment
↓
Contact
↓
Gripper Close
↓
Lift
↓
Place
```

例如：

| Stage | Success Rate |
|---|---:|
| Reach | 90% |
| Alignment | 65% |
| Contact | 55% |
| Gripper Close | 20% |
| Lift | 5% |

这种结果更像：

```text
gripper / action interface / final precision
```

问题。

如果：

| Stage | Success Rate |
|---|---:|
| Reach | 20% |
| Alignment | 10% |
| Contact | 很少发生 |

则更像：

```text
Observation / Spatial Grounding / Action Direction
```

问题。

这比只观察 training loss 更适合真实机器人研究。

---

# 30. 一个更具研究价值的 Fourier + π0.5 实验路线

可以构造逐级 baseline：

```text
B₀
π0.5 + LoRA

B₁
B₀ + State / Action Representation Alignment

B₂
B₁ + Viewpoint Diversity / Augmentation

B₃
B₂ + RGB-derived Spatial Adapter
```

同时在多个 distribution shift 条件测试：

```text
Same View
Camera Height Shift
Camera Angle Shift
Object Position Shift
Object Appearance Shift
```

再增加 data efficiency 曲线：

```text
50 demos
100 demos
250 demos
500 demos
```

这样研究问题就从：

> LoRA 和 Full FT 哪个更好？

升级成：

> How can a pretrained VLA be efficiently adapted to a new embodiment under geometric distribution shift?

这已经是一个更完整的研究命题。

---

# 31. 常见误区与阶段纠偏

## 31.1 误区：Fine-tuning 就是 LoRA

纠偏：

```text
Fine-tuning 是上层概念
LoRA 是 parameter adaptation 技术
```

---

## 31.2 误区：Full FT 比 LoRA 更新更多参数，所以一定更强

纠偏：

```text
Adaptation capacity 更高
≠
目标性能一定更高
```

小数据时可能出现：

- overfitting；
- catastrophic forgetting；
- unstable optimization。

---

## 31.3 误区：如果模型不适应新机器人，只要训练更多参数即可

纠偏：

失败可能来自：

```text
Representation mismatch
Normalization mismatch
Coordinate frame mismatch
Camera distribution shift
Data coverage
```

参数不是唯一变量。

---

## 31.4 误区：SpatialVLA 的 Spatial Embedding Adaptation 就是一种 LoRA

纠偏：

```text
Spatial Embedding Adaptation
→ Representation

LoRA
→ Parameter
```

它们可以同时使用。

---

## 31.5 误区：加入 3D 信息之后，就解决了长时任务

纠偏：

```text
Spatial understanding
解决“在哪里、怎么移动”

Long-horizon planning
解决“先做什么、之后做什么、是否需要恢复”
```

属于不同问题。

---

## 31.6 误区：500 条 demonstration 一定足够

纠偏：

需要同时考虑：

```text
Quantity
Diversity
Coverage
Quality
Consistency
```

500 条高度重复数据可能不如 100 条高覆盖数据。

---

## 31.7 误区：LoRA 与 Full FT 都失败，就说明基础模型不行

纠偏：

这反而应该触发更系统的 failure diagnosis：

```text
Pipeline
Distribution
Representation
Parameters
```

---

# 32. 本章内容在 VLA 学习链中的位置

完整知识链：

```text
Robot Policy
↓
Transformer
↓
VLM
↓
Behavioral Cloning / SFT
↓
Action Representation
↓
Action Chunking
↓
Diffusion / Flow Matching
↓
VLA Foundation Model
↓
VLA Fine-tuning / Adaptation
↓
Agent + VLA
↓
VLA + RL
```

本章位于：

```text
Foundation VLA
↓
Target Robot / Target Task Adaptation
```

也是从“理解模型”进入“开展研究”的重要分界线。

在后续研究中，本章内容会继续与：

- Real-time inference；
- Agentic recovery；
- RL post-training；
- World Model；
- continual adaptation；

发生连接。

---

# 33. 研究者视角：看到一篇 VLA Fine-tuning 论文应该问什么

不要先问：

```text
它用 LoRA 还是 Full FT？
```

更应该按顺序问：

### 1. Distribution Shift 到底发生在哪里

```text
Task?
Robot?
Vision?
Action?
Environment?
```

### 2. Observation / State / Action Representation 是否一致

```text
Coordinate frame?
Scale?
Normalization?
Action convention?
Camera geometry?
```

### 3. 论文有没有做 Representation Adaptation

还是全部依赖模型参数自己吸收差异？

### 4. 更新哪些 Parameters

```text
Vision?
Projector?
Backbone?
Action Head?
LoRA?
Full FT?
```

### 5. Downstream Data 有多少

不仅看数量，还看：

```text
coverage
diversity
quality
```

### 6. 是否发生 Knowledge Forgetting

有没有 old-task retention evaluation？

### 7. Data Efficiency 如何

有没有：

```text
50 / 100 / 250 / 500 demos
```

这样的曲线？

### 8. 真机失败发生在哪个阶段

```text
Perception?
Reach?
Alignment?
Grasp?
Lift?
Place?
```

### 9. Ablation 是否真正支持论文结论

是否把：

```text
representation
parameter update
data volume
```

分别拆开测试？

### 10. Improvement 是否来自更好的研究机制

还是只是：

```text
更多数据
更多参数
更多训练步数
```

---

# 34. 综合思考

## 综合题 1

一个 pretrained VLA 在新机器人上使用 LoRA 后失败，Full Fine-tuning 仍失败。最合理的下一步是什么？

### 解析

不能继续简单得出：

```text
模型 capacity 还不够
```

应该重新检查：

```text
Pipeline
↓
Distribution
↓
Representation
↓
Parameters
```

特别关注：

- state / action semantics；
- normalization；
- gripper；
- camera shift；
- action distribution；
- dataset coverage。

---

## 综合题 2

如果只换了机器人控制接口，但视觉环境、任务语义几乎不变，为什么可以优先考虑冻结 backbone？

### 解析

因为：

```text
Visual / Semantic Representation
仍然有价值
```

真正变化主要发生在：

```text
Feature → Robot Action
```

如果无理由修改整个 backbone：

```text
计算成本增加
+
遗忘风险增加
```

却不一定获得收益。

---

## 综合题 3

为什么 Representation Adaptation 对 few-shot robot adaptation 特别重要？

### 解析

few-shot 数据无法支持模型重新学习所有东西。

好的 representation 可以：

```text
把 target robot 映射到 pretrained model 更熟悉的空间
↓
保留 prior
↓
减少需要 downstream data 学习的差异
```

因此：

```text
Data 越少
↓
Prior / Initialization / Representation 越重要
```

---

## 综合题 4

为什么 SpatialVLA 的经验不能简单变成“给 π0.5 加 Adaptive Action Grid”？

### 解析

因为两者 action generation mechanism 不同。

SpatialVLA：

```text
Autoregressive Spatial Action Tokens
```

π0.5：

```text
Continuous Flow-based Action Generation
```

真正值得迁移的是研究思想：

```text
显式空间表示
+
跨 embodiment representation alignment
+
few-shot adaptation
```

而不是机械复刻具体模块。

---

## 综合题 5

Fourier + π0.5 项目中，如果完全复刻数据采集时的 camera pose 后成功率明显提高，而稍微改变相机高度就失败，最有可能说明什么？

### 解析

这说明：

```text
base policy 在训练分布内基本可用
```

而主要 failure mode 与：

```text
Viewpoint / Geometry Distribution Shift
```

高度相关。

这时就有条件进一步研究：

```text
Viewpoint augmentation
Spatial adapter
RGB-derived geometry
```

而不是继续盲目增加 fine-tuning capacity。

---

# 35. 本章总结

完成本章后，需要形成以下六个核心结论。

### 第一：Fine-tuning 不是 LoRA 的同义词

Fine-tuning 是 adaptation 的上层概念。

LoRA、Full FT、Partial FT、Action Head-only 只是不同 parameter adaptation 方法。

### 第二：新机器人适配首先是 Distribution Shift 问题

必须先判断：

```text
什么变了
```

再决定：

```text
改什么
```

### 第三：VLA Adaptation 不只有 Parameter Adaptation

完整框架应为：

```text
Distribution Shift Diagnosis
+
Representation Adaptation
+
Parameter Adaptation
+
Knowledge Retention
+
Data Efficiency
```

### 第四：SpatialVLA 的核心启发是“表示也可以被适配”

它通过：

```text
Ego3D Position Encoding
+
Adaptive Action Grid
+
Spatial Embedding Adaptation
```

说明 observation 与 action representation 本身可以成为跨机器人泛化与 few-shot adaptation 的研究对象。

### 第五：少数据时 Prior 与 Initialization 特别重要

当 target robot demonstrations 较少时：

```text
更好的 pretrained prior
+
更好的 representation
+
更保守的 parameter update
```

往往比直接大规模 Full Fine-tuning 更重要。

### 第六：真实机器人失败必须先定位，再提出新方法

对于 Fourier + π0.5：

```text
500 demos
+
LoRA / Full FT 失败
```

不应该直接解释为“模型不够强”。

更合理的研究流程是：

```text
Pipeline Sanity
↓
Distribution Sanity
↓
Representation Sanity
↓
Parameter Adaptation
↓
Stage-wise Closed-loop Evaluation
↓
Research Method
```

最终应该形成的心智模型是：

```text
VLA Adaptation
不是“如何把模型训得更久”

而是：

如何让 pretrained knowledge
以更低的数据和计算成本
正确迁移到新的机器人、任务和环境中
```

---

# 36. 关键词

**Fine-tuning — 微调 — 在预训练参数基础上继续适配目标数据。**

**Supervised Fine-Tuning, SFT — 监督微调 — 使用专家动作标签继续训练模型。**

**Behavioral Cloning, BC — 行为克隆 — 通过监督学习模仿专家策略。**

**Post-training — 后训练 — Foundation / robot pretraining 完成后的进一步能力优化阶段。**

**Full Fine-tuning — 全量微调 — 更新模型全部或绝大多数参数。**

**Partial Fine-tuning — 部分微调 — 只更新选定模块。**

**Action Head-only — 仅动作头训练 — 冻结视觉 / VLM 主干，只适配动作映射。**

**LoRA — Low-Rank Adaptation — 使用低秩矩阵学习预训练权重的修正量。**

**PEFT — Parameter-Efficient Fine-Tuning — 参数高效微调方法的统称。**

**Distribution Shift — 分布偏移 — 训练与部署数据分布不一致。**

**Task Adaptation — 任务适配 — 在相同或相近 embodiment 上学习新的行为目标。**

**Robot / Embodiment Adaptation — 本体适配 — 将策略迁移到新的机器人平台。**

**Representation Adaptation — 表示适配 — 修改 observation、state 或 action 的表示方式，使其更适合迁移。**

**Ego3D Position Encoding — 自我中心三维位置编码 — 将 camera-centric 3D geometry 融合到视觉 token。**

**Adaptive Action Grid — 自适应动作网格 — 根据动作分布为不同空间区域分配不同表示分辨率。**

**Spatial Embedding Adaptation — 空间嵌入适配 — 根据新动作网格用旧 spatial embedding 插值初始化新 token。**

**Catastrophic Forgetting — 灾难性遗忘 — 新任务适配后旧能力显著退化。**

**Knowledge Retention — 知识保持 — 在适配新任务时尽量保存预训练能力。**

**Data Efficiency — 数据效率 — 达到目标性能所需的 downstream 数据量。**

**Action Representation — 动作表示 — 模型内部如何编码并输出机器人动作。**

**Camera-centric Coordinate — 相机中心坐标系 — 以摄像机为参考描述 3D 空间位置。**

**Canonical Action Space — 规范动作空间 — 用统一中间表示减少不同机器人 action interface 的差异。**

**Stage-wise Evaluation — 分阶段评估 — 将 Reach、Alignment、Grasp、Lift、Place 等阶段分别统计成功率。**

**Viewpoint Distribution Shift — 视角分布偏移 — 数据采集与部署时相机位置、角度或高度变化导致的视觉几何分布变化。**

**RGB-only Spatial Adaptation — 纯 RGB 空间适配 — 利用单目深度或几何特征增强 RGB VLA 的空间表示能力。**

---

## 结语

VLA Fine-tuning 的真正研究问题不是：

```text
LoRA rank 应该设多少？
```

而是：

```text
预训练模型与目标机器人之间究竟差在哪里？
↓
哪些差异应该由数据解决？
哪些差异应该由 representation 解决？
哪些差异才应该交给参数更新？
↓
如何用尽可能少的新机器人数据
保留尽可能多的 pretrained knowledge
并获得稳定的真实闭环性能？
```

一旦形成这套分析框架，Fine-tuning 就不再只是一个训练工程问题，而成为一个可以系统研究的 VLA adaptation 问题。
