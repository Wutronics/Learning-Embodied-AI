# 13｜RT-1：从单任务模仿学习到大规模通用机器人策略

## 1. 本章导言：为什么 RT-1 是 VLA 发展史上的关键节点

在现代机器人学习的发展过程中，一个长期存在的问题是：机器人能否像视觉模型、语言模型一样，通过扩大数据规模和模型容量，从大量不同任务中学习出一个共享的、能够泛化的策略，而不是为每一个任务单独训练一套控制网络？

RT-1（Robotics Transformer 1）正是在这一背景下出现的。

它所代表的技术路线可以概括为：

```text
单任务机器人策略
        ↓
多任务模仿学习
        ↓
语言条件化策略
        ↓
大规模真实机器人数据
        ↓
共享 Transformer 策略
        ↓
Generalist Robot Policy
```

RT-1 并不是今天意义上最完整的 Vision-Language-Action Model。它没有像后来的 RT-2 那样充分利用互联网规模视觉语言预训练模型中的世界知识，也没有采用 Diffusion、Flow Matching 等现代连续生成策略。

但它完成了一个非常重要的过渡：

> 机器人策略开始从“针对某个任务训练一个模型”，走向“使用大量真实机器人数据训练一个统一的、多任务、语言条件化策略”。

因此，从 VLA 的发展历史来看，RT-1 的价值不只在于某一个网络模块，而在于它把以下几个思想第一次较完整地结合到了真实机器人控制系统中：

```text
大规模真实机器人数据
+
自然语言任务条件
+
视觉感知
+
Transformer
+
统一动作表示
+
多任务行为克隆
```

理解 RT-1，可以帮助建立现代 VLA 最基本的一条数据流：

```text
视觉 Observation
        +
语言 Instruction
        ↓
多模态机器人策略
        ↓
Action
        ↓
真实机器人执行
```

本章将沿着“为什么需要 RT-1 → 输入输出 → 模块结构 → 数学机制 → 数据与训练 → 泛化 → 局限 → RT-2”的顺序展开。

---

# 2. 问题从哪里来：传统单任务机器人策略的扩展瓶颈

## 2.1 单任务策略的基本形式

传统模仿学习中，可以针对一个机器人任务训练策略：

$$
\pi_\theta(a_t \mid o_t)
$$

其中：

- oₜ：时刻 t 的机器人 observation；
- aₜ：时刻 t 的机器人动作；
- θ：策略参数。

例如，对于“抓取苹果”任务，可以训练：

$$
\pi_{\theta_A}(a_t \mid o_t)
$$

对于“打开抽屉”任务，再训练：

$$
\pi_{\theta_B}(a_t \mid o_t)
$$

对于“将杯子放进抽屉”任务，则可能需要第三个策略：

$$
\pi_{\theta_C}(a_t \mid o_t)
$$

这种方式在任务数量较少时是可行的。

但一旦机器人需要处理几十、几百甚至更多任务，问题就会迅速显现。

---

## 2.2 问题一：任务之间无法充分共享知识

考虑两个任务：

```text
把红色苹果放进盘子
把绿色苹果放进盘子
```

从人的角度看，两者共享大量知识：什么是苹果、如何接近目标、如何闭合夹爪、如何抬起物体、如何移动到盘子以及如何完成放置。

如果两个任务分别训练完全独立的策略，那么已经学到的大量操作技能无法自然共享。

机器人学习因此需要一种更一般的形式：

$$
\pi_\theta(a_t \mid o_t,l)
$$

其中 l 表示自然语言任务指令。

现在，同一套参数 θ 可以执行不同任务。比如，同一个视觉场景下，Instruction 分别为 “Pick up the apple.” 与 “Pick up the cup.” 时，策略应输出不同动作。

这就是 Language-conditioned Robot Policy 的基本思想。

---

## 2.3 问题二：任务规模难以扩展

如果每增加一个任务就需要“采集一套数据 → 训练一个模型 → 部署一个模型 → 维护一个模型”，那么机器人系统很难扩展到数百种技能。

更理想的方式是：

```text
大量不同任务的数据
        ↓
共享训练
        ↓
一个统一策略
```

也就是：

$$
a_t=\pi_\theta(o_t,l)
$$

模型参数 θ 在所有任务之间共享。

这种参数共享，是 RT-1 实现多任务学习和任务泛化的重要基础。

---

# 3. 阶段思考 A：RT-1 真正改变了什么？

### 问题 1

RT-1 相比传统 Behavior Cloning，是否提出了一种完全不同的学习算法？

### 解析

不是。RT-1 的训练本质仍然属于监督式模仿学习，也就是 Behavior Cloning。

真正发生变化的是：单任务变成多任务，小规模数据变成大规模真实机器人数据，无语言条件变成 Language-conditioned，普通策略网络变成 Transformer Policy，连续动作回归变成离散 Action Token 预测。

因此，RT-1 的贡献不能简单理解为“发明了一种新的 Loss”。它真正推动的是机器人策略的规模化与通用化。

### 问题 2

如果策略已经有图像输入，为什么还必须加入语言指令？

### 解析

因为一个视觉状态可能对应多个合理任务。Image 描述“现在世界是什么样”，Language 描述“当前希望机器人做什么”，两者承担不同功能。

---

# 4. RT-1 的策略输入与输出

RT-1 可以抽象为：

$$
a_t
=
\pi_\theta(I_{t-h+1:t},l)
$$

其中：

- I：机器人视觉图像；
- l：自然语言任务指令；
- h：视觉历史长度；
- aₜ：当前机器人动作。

RT-1 使用最近 6 帧视觉观测，因此可以写为：

$$
a_t
=
\pi_\theta(I_{t-5:t},l)
$$

这里的一个重要特点是：RT-1 主要预测当前需要执行的动作，而不是像 ACT 那样一次生成较长的未来 Action Chunk。

因此其闭环运行形式可以理解为：

```text
最近 6 帧图像
      +
语言任务指令
      ↓
RT-1
      ↓
当前 Action
      ↓
机器人执行
      ↓
产生新的 Observation
      ↓
重新推理
```

这是一种持续闭环的 learned policy。

---

# 5. 为什么不能只使用当前一帧图像？

单张图像只能提供当前空间状态，却无法仅根据这一帧准确判断机械臂正在靠近还是正在离开、当前动作是否刚刚发生、夹爪运动趋势是什么、操作过程发展到了什么阶段。

使用多个历史帧：

$$
I_{t-5:t}
$$

可以为策略提供一定的时间信息。

需要注意，这并不等于模型显式获得了完整的机器人动力学模型。它只是通过最近一段视觉历史，让策略能够从时序变化中推断部分动态信息。

---

# 6. RT-1 的完整网络数据流

RT-1 的主要架构可以概括为：

```text
Language Instruction
        ↓
语言表示
        ↓
        ├───────────────┐
        │               │
Image → EfficientNet → FiLM
                        ↓
                  视觉语言特征
                        ↓
                  TokenLearner
                        ↓
                  每帧 8 Token
                        ↓
               6 帧共 48 Token
                        ↓
               Decoder-only
                 Transformer
                        ↓
                 Action Tokens
                        ↓
                Robot Action
```

这一结构包含四个尤其重要的模块：EfficientNet、FiLM、TokenLearner、Transformer。除此之外，还有一个非常关键但容易被忽视的设计：Action Discretization，即把机器人连续控制量转换为离散动作类别。

---

# 7. EfficientNet：从原始图像到视觉特征

假设机器人相机输入：

$$
I_t\in\mathbb R^{300\times300\times3}
$$

原始 RGB 图像包含大量低层像素。机器人真正需要的是物体、位置、形状、空间关系、操作对象以及机器人自身状态在图像中的表现。

因此需要视觉编码器：

$$
F_t=f_{\text{vision}}(I_t)
$$

其中：

- Iₜ：原始 RGB 图像；
- fvision：视觉编码网络；
- Fₜ：高维视觉 Feature Map。

RT-1 使用 EfficientNet 作为视觉 Backbone。EfficientNet 并不直接决定机器人应该做什么，它首先解决的是：如何把高维 RGB 图像转换成后续策略网络能够处理的视觉表示。

---

# 8. FiLM：让语言直接影响视觉特征

RT-1 架构中很重要的一步，是 Feature-wise Linear Modulation，也就是 FiLM。

## 8.1 为什么需要语言调制？

当视觉图像完全相同，但任务分别是 “Pick up the apple.” 和 “Pick up the cup.” 时，同一张图像中真正与任务有关的区域发生了变化。

因此，我们希望视觉特征并不是“Image → 永远得到相同 Feature”，而是：

```text
Image + Language
↓
Task-conditioned Visual Feature
```

FiLM 就承担这个功能。

## 8.2 FiLM 的数学形式

首先编码语言：

$$
e_l=f_{\text{text}}(l)
$$

接着，根据语言表示产生调制参数：

$$
\gamma=g_\gamma(e_l)
$$

$$
\beta=g_\beta(e_l)
$$

最后作用于视觉特征：

$$
F'
=
\gamma\odot F+\beta
$$

其中：

- F：原始视觉特征；
- γ：语言条件产生的缩放参数；
- β：语言条件产生的偏移参数；
- ⊙：逐元素乘法或按 Feature Channel 广播；
- F′：经过语言调制后的视觉特征。

## 8.3 一个简化数值例子

假设某处视觉表示简化为：

$$
F=
\begin{bmatrix}
0.8\\
0.5\\
0.2
\end{bmatrix}
$$

若语言指令是 “Pick up the red apple.”，模型可能产生：

$$
\gamma=
\begin{bmatrix}
2\\
0.5\\
0.1
\end{bmatrix}
$$

暂时忽略 β，则：

$$
F'=
\begin{bmatrix}
1.6\\
0.25\\
0.02
\end{bmatrix}
$$

这意味着与任务相关的信息被加强，而不重要的信息受到抑制。实际神经网络中的某一个 Feature Channel 并不会简单对应“苹果”或者“杯子”，这个例子只是为了建立 FiLM 的直觉。

---

# 9. 阶段思考 B：FiLM 和 Token 拼接是一回事吗？

### 问题

既然 Transformer 本身就可以融合不同 Token，为什么 RT-1 不直接把 Language Token 和 Vision Token 拼在一起，而要使用 FiLM？

### 解析

两种方法都可以实现多模态条件化，但作用位置不同。

直接 Token 拼接是在 Transformer 内部完成跨模态交互；FiLM 则是利用语言产生 γ、β，直接改变视觉 Feature，使视觉特征在进入 Transformer 之前就已经 Task-conditioned。

因此，FiLM 不能简单等同于 Token Concatenation。

---

# 10. TokenLearner：为什么需要压缩视觉信息？

视觉 Transformer 面临一个基本问题：Token 太多。

标准 Self-Attention 为：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

若 Token 数量为 N，则注意力矩阵为：

$$
N\times N
$$

计算量随 N 的增长很快。

假设一帧产生 196 个视觉 Token，6 帧就是：

$$
196\times6=1176
$$

Attention Matrix 大小为：

$$
1176\times1176
$$

对于需要持续运行的真实机器人而言，这会明显增加推理计算量。RT-1 因此使用 TokenLearner。

---

# 11. TokenLearner 的基本机制

TokenLearner 的目标不是简单从图像里固定选择 8 个 Patch，而是从整张 Feature Map 中学习如何聚合重要信息，形成少量 Learned Tokens。

假设视觉 Feature Map 为：

$$
F\in\mathbb R^{H\times W\times C}
$$

对于第 i 个输出 Token，可以学习一个空间权重：

$$
\alpha_i(h,w)
$$

然后进行加权聚合：

$$
z_i
=
\sum_{h=1}^{H}
\sum_{w=1}^{W}
\alpha_i(h,w)F(h,w)
$$

其中：

- F(h,w)：空间位置 (h,w) 上的视觉 Feature；
- αᵢ(h,w)：该位置对第 i 个 Token 的重要程度；
- zᵢ：最终聚合出的视觉 Token。

RT-1 将每帧压缩成 8 个 Token：

$$
Z_t\in\mathbb R^{8\times C}
$$

对于最近 6 帧：

$$
6\times8=48
$$

最终只有 48 个视觉 Token 进入 Transformer。

如果用于理解的对照方案具有 1176 个 Token，而 RT-1 只有 48 个，则：

$$
\frac{1176^2}{48^2}
=
600.25
$$

单纯比较 Attention Matrix 元素数量，会下降约 600 倍。但这并不意味着整个 RT-1 的总计算量直接减少 600 倍，因为 EfficientNet、FiLM、TokenLearner 自身以及 Transformer 中的 MLP 等模块仍然需要计算。

---

# 12. 阶段思考 C：TokenLearner 到底在“选什么”？

### 问题

每帧最终只有 8 个 Token，能否理解为“模型从原图里挑出了 8 个最重要的固定图像块”？

### 解析

不能。TokenLearner 的思想更接近：整张 Feature Map → 学习多个空间加权方式 → 每种方式聚合一个表示 → 得到 8 个 Learned Tokens。

因此，一个 Token 可以融合多个空间位置的信息，而不是必须对应原始图像中的某一个固定方块。这也是 Token Compression 与简单下采样之间的重要区别。

---

# 13. Transformer：对视觉历史进行序列建模

经过前面的处理，每一帧得到 8 个 Token：

$$
Z_t
=
[z_t^1,z_t^2,\ldots,z_t^8]
$$

6 帧视觉历史合并后：

$$
Z
=
[Z_{t-5},Z_{t-4},\ldots,Z_t]
$$

视觉 Token 总数：

$$
N=48
$$

这些 Token 被送入 Decoder-only Transformer。本章讨论的 RT-1 使用 8 层 Transformer。

其核心任务可以抽象成：

$$
\pi_\theta
\left(
a_t
\mid
I_{t-5:t},l
\right)
$$

Transformer 利用注意力机制建立不同时间和不同视觉特征之间的关系，从而对经过压缩后的视觉历史进行时序和上下文建模，并生成当前动作。

---

# 14. Transformer 不等于语言模型

这是理解 RT-1 时一个非常重要的概念纠偏。

RT-1 使用 Decoder-only Transformer，ChatGPT 等语言模型也大量使用 Decoder-only Transformer，但相同架构不等于相同能力。

一个模型能够具备什么能力，取决于训练数据、训练目标、模型规模和表示方式。

语言模型主要学习：

$$
P(x_t\mid x_{<t})
$$

其中 x 通常是文本 Token。

RT-1 学习的是：

$$
P(a_t\mid I_{t-5:t},l)
$$

其中输出是机器人动作表示。

因此，RT-1 使用 Transformer，并不意味着它天然获得了大语言模型中的世界知识、常识和语言推理能力。

---

# 15. Action Representation：为什么 RT-1 把连续动作离散化？

机器人控制变量通常是连续量。例如，可以抽象成：

$$
a_t
=
\begin{bmatrix}
\Delta x\\
\Delta y\\
\Delta z\\
\Delta r_x\\
\Delta r_y\\
\Delta r_z\\
g
\end{bmatrix}
$$

传统连续 Behavior Cloning 可以直接预测：

$$
\hat a_t\in\mathbb R^D
$$

然后采用 MSE：

$$
\mathcal L
=
\left\|
\hat a_t-a_t^*
\right\|_2^2
$$

RT-1 采用了另一种思路：将连续动作区间离散化。

---

# 16. 256-bin Action Discretization

假设一个动作维度已经归一化到：

$$
[-1,1]
$$

将整个区间平均分成 256 个 Bin。

每个 Bin 宽度：

$$
\Delta
=
\frac{2}{256}
=
0.0078125
$$

假设专家动作：

$$
a=0.5
$$

那么它对应的离散类别近似可以写成：

$$
k
=
\left\lfloor
\frac{a-(-1)}{2}
\times256
\right\rfloor
$$

代入：

$$
k
=
\left\lfloor
\frac{1.5}{2}
\times256
\right\rfloor
=
192
$$

于是连续动作 0.5 被映射成离散 Action Token 192。模型不再直接回归 0.5，而是预测 0～255 中的一个类别。

---

# 17. 为什么离散后使用 Cross-Entropy？

对于某一个动作维度，模型输出：

$$
P_\theta(k\mid o,l)
$$

专家标签为：

$$
k^*
$$

对应损失：

$$
\mathcal L
=
-\log
P_\theta(k^*\mid o,l)
$$

这就是分类任务中的 Cross-Entropy / Negative Log-Likelihood。

因此，RT-1 使用交叉熵的根本原因不是 Cross-Entropy 天然比 MSE 更先进，而是动作表示被离散化后，预测问题变成了分类。

---

# 18. 动作离散化带来的代价：Quantization Error

离散化一定会损失部分精度。

如果区间宽度：

$$
\Delta
=
\frac{2}{256}
$$

使用 Bin Center 还原动作时，最大量化误差约为：

$$
\epsilon_{\max}
=
\frac{\Delta}{2}
$$

因此：

$$
\epsilon_{\max}
=
\frac{1}{256}
\approx
0.0039
$$

这里的 0.0039 是归一化 Action Space 中的数值，不能直接理解为机械臂误差等于 0.0039 米。真实物理误差还取决于动作归一化之前对应的实际量程。

---

# 19. 阶段思考 D：类别 128 对应什么动作？

假设一个动作维度范围为：

$$
[-1,1]
$$

共有 256 个 Bin。

如果使用第 k 个 Bin 的中心还原：

$$
a_k
=
-1+
\left(
k+\frac12
\right)
\frac{2}{256}
$$

当：

$$
k=128
$$

得到：

$$
a_{128}
\approx
0.0039
$$

也就是非常接近 0。

这个问题检查的是是否真正理解：Action Token 不等于动作本身，而是对应一个连续区间，再解码回近似连续动作。

---

# 20. RT-1 的数据组织：从 Episode 到训练样本

RT-1 的成功不仅来自网络架构，还与机器人数据规模密切相关。

论文中的训练数据大致具有如下规模：约 13 台真实机器人、约 17 个月数据采集、约 13 万条机器人演示、700 多种任务。

这些数据使 RT-1 成为早期非常典型的 Large-scale Robot Learning 系统。

---

# 21. 一条机器人轨迹如何表示？

机器人数据集可以表示为：

$$
D
=
\{\tau_1,\tau_2,\ldots,\tau_N\}
$$

第 i 条轨迹：

$$
\tau_i
=
\left(
l_i,
\{(I_t,a_t)\}_{t=1}^{T_i}
\right)
$$

其中：

- lᵢ：该 Episode 对应的语言任务；
- Iₜ：时刻 t 的图像；
- aₜ：专家动作；
- Tᵢ：轨迹长度。

例如，一个 “Put the apple on the plate.” 任务可以包含从接近苹果、闭合夹爪、抬起、移动到盘子上方、下降到松开夹爪的完整轨迹。

实际数据中的 Action 不是“向苹果移动”这样的自然语言，而是对应机器人控制空间中的数值。

---

# 22. 从完整轨迹构造监督样本

对于某个时间点 t：

输入：

$$
x_t
=
(I_{t-5:t},l_i)
$$

监督目标：

$$
y_t=a_t^*
$$

于是：

$$
(I_{t-5:t},l_i)
\longrightarrow
a_t^*
$$

一条较长的机器人轨迹，可以从不同时间位置产生大量训练样本。

这说明 RT-1 的核心训练逻辑仍然属于：专家轨迹 → Observation-Action Pair → 监督学习 → 拟合专家策略，也就是 Behavior Cloning。

---

# 23. RT-1 的训练目标

如果第 j 个动作维度离散后的专家类别为：

$$
k_t^j
$$

模型学习：

$$
P_\theta
\left(
k_t^j
\mid
I_{t-5:t},l
\right)
$$

可以将整体目标理解为对动作 Token 进行分类预测。

一个简化的多维动作损失可以写成：

$$
\mathcal L
=
-\sum_{j=1}^{D}
\log
P_\theta
\left(
k_t^j
\mid
I_{t-5:t},l
\right)
$$

其中：

- D：动作维度；
- j：某一个动作维度；
- kₜʲ：对应维度的专家离散标签。

核心思想是：专家连续动作 → Action Discretization → Ground-truth Action Tokens → Transformer Prediction → Cross-Entropy。

---

# 24. RT-1 与 ACT、Diffusion Policy、Flow Matching 的区别

| 方法 | 典型动作表示 | 输出形式 | 典型训练思想 |
|---|---|---|---|
| 普通 BC | 连续动作 | 单步 Action | 连续回归 |
| RT-1 | 离散动作类别 | 主要为当前 Action | Action Token 分类 |
| ACT | 连续 Action Chunk | 多步动作序列 | CVAE + Behavior Cloning |
| Diffusion Policy | 连续动作序列 | Action Horizon | Iterative Denoising |
| Flow Matching Policy | 连续动作序列 | Action Horizon | Vector Field Learning |

这里最重要的不是记表格，而是形成一个研究视角：不同机器人策略之间的重要差异之一，就是它们如何 Representation Action，以及如何定义 Training Objective。

---

# 25. 阶段思考 E：RT-1 和 ACT 的根本区别只是 Transformer 不同吗？

### 解析

不是。

至少要区分两个维度。

第一是动作时间结构：RT-1 主要预测当前 Action，ACT 一次预测未来 Action Chunk。

第二是 Action Representation 与 Objective：RT-1 采用离散 Action Token + 分类目标，ACT 采用连续 Action Chunk + CVAE / Reconstruction-based Objective。

因此，只比较“它们都用了 Transformer”会遗漏机器人策略中非常重要的 Action Design。

---

# 26. 为什么 RT-1 能实现一定程度的泛化？

RT-1 的 Generalization 并不是某一个特殊模块单独产生的，而是多个因素共同作用：大量任务、统一语言条件、共享参数、统一视觉表示、统一动作表示。

其中最重要的思想之一是：不同机器人任务不再被完全独立地训练，而是在同一个模型内部共享 Representation 和 Parameters。

---

# 27. 环境泛化、对象泛化与任务组合泛化

## 27.1 环境泛化

训练：在桌子 A 上抓苹果。

测试：在桌子 B 上抓苹果。

任务本身相似，但背景、位置、光照等视觉因素变化。

## 27.2 对象泛化

训练：抓红色杯子。

测试：抓蓝色杯子。

需要将已经学习的 Manipulation Skill 迁移到新的对象实例。

## 27.3 组合泛化

训练中存在“把苹果放进盘子”以及“把杯子放进抽屉”，测试任务为“把苹果放进抽屉”。

如果模型能够完成任务，意味着它在一定程度上重新组合了已有经验：Apple Knowledge + Drawer Interaction + Pick-and-Place Skill。

这类能力是 Generalist Policy 很重要的一部分。

---

# 28. 为什么参数共享可能产生组合能力？

传统独立策略：

$$
\pi_{\theta_1},
\pi_{\theta_2},
\pi_{\theta_3}
$$

不同任务分别更新不同参数。

而 RT-1 使用：

$$
\pi_\theta(a\mid I,l)
$$

所有任务共同更新 θ。

多个任务会共同塑造同一套内部 Representation，于是模型可能学习出跨任务可复用的知识，例如什么是苹果、什么是抽屉、什么是抓取、什么是放置、什么是接近目标，然后在新任务中重新组合。

但参数共享只是产生泛化能力的重要条件之一，并不保证模型一定能够实现任意组合泛化。能力仍然受到数据覆盖范围、模型容量、任务相似度、视觉分布和动作分布等因素限制。

---

# 29. Generalist Policy 不等于 Universal Robot

“Generalist Robot Policy”并不意味着任何机器人、任何环境、任何语言任务都能完成。

更准确的理解是：相比传统单任务策略，RT-1 用一个共享模型覆盖了大量任务，并表现出一定程度的环境、对象和任务组合泛化。

因此应该理解为一个连续的发展谱系：Single-task Policy → Multi-task Policy → Generalist Policy → 更强的 Open-world Generalization。

RT-1 是向 Generalist Robot Policy 迈出的重要一步，而不是通用机器智能问题的终点。

---

# 30. 阶段思考 F：成功完成新组合任务说明什么？

### 问题

训练时见过“苹果 → 盘子”“杯子 → 抽屉”，测试时成功“苹果 → 抽屉”，能否据此证明 RT-1 已经理解了任意机器人任务？

### 解析

不能。这个结果主要说明模型具有一定的组合泛化能力。

阅读 Generalist Robot Policy 论文时，应特别关注：新任务到底有多新？新物体是否在训练中出现过？新环境是否只是视觉变化？动作技能是否实际上已经在训练集中存在？测试任务是重新组合，还是需要真正的新知识？

这是判断机器人“泛化”含金量的重要研究视角。

---

# 31. 大规模真实数据为什么重要？

RT-1 的一个核心思想是：Scale 本身可以改善机器人策略。

当数据规模扩大时，更多物体、更多位置、更多背景、更多机器人状态、更多操作阶段和更多任务组合都可能进入训练分布。

于是模型面对真实世界变化时，更有可能遇到与训练数据相似的状态。

这也是为什么 RT-1 的贡献不能只归结为 Transformer。如果只复制网络结构，却只有很少的数据，并不会自动得到 RT-1 所展示的大规模多任务能力。

---

# 32. 但大数据为什么仍不能解决 Distribution Shift？

Behavior Cloning 的核心问题之一是：

$$
d_{\pi_\theta}(s)
\neq
d_{\pi_E}(s)
$$

其中：

- πE：专家策略；
- πθ：学习到的策略；
- dπE：专家轨迹访问的状态分布；
- dπθ：部署时学习策略实际访问的状态分布。

训练数据主要来自专家。但真机部署时，一旦模型产生一个小错误，就可能出现“动作略偏 → 机器人进入专家很少访问的状态 → 预测更加不可靠 → 产生更大误差 → 继续偏离训练分布”的链条。

因此大规模数据的作用主要是扩大专家状态覆盖，降低遇到完全陌生状态的概率，但不能保证：

$$
d_{\pi_\theta}(s)
=
d_{\pi_E}(s)
$$

所以 Scale 可以缓解 Distribution Shift，但不等于从理论上消除了 Behavior Cloning 的 Distribution Shift。

---

# 33. RT-1 的主要局限

## 33.1 世界知识有限

RT-1 的主要能力来源是机器人数据。

考虑任务：

```text
Pick up the object that can be used to clean the table.
```

桌面上存在苹果、海绵、杯子。完成任务不仅需要识别物体，还需要常识：海绵可以用于清洁桌面。

这种知识未必能够从有限的机器人演示中充分学习。互联网规模的视觉语言模型，则可能已经通过大量图文数据建立 Object ↔ Language ↔ Semantic Knowledge 之间的联系。

这正是 RT-2 的核心动机之一。

## 33.2 Action Discretization 存在精度损失

离散化会引入 Continuous Action → Quantization → Discrete Bin → Approximate Continuous Action，因此存在量化误差。

对于精细装配、精密插入、小尺度接触任务，动作分辨率可能变得更加重要。

## 33.3 Behavior Cloning 的 Distribution Shift 仍然存在

RT-1 并没有改变“离线专家数据训练 → 学习策略部署”这一核心范式，因此依然存在 Compounding Error、Covariate Shift、Out-of-distribution State 等问题。

## 33.4 跨机器人泛化仍然受限

不同机器人可能拥有完全不同的 DOF、Kinematics、Action Space、Gripper、Base、Controller Interface。

例如 7-DOF 单臂、6-DOF 工业机械臂、双臂机器人、移动操作机器人，它们并不能仅因为执行同一个语言任务，就自动共享完全相同的 Action Representation。

因此 RT-1 的 Generalist 主要是在其机器人平台和训练分布范围内讨论，它并没有彻底解决 Cross-Embodiment Policy 的问题。

---

# 34. RT-1 在机器人控制系统的哪一层？

理解现代 VLA 时，必须始终区分：

```text
High-level Task Planning
        ↓
Learned Robot Policy / VLA
        ↓
Action / Trajectory Command
        ↓
Low-level Controller
        ↓
Motor / Actuator
```

RT-1 主要处在 Learned Robot Policy 这一层。

它根据 Vision + Language 决定机器人当前应该执行什么动作，但这并不意味着 Transformer 直接替代所有底层控制系统。

例如，机械臂最终的 Joint Servo、Motor Current Control、Position Loop、Velocity Loop 仍然可能由传统控制系统完成。

因此在研究 VLA 时，应避免把 Policy 和 Low-level Control 混为一谈。

---

# 35. 完整机器人实例：把苹果放进抽屉

任务：

```text
Put the apple in the drawer.
```

当前摄像头中存在 Apple、Cup、Drawer、Robot Gripper 和 Background。

## 第一步：语言条件

语言输入被编码成 Language Representation：

$$
e_l=f_{\text{text}}(l)
$$

它告诉策略目标对象是 Apple，目标位置是 Drawer。

## 第二步：视觉编码

最近 6 帧：

$$
I_{t-5:t}
$$

分别经过 EfficientNet：

$$
F_t=f_{\text{vision}}(I_t)
$$

得到空间视觉 Feature。

## 第三步：FiLM 条件化

语言表示产生：

$$
\gamma,\beta
$$

并调制视觉特征：

$$
F_t'
=
\gamma\odot F_t+\beta
$$

这样，与 Apple / Drawer / 当前任务相关的视觉信息能够更加直接影响后续策略。

## 第四步：TokenLearner 压缩

每一帧：

$$
F_t'
\rightarrow
8\ \text{tokens}
$$

6 帧：

$$
6\times8=48
$$

得到 48 Visual Tokens。

## 第五步：Transformer 推理

Transformer 根据视觉历史建立当前操作上下文，例如判断机械臂正在向 Apple 靠近，或者 Apple 已经被抓起、正在向 Drawer 移动。

## 第六步：预测 Action Tokens

假设当前需要向前移动、略微下降、调整姿态、保持夹爪打开，这些连续控制量经过 Action Discretization 后具有对应的 Bin，Transformer 预测各动作类别。

## 第七步：动作解码与执行

离散 Action Tokens 被转换回 Δx、Δy、Δz、Rotation、Gripper、Base 等机器人控制量，再交给底层机器人控制系统执行。

## 第八步：重新观察

机器人运动后，相机获得新图像。新的 6 帧 Observation Window 形成后，再次进行推理，于是得到闭环：Observation → Policy → Action → Environment → New Observation。

---

# 36. 容易混淆的概念与阶段纠偏

## 36.1 Transformer 与 LLM

Transformer 是一种神经网络架构；LLM 是使用大规模文本等数据和特定训练目标训练出的模型。

因此，用了 Transformer ≠ 自动成为 LLM ≠ 自动拥有世界知识。

## 36.2 Multi-task Policy 与 Generalist Policy

Multi-task 指同一模型执行多个任务；Generalist 不仅覆盖很多任务，还希望通过共享 Representation 获得一定对象、环境、任务组合泛化。

## 36.3 Generalist 与 Universal

Generalist 并不等于任何机器人、任何环境、任何任务。RT-1 的泛化仍然受到训练数据和硬件平台限制。

## 36.4 TokenLearner 与固定 Patch Selection

TokenLearner 学习如何从整个 Feature Map 聚合信息；固定 Patch Selection 则始终保留固定空间位置。两者不是同一个概念。

## 36.5 FiLM 与 Multimodal Token Concatenation

FiLM 是语言改变视觉 Feature；Token 拼接是语言和视觉分别成为 Token，再由 Transformer 交互。都可以实现条件化，但位置和机制不同。

## 36.6 Cross-Entropy 与 MSE

不能简单认为 Cross-Entropy > MSE。更准确的关系是：离散 Action → Classification → Cross-Entropy；连续 Action → Regression → MSE 等连续损失。Loss 与 Representation 是配套设计。

## 36.7 大规模数据与 Distribution Shift

错误理解是“数据足够多 → Distribution Shift 消失”。更准确的是“更多数据 → Training Distribution 覆盖更广 → OOD 概率下降”，但学习策略仍可能进入专家从未访问过的状态。

---

# 37. RT-1 为什么自然引出了 RT-2？

RT-1 已经证明：

$$
\text{Vision}
+
\text{Language}
\longrightarrow
\text{Robot Action}
$$

可以在大规模真实机器人数据上训练。

但仍然存在一个核心瓶颈：机器人数据本身包含的语义知识极其有限。

互联网视觉语言数据规模远大于机器人演示数据，因此自然产生一个问题：能否让机器人策略继承已经由大规模 VLM 学到的视觉语义知识和世界知识？

RT-1 的学习逻辑可以概括为：

$$
\text{Robot Data}
\longrightarrow
\text{Robot Policy}
$$

而 RT-2 开始尝试：

$$
\text{Internet-scale VLM Knowledge}
+
\text{Robot Data}
\longrightarrow
\text{Vision-Language-Action Model}
$$

因此，从 VLA 历史发展来看，RT-1 的重点是 Scale Robot Learning，而 RT-2 的重点是 Transfer VLM Knowledge into Robot Action。

---

# 38. RT-1 在 VLA 知识体系中的位置

```text
Robot Control
      ↓
Behavior Cloning
      ↓
Multi-task Robot Learning
      ↓
Language-conditioned Policy
      ↓
RT-1
      ↓
利用预训练 VLM
      ↓
RT-2
      ↓
Modern VLA
      ↓
OpenVLA / π0 / π0.5 等
      ↓
Fine-tuning / Agent / RL
```

从知识功能来看，RT-1 把 Transformer、Vision、Language、Behavior Cloning、Action Representation、Real Robot Data 等概念第一次真正连接起来。

因此它是从“学习现代深度学习组件”进入“学习完整 VLA 系统”的重要节点。

---

# 39. 研究视角：阅读 RT-1 类论文时真正应该看什么？

研究机器人策略时，不应只问“用了什么 Backbone？”，更重要的是沿着完整系统检查：

### 数据

数据有多大？来自多少任务？是否为真实机器人数据？测试任务和训练任务到底差多少？

### Observation

输入是单帧还是多帧？有没有机器人 proprioception？语言如何进入模型？

### Representation

视觉最终变成多少 Token？是否进行 Token Compression？不同模态在哪里融合？

### Action

输出是 Joint Position、End-effector Delta、Discrete Token 还是 Action Chunk？Action Space 如何归一化？是否离散化？

### Objective

模型究竟在优化 MSE、Cross-Entropy、Diffusion Loss 还是 Flow Matching Loss？

### Generalization

所谓 Generalization 是新背景、新物体、新位置、新任务组合，还是完全新的语义知识？

### Deployment

策略产生的是高层技能还是低层动作？底层 Controller 是否仍然独立存在？

这套分析框架不仅适用于 RT-1，也适用于后续 RT-2、OpenVLA、π0、π0.5 等模型。

---

# 40. 综合思考

## 问题 1

RT-1 最核心的创新究竟是 Transformer，还是 Large-scale Multi-task Robot Learning？

### 解析

Transformer 很重要，但如果只把 RT-1 理解成“机器人用了 Transformer”，会严重低估它的意义。

RT-1 更重要的研究价值在于证明：大规模真实机器人数据 + 共享语言条件策略 + 统一模型，能够覆盖大量不同任务，并产生一定泛化能力。因此 Architecture 和 Scale 必须一起理解。

## 问题 2

为什么 RT-1 不仅仅是“分类版 BC”？

### 解析

从损失函数角度看，它确实仍然是 Behavior Cloning。但完整系统同时包含语言条件、多任务数据、视觉历史、Task-conditioned Visual Feature、Token Compression、Transformer Sequence Modeling、Action Tokenization。

因此“分类版 BC”只描述了训练目标的一部分，不能代表整个系统创新。

## 问题 3

RT-1 能否解决机器人所有的 OOD 问题？

### 解析

不能。Scale 可以扩大训练覆盖，但不能保证部署策略永远停留在训练状态分布中。

因此：

$$
d_{\pi_\theta}
\neq
d_{\pi_E}
$$

仍然是 Behavior Cloning 系统的重要问题。

## 问题 4

为什么 RT-1 已经有 Language Input，却仍然需要 RT-2？

### 解析

“输入 Language”与“拥有大规模语言和视觉知识”完全不是一个概念。

RT-1 中 Language 主要作为 Task Condition；RT-2 希望进一步利用预训练 VLM 中已经存在的语义知识，帮助机器人理解新的对象、概念和任务。

因此 RT-2 解决的不是“RT-1 没有语言输入”，而是 RT-1 的语言与视觉语义知识仍然主要受到机器人数据规模的限制。

---

# 41. 本章总结：应建立怎样的 RT-1 心智模型？

第一，RT-1 本质上仍然属于模仿学习。它没有抛弃 Behavior Cloning，而是把 BC 扩展到了大规模、多任务、语言条件化、Transformer Policy 的机器人学习场景。

第二，RT-1 的完整信息流是：最近 6 帧 RGB + Language → EfficientNet → FiLM → Task-conditioned Visual Features → TokenLearner → 每帧 8 Token → 48 Visual Tokens → Transformer → Discrete Action Tokens → Robot Action。

第三，Action Representation 是理解 RT-1 的关键。RT-1 将连续动作离散化为 256 个 Bin，使机器人动作可以转化为分类预测问题，并使用 Cross-Entropy 训练。

第四，RT-1 的 Generalist 能力来自大规模多任务数据 + 参数共享 + 语言条件 + 统一视觉与动作表示，而不是 Transformer 本身自动带来的“智能”。

第五，大规模数据改善了泛化，却没有消除 Behavior Cloning 的根本问题。Distribution Shift、Compounding Error、跨机器人 Action Space、世界知识不足仍然存在。

第六，RT-1 最重要的历史作用是完成了一次范式过渡：Single-task Robot Policy → Large-scale Generalist Robot Policy。它留下的“机器人数据中的语义知识仍然有限”这一问题，直接推动了 RT-2 与后续现代 VLA 的发展。

---

# 42. 关键词

**RT-1 — Robotics Transformer 1 — 使用大规模真实机器人数据训练的语言条件多任务机器人策略。**

**Generalist Robot Policy — 通用机器人策略 — 使用共享模型覆盖大量任务，并具有一定泛化能力。**

**Language-conditioned Policy — 语言条件策略 — 根据自然语言任务指令决定当前机器人行为。**

**EfficientNet — 高效卷积视觉网络 — RT-1 中用于提取 RGB 图像视觉特征。**

**FiLM — Feature-wise Linear Modulation — 使用语言条件对视觉 Feature 进行缩放和偏移调制。**

**TokenLearner — Token 学习压缩模块 — 从高维空间 Feature Map 中学习少量高信息量 Token。**

**Visual History — 视觉历史 — 使用最近多帧图像提供一定的时序上下文。**

**Action Representation — 动作表示 — 将真实机器人控制量转换为模型能够学习和生成的表示形式。**

**Action Discretization — 动作离散化 — 将连续动作空间划分为有限数量的区间。**

**Action Token — 动作 Token — 离散化后的机器人动作类别。**

**Behavior Cloning — 行为克隆 — 使用专家 Observation-Action 数据进行监督式策略学习。**

**Cross-Entropy — 交叉熵 — RT-1 离散动作分类中使用的训练目标。**

**Quantization Error — 量化误差 — 连续动作离散化后产生的近似误差。**

**Parameter Sharing — 参数共享 — 多种机器人任务共同使用同一套模型参数。**

**Compositional Generalization — 组合泛化 — 将训练中已有的对象、目标和技能重新组合完成新任务。**

**Distribution Shift — 分布偏移 — 部署策略访问的状态分布与专家训练数据分布不一致。**

**RT-2 — Robotics Transformer 2 — 在 RT-1 思路基础上进一步利用大规模视觉语言模型知识进行机器人控制的后续路线。**
