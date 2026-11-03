# 11｜Diffusion Policy：从条件生成到滚动时域动作控制

## 1. 本章导言

Diffusion Policy 是现代机器人学习中一个非常重要的生成式策略（Generative Policy）范式。它将扩散模型从图像生成问题迁移到机器人动作生成问题：模型不再从随机噪声生成图片，而是在当前机器人观测的条件约束下，从随机噪声逐步生成一段未来动作轨迹。

在整套 VLA 学习路线中，本章处于一个明显的承上启下位置：

```text
Behavioral Cloning
↓
Action Representation
↓
Action Chunking
↓
ACT
↓
Diffusion 基础
↓
Diffusion Policy
↓
Flow Matching
↓
π0 等生成式 VLA
```

本章真正需要建立的不是“Diffusion Policy 用了扩散模型”这一句结论，而是下面这个完整心智模型：

```text
机器人当前 observation
        ↓
作为条件信息 condition
        ↓
随机 action trajectory
        ↓
多步条件去噪
        ↓
具有时间一致性的 action chunk
        ↓
只执行其中一部分
        ↓
获取新 observation
        ↓
重新生成动作
```

也就是说，Diffusion Policy 同时连接了三个此前已经学习过的重要主题：

1. 模仿学习中的动作分布建模；
2. Action Chunking 中的时间序列预测；
3. 控制系统中的 Receding-Horizon Control。

---

# 2. 问题从哪里来：为什么机器人策略需要生成式建模？

## 2.1 普通 Behavioral Cloning 的基本形式

最简单的连续动作 Behavioral Cloning 可以写成：

$$
\hat A=f_\theta(o)
$$

其中：

- o：当前机器人 observation；
- A：动作或未来动作序列；
- fθ：参数为 θ 的策略网络；
- Â：模型预测出的动作。

如果使用均方误差进行监督训练：

$$
L_{\mathrm{BC}}
=
\left\|
f_\theta(o)-A^*
\right\|_2^2
$$

其中 A* 是专家演示中的真实动作。

这种方法隐含了一个很强的假设：

> 对给定 observation，存在一个比较明确的“正确动作”。

但机器人 manipulation 经常并不是这样。

---

## 2.2 机器人动作为什么可能具有 Multimodality？

考虑一个机械臂从当前位置移动到目标物体附近，中间存在障碍物。

机器人可以：

```text
方案 A：从障碍物左侧绕过

方案 B：从障碍物右侧绕过
```

两种轨迹都可能是完全正确的专家策略。

于是给定相同 observation，合理动作轨迹的条件分布可能不是单峰的：

$$
p(A\mid o)
$$

可能包含多个高概率区域。

如果训练数据中一半演示从左绕、一半从右绕，而模型被要求直接通过 MSE 给出一个唯一答案，就可能出现所谓的“平均动作”。

概念上可以表示为：

```text
正确模式 A
      ↘
       平均结果
      ↗
正确模式 B
```

而这个平均结果本身可能根本不是有效轨迹。

例如：

```text
左绕障碍物：正确

右绕障碍物：正确

平均：
直接朝障碍物走
```

这就是为什么动作分布建模很重要。

---

## 2.3 Diffusion Policy 的基本思想

Diffusion Policy 不再简单学习一个确定映射：

$$
A=f_\theta(o)
$$

而希望表示：

$$
p(A\mid o)
$$

即：

> 在当前 observation 条件下，哪些 action trajectory 是合理的？

模型可以从这个条件分布中生成一条完整、一致的动作轨迹。

因此，Diffusion Policy 的关键优势不是“它输出了 Action Chunk”。

ACT 也可以输出 Action Chunk。

真正的区别是：

> Diffusion Policy 使用生成式方法建模条件动作分布。

---

# 3. 核心概念：真正被 Diffusion 的是什么？

这是理解 Diffusion Policy 时最容易出现的第一个混淆。

假设当前机器人观测为：

$$
o_t
$$

未来 H 步动作组成：

$$
A_t
=
[
a_t,
a_{t+1},
\ldots,
a_{t+H-1}
]
$$

如果每一步 action 维度为 dₐ，那么：

$$
A_t
\in
\mathbb R^{H\times d_a}
$$

例如：

```text
H = 20
dₐ = 7
```

则：

```text
Action trajectory shape = 20 × 7
```

总元素数量是：

```text
20 × 7 = 140
```

但 tensor shape 仍然应该写为：

```text
20 × 7
```

而不能把 shape 写成 140。

Diffusion Policy 中真正被不断加噪、去噪的是：

> **Action trajectory。**

而不是 observation。

其最核心的数据流为：

```text
Observation
      +
Noisy Action Trajectory
      ↓
Conditional Diffusion Model
      ↓
Denoising
      ↓
Action Chunk
```

---

## 3.1 Observation 的角色：Condition，而不是扩散对象

机器人 observation 可以包含：

```text
RGB 图像
机器人关节状态
末端位姿
夹爪状态
历史 observation
其他 proprioception
```

可抽象为：

$$
o_t=(I_t,s_t)
$$

其中：

- Iₜ：视觉输入；
- sₜ：机器人自身状态。

例如：

$$
s_t
=
[
q_1,\ldots,q_7,
\dot q_1,\ldots,\dot q_7,
g
]
$$

其中：

- q：关节位置；
- q̇：关节速度；
- g：gripper state。

Observation 并不会被不断加入 Gaussian noise。

它的作用是：

> 告诉生成模型“在当前环境和机器人状态下，应该生成什么样的动作”。

因此更准确的理解不是：

```text
Observation 是模型的一个输入项
```

而是：

```text
Observation 是 action generation 的 conditioning information。
```

这是一个重要区别。

---

## 阶段思考 A：扩散对象与条件信息

### 问题 1

Diffusion Policy 中真正被不断加噪、去噪的是 observation 还是 action？

### 解析

答案是 action，更准确地说，是未来一段 action trajectory。

Observation 负责提供条件信息。

正确数据流是：

```text
Observation
      ↓
Condition

Random / Noisy Action
      ↓
Denoising
      ↓
Action
```

而不是：

```text
Image
↓
加噪
↓
去噪
↓
Action
```

后者错误地把视觉 observation 当成了 diffusion 对象。

---

### 问题 2

如果：

```text
H = 20
dₐ = 7
```

action trajectory 的 tensor shape 是多少？

### 解析

正确答案是：

```text
20 × 7
```

140 是张量中标量元素的数量，而不是其二维 shape。

这一类区别在阅读机器人代码时非常重要：

```text
shape ≠ number of elements
```

---

### 问题 3

为什么说 Diffusion Policy 对 multimodal action distribution 更自然？

### 解析

只回答“因为它不是单峰分布”方向正确，但还不够完整。

更准确的因果链是：

```text
同一 observation
↓
可能存在多个正确 trajectory
↓
确定性 MSE regression 需要压成一个输出
↓
可能得到多个模式之间的平均
↓
Diffusion 建模 p(A | o)
↓
可以从多个合理 mode 中生成一条完整 trajectory
```

关键不是“Diffusion 比较灵活”，而是：

> 它的建模目标本身允许条件动作分布具有多个 mode。

---

# 4. Diffusion Policy 生成的是 Action Chunk

Diffusion Policy 通常并不只预测下一时刻单个动作。

它生成未来一段动作：

$$
A_t
=
[
a_t,
a_{t+1},
\ldots,
a_{t+H-1}
]
$$

例如：

```text
H = 16
dₐ = 7
```

则：

```text
Aₜ shape = 16 × 7
```

扩散过程操作的也是整个：

```text
16 × 7
```

action trajectory。

因此可以先建立一个简化理解：

> Diffusion Policy = Action Chunking + Conditional Generative Modeling。

但这只是直觉表达，不代表所有使用 Action Chunking 的方法都是 Diffusion Policy。

ACT 同样使用 Action Chunking，但生成机制完全不同。

---

# 5. Diffusion Policy 的数学机制

## 5.1 训练数据

来自专家 demonstration 的数据可以抽象成：

$$
(o_t,A_t^*)
$$

其中：

- oₜ：当前或一小段历史 observation；
- Aₜ*：专家未来动作序列。

例如：

$$
A_t^*
=
[
a_t^*,
a_{t+1}^*,
\ldots,
a_{t+H-1}^*
]
$$

这就是训练时的 clean action trajectory。

---

# 6. Forward Process：给专家动作加噪

训练过程中，首先随机采样一个 diffusion timestep：

$$
k
\sim
\{1,\ldots,K\}
$$

其中 K 表示最大噪声等级。

然后采样 Gaussian noise：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

噪声 ε 与 action trajectory 具有相同 shape。

如果：

```text
A* shape = 16 × 7
```

那么：

```text
ε shape = 16 × 7
```

之后按照噪声调度将专家动作污染：

$$
A^k
=
\sqrt{\bar\alpha_k}
A^*
+
\sqrt{1-\bar\alpha_k}
\epsilon
$$

其中：

- A*：真实专家 action trajectory；
- Aᵏ：第 k 个 noise level 下的 noisy action；
- ε：人为采样的 Gaussian noise；
- ᾱₖ：由 noise schedule 决定的系数；
- k：diffusion timestep。

直觉上：

```text
k 较小
↓
Aᵏ 仍然很接近专家动作

k 较大
↓
Aᵏ 中噪声占比越来越高
```

最终在极高 noise level 下，Aᵏ 可以接近纯随机噪声。

---

# 7. 网络到底学习什么？

Diffusion Policy 的 denoiser 可以写成：

$$
\hat\epsilon
=
\epsilon_\theta(A^k,k,o)
$$

模型输入包括：

```text
noisy action trajectory Aᵏ
diffusion timestep k
observation o
```

输出：

```text
predicted noise ε̂
```

常见训练损失为：

$$
L
=
\left\|
\epsilon
-
\epsilon_\theta(A^k,k,o)
\right\|_2^2
$$

这里非常值得注意：

> 模型通常不是直接预测 clean action，而是在预测当初加入的 noise。

---

## 7.1 为什么可以知道真实 noise？

因为训练时的 ε 并不是未知环境噪声。

它是训练程序自己采样出来的：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

因此：

```text
程序自己生成 ε
↓
使用 ε 污染专家 action
↓
得到 Aᵏ
↓
模型预测 ε̂
↓
可以直接比较 ε 和 ε̂
```

所以真实 noise 就是天然的监督信号。

---

# 8. 一个容易混淆的问题：Diffusion Policy 也用 MSE，为什么不会退化成普通 MSE BC？

普通 BC 可以直接优化：

$$
L_{\mathrm{BC}}
=
\left\|
A^*-\hat A
\right\|_2^2
$$

其目标是：

```text
observation
↓
直接得到唯一 action
```

而 Diffusion Policy 优化：

$$
L_{\mathrm{diff}}
=
\left\|
\epsilon-\hat\epsilon
\right\|_2^2
$$

其目标是：

```text
noisy action
+
observation
+
noise level
↓
学习去噪方向
↓
最终形成一个条件生成过程
```

两者虽然数学上都可能出现 MSE，但 MSE 所约束的变量完全不同。

因此：

```text
使用 MSE
≠
一定是 deterministic action regression
```

判断一个方法是不是普通 MSE BC，不能只看 loss 形式，而要看：

> 模型究竟在预测什么随机变量，以及整个 inference process 是什么。

---

## 阶段思考 B：Training 机制

### 问题 1

训练最开始使用的是专家 action 还是 random noise？

### 解析

训练从专家 action A* 开始。

随后人为加入 Gaussian noise 得到 Aᵏ。

不要把训练和推理混淆：

```text
Training：
expert action → 加噪

Inference：
pure noise → 去噪
```

---

### 问题 2

训练时模型通常预测 action 还是 noise？

### 解析

本章讨论的标准形式中预测 noise：

$$
\epsilon_\theta(A^k,k,o)
$$

监督目标来自训练程序自己采样的 ε。

---

### 问题 3

一次 training step 是否需要真的执行：

```text
K → K-1 → ... → 0
```

整条 reverse process？

### 解析

通常不需要。

一次 training step 可以随机采样一个 k，直接构造对应 Aᵏ：

```text
A*
+
随机 ε
+
随机 k
↓
直接得到 Aᵏ
↓
预测 ε
```

于是网络在大量 batch 中逐渐学会处理不同 noise level。

这使训练不需要每次都显式跑完整的 K 次反向去噪。

---

# 9. Inference：为什么可以从纯随机噪声得到动作？

部署时已经没有专家动作 A*。

只有机器人当前 observation：

$$
o_t
$$

因此先采样：

$$
A^K
\sim
\mathcal N(0,I)
$$

假设 action chunk shape 为：

```text
16 × 7
```

那么就随机生成一个：

```text
16 × 7
```

的 noise tensor。

之后不断使用同一个 denoiser：

$$
\epsilon_\theta(A^k,k,o_t)
$$

逐步执行：

$$
A^K
\rightarrow
A^{K-1}
\rightarrow
A^{K-2}
\rightarrow
\cdots
\rightarrow
A^0
$$

最终：

$$
A^0
=
[
a_t,
a_{t+1},
\ldots,
a_{t+H-1}
]
$$

成为可执行的 action chunk。

---

## 9.1 为什么训练只学随机 k，推理却可以连续执行所有 k？

因为网络在训练期间会看到各种随机 noise level。

例如：

```text
batch 1：k = 5
batch 2：k = 81
batch 3：k = 43
batch 4：k = 97
batch 5：k = 20
...
```

因此同一个网络逐渐学会：

```text
在高噪声阶段怎样修正
在中噪声阶段怎样修正
在低噪声阶段怎样微调
```

于是 inference 时就可以连续使用它：

```text
K
↓
K-1
↓
K-2
↓
...
↓
0
```

这里可以形成一个很有用的直觉：

> 训练是在随机抽取不同“去噪难度”的局部问题进行学习；推理则把这些局部去噪能力串联起来。

---

# 10. Observation 为什么会改变最终生成轨迹？

假设随机初始化得到完全相同的 noisy action：

$$
A^K
$$

但 observation 不同。

场景一：

```text
杯子在左侧
```

场景二：

```text
杯子在右侧
```

那么：

$$
\epsilon_\theta(A^k,k,o_{\mathrm{left}})
$$

与：

$$
\epsilon_\theta(A^k,k,o_{\mathrm{right}})
$$

应当产生不同修正。

最终得到：

```text
左侧 observation
→ 向左接近杯子

右侧 observation
→ 向右接近杯子
```

因此模型学习的不是：

$$
p(A)
$$

而是：

$$
p(A\mid o)
$$

这就是 Conditional Diffusion 的机器人版本。

---

# 11. Training 与 Inference 的完整对应

| 环节 | Training | Inference |
|---|---|---|
| 是否有专家 action | 有 | 没有 |
| 初始 action | 专家 trajectory | Random noise |
| 是否主动加噪 | 是 | 否 |
| Diffusion timestep | 随机抽一个 k | 从高噪声逐步走向低噪声 |
| 网络预测 | Noise | Noise |
| 是否计算 loss | 是 | 否 |
| 是否更新参数 | 是 | 否 |
| 最终目的 | 学会 conditional denoising | 生成 action trajectory |

最核心的关系是：

```text
训练：
把正确动作破坏掉
↓
学习如何恢复

推理：
没有正确动作
↓
直接从彻底破坏状态开始
↓
通过学到的恢复能力生成动作
```

---

# 12. Observation Horizon、Prediction Horizon 与 Execution Horizon

Diffusion Policy 不是只涉及 diffusion，它同时也是一个时序机器人策略。

因此必须区分三个 horizon。

---

## 12.1 Observation Horizon

记作 Tₒ。

表示：

> 一次 policy inference 使用多少个历史 observation。

例如：

```text
Tₒ = 2
```

输入可能是：

$$
[o_{t-1},o_t]
$$

为什么需要多个 observation？

单张静态图像经常无法获得动态趋势。

例如只看到：

```text
杯子位于画面中央
```

无法判断它是：

```text
静止
向左运动
向右运动
```

如果有连续 observation：

```text
oₜ₋₁
oₜ
```

模型就获得了一定的时间动态信息。

---

## 12.2 Prediction Horizon

记作 Tₚ。

表示：

> 模型一次生成多长的未来 action trajectory。

例如：

```text
Tₚ = 16
```

则：

$$
A_t
=
[
a_t,
a_{t+1},
\ldots,
a_{t+15}
]
$$

如果 dₐ = 7：

```text
Action trajectory shape = 16 × 7
```

Diffusion process 就是在这个完整 action trajectory 上运行。

---

## 12.3 Action Horizon / Execution Horizon

记作 Tₐ。

表示：

> 每次真正发送给机器人并执行多少个预测动作。

例如：

```text
Tₚ = 16
Tₐ = 8
```

意味着：

```text
模型预测未来 16 步
↓
只执行前 8 步
↓
重新观察
↓
再次预测未来 16 步
```

因此：

```text
Prediction Horizon
≠
Execution Horizon
```

这是非常重要的区别。

---

# 13. 为什么预测 16 步却只执行 8 步？

如果机器人在时刻 t 得到：

$$
[a_t,a_{t+1},\ldots,a_{t+15}]
$$

然后真的把全部 16 步一次执行完，那么后半段动作始终基于时刻 t 的 observation。

但执行过程中可能发生：

```text
机械臂存在跟踪误差
物体被碰动
目标位置改变
夹爪接触产生偏差
外界发生扰动
策略本身前几步产生误差
```

因此越靠后的预测越可能变成：

> stale action。

更合理的执行方式是：

```text
Observe
↓
Generate long chunk
↓
Execute short prefix
↓
Observe again
↓
Regenerate
```

这就是 Receding-Horizon Control。

---

## 13.1 一个具体时间例子

设：

```text
Tₒ = 2
Tₚ = 16
Tₐ = 8
```

第一次输入：

$$
[o_{-1},o_0]
$$

预测：

$$
[a_0,a_1,\ldots,a_{15}]
$$

实际只执行：

$$
[a_0,\ldots,a_7]
$$

到 t = 8：

$$
[o_7,o_8]
$$

重新进入 policy。

得到新预测：

$$
[a_8',a_9',\ldots,a_{23}']
$$

这里：

$$
a_8'
$$

并不需要等于旧预测中的：

$$
a_8
$$

因为机器人此时已经得到了新的 observation。

这种“不断重新预测未来”的行为，才真正形成 closed-loop control。

---

## 阶段思考 C：Horizon 与闭环

### 问题

如果模型预测未来 16 步动作，为什么不直接把全部 16 步执行完？

### 解析

完整因果链应是：

```text
预测基于当前 observation
↓
执行需要时间
↓
环境和机器人状态不断变化
↓
越靠后的动作依赖越旧的信息
↓
旧 chunk 与真实状态逐渐失配
↓
需要提前重新观察与重规划
```

因此 prediction horizon 可以比较长，以学习完整动作结构；execution horizon 可以更短，以提高反馈修正能力。

---

# 14. 与传统 MPC 的联系

对于有控制背景的读者，Diffusion Policy 的执行逻辑与 Model Predictive Control 非常容易建立联系。

传统 MPC：

```text
当前状态
↓
预测未来
↓
求未来控制序列
↓
执行前一部分
↓
获得新状态
↓
重新优化
```

Diffusion Policy：

```text
当前 observation
↓
生成未来 action trajectory
↓
执行前一部分
↓
获得新 observation
↓
重新生成
```

共同思想是：

> 预测未来，但不盲目把整个未来计划一次执行到底。

区别在于未来控制序列是如何获得的。

传统 MPC 通常依赖：

```text
显式动力学模型
+
在线优化
```

Diffusion Policy 依赖：

```text
专家 demonstration
+
学习得到的条件生成模型
```

因此可以把它理解为一种：

> learned receding-horizon visuomotor policy。

但不能因此认为 Diffusion Policy 就是 MPC。

它们的优化机制、模型假设和训练方式仍然完全不同。

---

# 15. Diffusion Policy 的网络结构

Diffusion Policy 可以抽象成三个部分：

```text
Observation Encoder
+
Action Denoiser
+
Diffusion Process
```

整体数据流为：

```text
Camera / Robot State
        ↓
Observation Encoder
        ↓
Condition c
        ↓
Noisy Action Aᵏ
+
Timestep k
        ↓
Action Denoiser
        ↓
Predicted Noise ε̂
        ↓
Reverse Update
        ↓
Aᵏ⁻¹
```

其中 observation encoder 可以写作：

$$
c=f_{\mathrm{obs}}(o)
$$

随后 denoiser：

$$
\hat\epsilon
=
\epsilon_\theta(A^k,k,c)
$$

---

# 16. 为什么 timestep k 必须输入网络？

假设网络只看到 Aᵏ，却不知道 k。

那么它无法区分：

```text
当前 trajectory 几乎是纯 noise

还是

当前 trajectory 已经非常接近正确动作
```

这两种情况下应该采取的修正幅度显然不同。

因此网络需要知道：

```text
当前处于哪个 noise level
```

通常将 k 转换成 timestep embedding，再与其他 feature 一起输入模型。

于是网络学习的并不是一个固定去噪操作，而是：

> 在不同 noise level 下应该怎样修正 action trajectory。

---

# 17. Action Denoiser 一定是某种固定网络吗？

不是。

Diffusion 是：

> 一种生成机制。

它不等于某一种固定 neural network backbone。

只要网络能够实现：

$$
(A^k,k,o)
\rightarrow
\hat\epsilon
$$

就可以作为 denoiser。

可以使用适合时序动作建模的结构，例如：

```text
Temporal Convolution
1D Convolution
Transformer
```

因此：

```text
Diffusion Policy
≠
某一个固定网络结构
```

即使某个 Diffusion Policy 使用 Transformer denoiser，它仍然和 ACT 存在本质区别，因为二者的 action generation process 不同。

---

# 18. 完整机器人实例：Pick-and-Place

考虑一个机械臂任务：

> 将桌上的盒子抓起并放入目标篮子。

---

## 18.1 Observation

当前 observation 可以包括：

```text
RGB Camera
+
Robot Joint State
+
Gripper State
```

形式上：

$$
o_t=(I_t,s_t)
$$

Visual Encoder 从图像中获得：

```text
物体位置
目标区域
机械臂与物体相对关系
场景语义
```

Robot State 提供：

```text
当前关节位置
夹爪状态
机械臂 configuration
```

---

## 18.2 Expert Action Chunk

训练数据中的专家未来动作可能表示为：

$$
A_t^*
=
[
a_t^*,
a_{t+1}^*,
\ldots,
a_{t+15}^*
]
$$

这些连续 action 在语义上可能对应：

```text
向盒子靠近
↓
调整末端姿态
↓
下降
↓
闭合夹爪
↓
上提
↓
向目标篮移动
```

模型实际并不生成这些文字，而是在生成连续控制量。

例如每一步 action 可以是：

```text
Δx
Δy
Δz
Δroll
Δpitch
Δyaw
gripper
```

于是：

```text
dₐ = 7
```

---

## 18.3 Training

首先从真实 demonstration 取得：

```text
Observation oₜ
Expert Action A*
```

采样：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

随机选择 k：

$$
k\sim\{1,\ldots,K\}
$$

构造：

$$
A^k
=
\sqrt{\bar\alpha_k}A^*
+
\sqrt{1-\bar\alpha_k}\epsilon
$$

网络输入：

```text
Camera feature
Robot state feature
Noisy action trajectory
Timestep
```

输出：

```text
Predicted noise
```

优化：

$$
L
=
\|\epsilon-\hat\epsilon\|_2^2
$$

---

## 18.4 Inference

真实部署时：

```text
没有 expert action
```

因此初始化：

$$
A^K
\sim
\mathcal N(0,I)
$$

例如一开始可能是：

```text
step 1    随机数字
step 2    随机数字
step 3    随机数字
...
step 16   随机数字
```

经过多轮 conditional denoising：

```text
Random trajectory
↓
出现粗略运动方向
↓
形成接近目标的结构
↓
进一步修正姿态与夹爪动作
↓
得到完整可执行 action chunk
```

最终：

$$
A^0
=
[
a_t,\ldots,a_{t+15}
]
$$

机器人只执行前若干步，再重新观测。

这就形成：

```text
Perception
↓
Action Generation
↓
Partial Execution
↓
New Perception
↓
Replanning
```

完整闭环。

---

# 19. 为什么生成完整 trajectory，而不是独立生成每一个 action？

机械臂操作具有明显 temporal correlation。

例如抓取动作：

```text
接近
↓
调整姿态
↓
下降
↓
夹紧
↓
上提
```

这些动作不能彼此独立。

如果分别独立生成：

$$
p(a_t\mid o)
$$

$$
p(a_{t+1}\mid o)
$$

$$
p(a_{t+2}\mid o)
$$

就可能丢失动作间的整体协调性。

Diffusion Policy 更自然地建模：

$$
p(
a_t,
a_{t+1},
\ldots,
a_{t+H-1}
\mid
o
)
$$

即：

> 把整个 trajectory 当作一个联合随机变量。

因此模型不仅在学习“每一步做什么”，还在学习：

> 多步 action 之间应该怎样彼此协调。

---

# 20. ACT 与 Diffusion Policy：真正的核心差异

ACT 和 Diffusion Policy 都可以生成 Action Chunk。

因此不能把二者区别简单理解成：

```text
ACT：单步

Diffusion：多步
```

这是错误的。

真正区别在生成机制。

---

## 20.1 ACT：Direct Prediction

ACT 可以概括为：

$$
A=f_\theta(o)
$$

数据流：

```text
Observation
↓
Transformer
↓
Action Chunk
```

一次主要 forward 就直接输出 action trajectory。

核心是：

> direct action chunk prediction。

---

## 20.2 Diffusion Policy：Iterative Generative Prediction

Diffusion Policy：

$$
A^K\sim\mathcal N(0,I)
$$

然后：

$$
A^K
\rightarrow
A^{K-1}
\rightarrow
\cdots
\rightarrow
A^0
$$

数据流：

```text
Observation
      ↓
Condition

Random Action Noise
      ↓
Denoise
      ↓
Denoise
      ↓
...
      ↓
Action Chunk
```

核心是：

> iterative conditional generation。

---

## 20.3 一个重要的表达修正

可以说：

```text
ACT 是顺向直接预测
Diffusion Policy 是条件去噪生成
```

但“顺向推理”这个说法还不够精确。

因为任何 neural network inference 都可以称为 forward inference。

真正应当抓住的是：

```text
ACT：
一次直接映射

Diffusion Policy：
多步迭代生成
```

这才是二者最稳定的区分标准。

---

# 21. ACT vs Diffusion Policy 对比

| 维度 | ACT | Diffusion Policy |
|---|---|---|
| 基本任务 | Action Chunk Prediction | Action Chunk Generation |
| 输入 | Observation | Observation + Noisy Action |
| 输出 | Action Chunk | Action Chunk |
| 生成方式 | Direct Prediction | Iterative Denoising |
| 推理起点 | Observation representation | Random Action Noise |
| Multimodality | 相对受直接回归机制限制 | 生成式建模是核心优势 |
| 时间建模 | Action Chunk | Action Trajectory Distribution |
| 训练形式 | 模仿学习 / reconstruction 类目标 | Noise Prediction |
| 推理网络调用 | 通常较少 | 通常需要多次 |
| Receding Horizon | 可以使用 | 可以使用 |
| 优势 | 简洁、高效 | 多模态、轨迹级生成能力强 |
| 主要代价 | 分布表达能力受架构与目标限制 | 推理成本较高 |

---

# 22. Diffusion Policy 的主要优势

## 22.1 Multimodal Action Distribution

模型学习：

$$
p(A\mid o)
$$

而不是只学习一个唯一映射。

因此特别适合：

```text
多种抓取方式
多种绕障路径
不同有效姿态
复杂 contact-rich manipulation
```

---

## 22.2 Trajectory-Level Modeling

生成对象是：

$$
A
\in
\mathbb R^{H\times d_a}
$$

而不是孤立 action。

因此可以学习完整时间结构。

---

## 22.3 Temporal Consistency

由于整个 trajectory 一起建模，动作之间可以保持较强的时间相关性。

这对于 manipulation 中的：

```text
approach
grasp
lift
place
```

等连续阶段很重要。

---

## 22.4 与 Action Chunking 天然兼容

模型直接生成一段未来动作。

因此很容易和：

```text
Action Chunk
+
Receding Horizon
```

结合。

---

# 23. Diffusion Policy 的主要局限

## 23.1 推理速度

最大的工程问题之一就是：

> iterative denoising 需要多次调用 denoiser。

如果一次 denoising forward 需要 20 ms，进行 20 次：

```text
20 ms × 20
=
400 ms
```

在这 400 ms 期间：

```text
机器人可能已经运动
目标物可能已经变化
外部环境可能改变
```

于是 inference 完成时，模型使用的 observation 已经变旧。

形成：

> stale observation。

这就是生成能力与实时性之间的直接矛盾。

---

## 23.2 Generative Quality 与 Real-Time Control 的冲突

更多 denoising steps 往往意味着：

```text
更充分的生成过程
```

但同时意味着：

```text
更高 latency
```

而机器人控制希望：

```text
高反馈频率
低延迟
持续闭环修正
```

所以机器人生成式 policy 的一个核心问题逐渐显现出来：

```text
复杂分布表达能力
↕
实时控制能力
```

这也是后续更高效 action generation 方法的重要研究动机之一。

---

## 23.3 训练目标不是 Task Success

Diffusion Policy 的训练目标通常是：

$$
L
=
\|\epsilon-\hat\epsilon\|_2^2
$$

但机器人真正关心的是：

```text
任务是否成功
物体是否真正抓起来
是否准确放入目标区域
是否发生碰撞
```

也就是：

```text
Success / Failure
Reward
Task Completion
```

Noise prediction loss 很低，并不严格等价于高任务成功率。

因此它仍然继承 imitation learning 的根本限制：

> 模仿专家行为，并不等于直接优化环境反馈。

这也为后续 VLA + RL / Robot RL post-training 埋下了问题。

---

# 24. 容易混淆的概念

## 24.1 Observation 是输入，还是 Condition？

两句话都不完全错，但后者更精确。

“Observation 是输入”只是描述网络接口。

“Observation 是 Condition”则说明它在生成模型中的功能：

> 它决定 action distribution 应该生成什么。

因此阅读论文时更应该理解：

```text
conditioned on observation
```

而不仅仅是：

```text
observation is concatenated into input
```

---

## 24.2 Action Chunk 和 Diffusion Policy 是一回事吗？

不是。

Action Chunking 是：

> 动作在时间维度上的表示与执行方式。

Diffusion 是：

> action distribution 的生成机制。

所以可以有：

```text
ACT + Action Chunk

Diffusion Policy + Action Chunk

Flow Policy + Action Chunk
```

Action Chunk 并不限定生成模型类型。

---

## 24.3 Diffusion Policy 就是“给动作加噪”吗？

不是。

加噪只是训练机制的一部分。

完整方法应该包含：

```text
conditional action distribution
+
forward noising
+
noise prediction
+
iterative reverse generation
+
action chunk execution
+
closed-loop replanning
```

如果只记住“加噪—去噪”，会漏掉机器人 policy 最关键的 observation conditioning 和 receding-horizon execution。

---

## 24.4 Prediction Horizon 等于 Execution Horizon 吗？

不等于。

Prediction Horizon：

```text
生成多少未来动作
```

Execution Horizon：

```text
实际执行多少预测动作
```

二者分离的目的就是：

> 同时获得较长时序结构和较强 closed-loop correction。

---

## 24.5 Diffusion Policy 使用 MSE，所以它还是普通 MSE BC 吗？

不是。

关键不是 loss 名称，而是 loss 约束的变量。

普通 BC：

```text
预测 action
```

Diffusion：

```text
预测 noise
```

并通过完整 generative process 表示：

$$
p(A\mid o)
$$

---

# 25. Diffusion Policy 在完整机器人系统中的位置

把它放回机器人系统，可以得到：

```text
High-Level Task / Instruction
        ↓
Perception / Observation
        ↓
Diffusion Policy
        ↓
Action Chunk
        ↓
Robot Controller / IK / Servo
        ↓
Actuator
        ↓
Robot + Environment
        ↓
New Observation
```

因此 Diffusion Policy 主要处于：

> Learned Policy / Trajectory Generation Layer。

它通常不会直接替代所有底层控制器。

例如模型可能输出：

```text
EEF delta pose
Joint target
Cartesian velocity
Gripper action
```

之后仍然可能经过：

```text
IK
Operational Space Controller
Joint Position Controller
Servo Loop
```

最终才到电机。

所以需要保持此前 Action Space 章节建立的层次：

```text
Policy
≠
Low-level Motor Controller
```

---

# 26. 本章在 VLA 知识链中的位置

整条主线现在变成：

```text
Robot Policy
↓
Transformer / VLM
↓
Behavioral Cloning
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
VLA
```

此前几个 Chat 分别解决：

```text
BC：
策略如何模仿专家？

Action Representation：
模型到底预测什么？

Action Chunking：
为什么一次预测多个动作？

ACT：
Transformer 如何直接预测 Action Chunk？

Diffusion：
如何表示和生成复杂分布？
```

而 Diffusion Policy 将这些内容组合起来：

> 使用 conditional diffusion，在机器人 observation 的条件下生成 temporal-consistent action chunk。

---

# 27. 为什么 Diffusion Policy 自然通向 Flow Matching？

到这里已经出现一个非常自然的研究问题：

> 如果生成式 action policy 确实有利于建模复杂、多模态动作分布，是否一定需要大量 iterative denoising steps？

Diffusion Policy 的逻辑是：

```text
Noise
↓
逐步去噪
↓
Action
```

后续 Flow Matching 会换一个角度理解这个生成问题。

它关注：

> 从 noise distribution 到 action distribution 的连续概率流动。

其核心形式会写成：

$$
\frac{dx}{dt}
=
v_\theta(x,t\mid c)
$$

其中模型学习一个 velocity field：

```text
当前位置 x
+
时间 t
+
条件 c
↓
应该朝哪个方向流动
```

因此可以把本章结束时的研究动机概括成：

```text
我们希望保留：
Generative Policy
Multimodality
Trajectory Modeling

但希望改善：
Iterative Inference Cost
Latency
Real-Time Deployment
```

---

# 28. 研究视角：阅读 Diffusion Policy 类论文时应该检查什么？

以后拿到一篇 Diffusion / Generative Robot Policy 论文，不应该只问：

> “它是不是用了 diffusion？”

更有研究价值的问题包括：

### 第一类：Observation

```text
输入哪些相机？
是否使用 proprioception？
Observation horizon 是多少？
如何编码视觉？
```

### 第二类：Action

```text
Action space 是什么？
Joint 还是 Cartesian？
Absolute 还是 Delta？
Action dimension 是多少？
```

### 第三类：Trajectory

```text
Prediction horizon 多长？
Execution horizon 多长？
是否 Receding Horizon？
```

### 第四类：Generative Model

```text
Denoiser 是什么结构？
预测 noise 还是其他目标？
需要多少 inference steps？
```

### 第五类：控制与实时性

```text
Policy frequency 是多少？
Control frequency 是多少？
Inference latency 是多少？
是否出现 stale observation？
```

### 第六类：实验能力

```text
是否真正测试 multimodality？
是否在 contact-rich manipulation 上有效？
是否对 perturbation 有恢复能力？
```

### 第七类：数据

```text
多少 demonstrations？
数据来自单一 embodiment 还是多个 robot？
训练分布与测试分布差异多大？
```

这些问题比单纯记住网络名字更接近科研阅读。

---

# 29. 综合思考

## 问题 1

已知：

```text
Observation Horizon = 3
Prediction Horizon = 20
Action Dimension = 7
Execution Horizon = 5
```

请解释模型一次 inference：

1. 看多少个历史 observation；
2. 生成什么 shape 的 action trajectory；
3. 实际执行多少动作；
4. 何时重新 inference。

### 关键思路

应得到：

```text
输入最近 3 个 observation

生成：
20 × 7

真正执行：
前 5 个 action

随后：
获取新 observation，并重新生成新的 20 步 trajectory
```

该问题同时检查 observation、shape、action chunk 和 receding horizon。

---

## 问题 2

对于同一个 observation，有两条完全合理但方向不同的绕障轨迹。为什么普通 MSE regression 可能产生危险结果，而 Diffusion Policy 更合适？

### 关键思路

完整回答应包含：

```text
数据存在两个 mode
↓
普通 deterministic regression 被要求输出单值
↓
MSE 可能把两种模式进行平均
↓
平均 trajectory 未必对应任何真实安全行为

Diffusion Policy：
建模 p(A | o)
↓
可以生成某一个完整 mode
↓
保持 trajectory 内部一致性
```

---

## 问题 3

为什么 Diffusion Policy 明明是 closed-loop policy，却仍然可能受到 stale observation 影响？

### 关键思路

Closed-loop 不意味着零延迟。

真正过程是：

```text
采集 observation
↓
开始 diffusion inference
↓
多步 denoising 需要时间
↓
Inference 完成
↓
环境已经改变
```

因此即使系统会不断重新 observation，单次 inference latency 太高仍然会造成：

> observation-age problem。

---

## 问题 4

ACT 和 Diffusion Policy 都输出 Action Chunk，为什么仍然属于两条不同技术路线？

### 关键思路

因为：

```text
Action Chunk
```

描述的是输出的时间组织形式。

而：

```text
ACT
Diffusion
```

描述的是生成机制。

ACT：

```text
Observation → Direct Prediction → Chunk
```

Diffusion：

```text
Observation + Noise → Iterative Denoising → Chunk
```

---

# 30. 本章总结

Diffusion Policy 最重要的不是“在机器人上使用了 Diffusion”这一事实，而是它重新定义了动作生成问题。

第一，机器人策略不再被简单建模成唯一映射：

$$
A=f(o)
$$

而是条件动作分布：

$$
p(A\mid o)
$$

因此能够更自然地处理机器人 manipulation 中常见的 multimodality。

第二，Diffusion Policy 的扩散对象不是 observation，而是未来 action trajectory。Observation 负责提供 conditioning information。

第三，训练时从专家 action 出发，人为加入 Gaussian noise，并学习预测所加入的 noise：

$$
L
=
\|\epsilon-\epsilon_\theta(A^k,k,o)\|_2^2
$$

而部署时没有专家 action，因此从随机 noise 开始，通过多步 conditional denoising 得到 action chunk。

第四，Diffusion Policy 不只是一个生成模型，还需要放入闭环控制执行流程中理解：

```text
Observation
↓
Generate Action Chunk
↓
Execute Prefix
↓
Observe Again
↓
Regenerate
```

这使其和 Receding-Horizon Control 建立了直接联系。

第五，ACT 与 Diffusion Policy 都可以预测 Action Chunk，但前者主要是 direct prediction，后者是 iterative generative prediction。两者的根本差异在 action generation mechanism，而不是是否使用 action chunk。

最终应该形成如下心智模型：

```text
机器人 observation
        ↓
定义当前任务条件
        ↓
随机 action trajectory
        ↓
conditional denoising
        ↓
生成具有时间一致性的 action chunk
        ↓
只执行部分动作
        ↓
新 observation
        ↓
重新生成
```

而 Diffusion Policy 暴露出的主要问题——尤其是 iterative inference 带来的 latency——又自然提出了下一阶段的问题：

> 能否保留生成式策略对复杂动作分布的表达能力，同时获得更高效的 action generation？

这正是 Flow Matching 进入机器人 VLA 学习主线的起点。

---

# 31. 关键词

**Diffusion Policy — 扩散策略 — 使用条件扩散模型生成机器人动作轨迹的策略。**

**Conditional Diffusion — 条件扩散 — 在 observation 等条件约束下进行生成。**

**Action Trajectory — 动作轨迹 — 一段连续未来动作组成的联合变量。**

**Action Chunk — 动作块 — 一次 policy inference 预测的一段动作序列。**

**Multimodality — 多模态分布 — 相同 observation 下可能存在多种合理动作模式。**

**Gaussian Noise — 高斯噪声 — Diffusion forward process 中用于污染真实 action 的随机噪声。**

**Noise Prediction — 噪声预测 — Diffusion 模型常见的训练目标。**

**Denoising — 去噪 — 将 noisy action trajectory 逐渐恢复为有效动作轨迹的过程。**

**Diffusion Timestep — 扩散时间步 — 表示当前 action 所处噪声等级。**

**Observation Horizon — 观测时域 — 一次推理使用多少个历史 observation。**

**Prediction Horizon — 预测时域 — 一次生成多少未来动作。**

**Execution Horizon / Action Horizon — 执行时域 — 每次真正执行多少个预测动作。**

**Receding-Horizon Control — 滚动时域控制 — 预测未来但只执行一部分，然后根据最新观测重新规划。**

**Temporal Correlation — 时间相关性 — trajectory 中相邻动作之间存在结构化依赖。**

**Direct Prediction — 直接预测 — 例如 ACT 一次 forward 直接生成 Action Chunk。**

**Iterative Generation — 迭代生成 — Diffusion 从 random noise 经多轮修正得到 Action Chunk。**

**Stale Observation — 过时观测 — 推理完成时用于推理的 observation 已不能准确表示当前环境状态。**

**Generative Policy — 生成式策略 — 通过建模动作概率分布而非唯一确定映射来生成机器人行为。**
