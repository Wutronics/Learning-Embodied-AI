# Chat 09｜ACT：Action Chunking with Transformers

## 1. 本章导言：从 Action Chunking 走向真正的机器人策略网络

在上一阶段的学习中，Action Chunking 已经建立了一个重要思想：机器人策略不一定每个控制周期只预测一个动作，也可以一次预测未来一段连续动作。

普通单步策略通常写成：

$$
a_t=\pi_\theta(o_t)
$$

而 Action Chunking 将其扩展为：

$$
\hat A_t
=
\pi_\theta(o_t)
=
[
\hat a_t,
\hat a_{t+1},
\dots,
\hat a_{t+K-1}
]
$$

其中：

- o_t 表示当前 observation；
- a_t 表示机器人单步动作；
- K 表示 Action Chunk 长度；
- A_t 表示从当前时刻开始的一整段未来动作。

但“预测一段动作”仍然只是一个思想。真正的问题是：

> 怎样设计一个神经网络，使它能够根据当前视觉和机器人状态，一次产生几十甚至上百个连续、协调的未来动作？

ACT（Action Chunking with Transformers）就是这一问题的经典答案之一。

在 VLA 学习路线中，ACT 位于 Action Chunking 与 Diffusion Policy 之间。它第一次非常清楚地展示了：

```text
视觉 Observation
+
机器人状态
↓
Transformer
↓
Continuous Action Chunk
```

这种现代机器人策略范式。

它仍然不是 VLA，因为 ACT 本身不包含语言指令，也没有大型 Vision-Language Backbone。但它已经具备了后续现代 VLA 非常重要的几个结构思想：

- Transformer 建模机器人决策；
- 一次生成多个连续动作；
- 使用 learned query 生成未来动作槽位；
- 对动作序列进行联合建模；
- 在执行阶段不断重新预测；
- 对多个 overlapping action chunks 进行融合。

因此，理解 ACT 不只是为了理解一篇机器人论文，更是为了建立后续阅读 Diffusion Policy、OpenVLA、π0 等模型时所需的中间桥梁。

---

# 2. ACT 要解决什么问题？

## 2.1 单步 Behavioral Cloning 的基本形式

在最基础的 Behavioral Cloning 中，专家数据给出：

$$
(o_t,a_t^*)
$$

策略学习：

$$
\hat a_t=\pi_\theta(o_t)
$$

训练目标是让预测动作接近专家动作：

$$
\hat a_t\approx a_t^*
$$

这种方式非常直观，但在长时间机器人任务中存在一个明显问题：

> 策略需要进行大量连续决策。

假设一个 manipulation episode 长度为 1000 个 timestep。

单步 BC 就要依次完成：

```text
o₀ → a₀
o₁ → a₁
o₂ → a₂
...
o₉₉₉ → a₉₉₉
```

也就是说，整个任务具有大约 1000 步的有效决策 horizon。

只要其中某一步存在小误差：

```text
小动作误差
↓
机器人进入稍微不同的状态
↓
后续 observation 偏离专家数据
↓
策略预测误差进一步增大
↓
Distribution Shift
↓
Compounding Error
```

这就是机器人 imitation learning 中最经典的问题之一。

## 2.2 Action Chunking：把单步动作变成动作段

ACT 不再只预测：

$$
a_t
$$

而是一次预测：

$$
A_t
=
[
a_t,
a_{t+1},
\dots,
a_{t+K-1}
]
$$

于是策略变成：

$$
\hat A_t=\pi_\theta(o_t)
$$

假设：

```text
Episode length = 1000
Chunk size K = 100
```

从动作段的角度看：

```text
1000 个单步动作
↓
约 10 个长度为 100 的动作段
```

这就是所谓降低 effective horizon。

需要注意，这里的“10 个 chunk”并不意味着真实机器人一定只调用策略 10 次。

Effective horizon 描述的是：

> 策略不再把长任务完全看成数百个相互独立的单步决策，而开始联合学习较长时间范围内的动作结构。

所以：

```text
降低 effective horizon
≠
部署时必须 open-loop 执行完整 chunk
```

这是理解 ACT 的第一个重要区分。

# 3. 为什么 Action Chunk 比单步 Action 更有意义？

机器人动作并不是独立随机变量。

例如一个简单抓取过程：

```text
向目标靠近
↓
降低末端执行器
↓
对准物体
↓
闭合夹爪
↓
抬起
```

这些动作存在明显的 temporal correlation。

如果分别预测：

$$
a_t
$$

$$
a_{t+1}
$$

$$
a_{t+2}
$$

策略主要学习“当前这一瞬间应该做什么”。

而 ACT 直接预测：

$$
[
a_t,
a_{t+1},
\dots,
a_{t+K-1}
]
$$

模型可以联合学习整段动作内部的结构。

例如：

```text
靠近
→ 对准
→ 抓取
→ 抬升
```

不再只是许多彼此割裂的瞬时控制命令，而成为一个连续 motion pattern。

因此，Action Chunking 的价值不仅仅是减少网络调用次数，更重要的是：

> 让策略显式学习动作在时间维度上的相关结构。

## 思考节点 A：Action Chunk 是否意味着 Open-loop？

假设 ACT 一次预测 100 个未来动作。

是否意味着机器人必须连续执行完这 100 个动作之后，才能再次获取 observation？

### 解析

不是。

预测 horizon 与 execution horizon 是两个不同概念。

模型可以预测：

```text
未来 100 步
```

但只执行其中的一部分，甚至下一 timestep 就再次获取 observation 并重新推理。

因此：

```text
Prediction Horizon
≠
Execution Horizon
```

Action Chunking 解决的是动作序列建模问题，而是否重新观察、何时重新规划属于执行策略问题。

这一区分在后续 Real-Time VLA 中仍然非常重要。

# 4. ACT 的第二个问题：专家 Demonstration 并不唯一

Action Chunking 解决了时间结构问题，但 imitation learning 还有另一个问题：

> 对同一个 observation，可能存在多个都正确的动作。

例如桌面上存在障碍物。

专家 A 选择：

```text
从左侧绕过
```

专家 B 选择：

```text
从右侧绕过
```

两个行为都正确。

因此真实动作条件分布并不一定是：

```text
一个 observation
→
唯一一个 action
```

更可能是：

```text
一个 observation
→
多个合理 action mode
```

也就是 multimodal action distribution。

## 4.1 为什么简单回归可能出问题？

假设两条合理轨迹分别为：

$$
A^{(1)}
$$

和：

$$
A^{(2)}
$$

如果直接进行普通回归，模型可能倾向预测某种平均行为：

$$
\hat A
\approx
\frac{
A^{(1)}+A^{(2)}
}{2}
$$

但在机器人控制中：

```text
左绕
+
右绕
```

的平均不一定是另一条正确路径。

它甚至可能变成：

```text
直接走中间
→
撞向障碍物
```

因此机器人动作生成不仅需要“预测准确”，还需要表示 demonstration 中存在的不同模式。

ACT 为此引入了 CVAE。

# 5. CVAE 与 latent variable z

## 5.1 z 的直觉

假设存在三条抓杯子的 demonstration：

```text
轨迹 A：从左侧靠近 → 抓取 → 抬起

轨迹 B：从右侧靠近 → 抓取 → 抬起

轨迹 C：从正面靠近 → 抓取 → 抬起
```

当前视觉状态可能非常接近，但动作序列不同。

ACT 引入一个 latent variable：

```text
z
```

可以暂时把它理解成：

> 当前 demonstration 所采用的隐藏动作模式。

例如为了建立直觉，可以想象：

```text
z₁ ≈ 左侧抓取模式
z₂ ≈ 右侧抓取模式
z₃ ≈ 正面抓取模式
```

但真实网络中的 z 并不是文字标签，而是连续向量，例如：

```text
[0.21, -0.73, 0.15, ...]
```

数据集中也没有：

```text
style = "left grasp"
```

这样的监督标签。

因此 z 被称为 latent variable，也就是潜变量。

## 5.2 CVAE 不是“增加数据多样性”

一个非常容易出现的误区是：

> CVAE 是为了提升 demonstration 数据的多样性。

这种说法不准确。

CVAE 并不会凭空让原始 demonstration 变得更多。

更准确的理解是：

> CVAE 用于建模 demonstration 数据中原本已经存在的多样性。

即：

```text
不是制造多样性
而是表示已有多样性
```

这是一个重要区别。

## 思考节点 B：CVAE 到底解决什么？

如果同一个 observation 对应多条合理专家轨迹，为什么不能简单把它们全部拿来做普通监督学习？

### 解析

因为普通回归倾向寻找一个能整体降低误差的输出。

当目标分布高度 multimodal 时，这种“平均预测”不一定对应任何真实有效的动作模式。

CVAE 通过 latent variable z 提供一个额外条件：

```text
Observation
+
z
↓
不同合理行为模式
```

因此模型可以表示：

```text
相同 Observation
+
不同 z
→
不同 Action Sequence
```

正确心智模型不是：

```text
CVAE 让数据更多
```

而是：

```text
CVAE 帮助策略表示 demonstration 中已有的行为多模态性
```

# 6. 训练阶段如何得到 z？

训练时，我们拥有专家 future action chunk：

$$
A_t^*
=
[
a_t^*,
a_{t+1}^*,
\dots,
a_{t+K-1}^*
]
$$

CVAE Encoder 接收当前机器人状态和专家动作序列：

$$
(q_t,A_t^*)
\rightarrow
\text{CVAE Encoder}
$$

Encoder 不直接输出一个固定 z，而是产生高斯分布参数：

$$
\mu
$$

以及：

$$
\sigma
$$

从而定义：

$$
q_\phi(z|q_t,A_t^*)
=
\mathcal N(\mu,\sigma^2)
$$

随后通过重参数化采样：

$$
z
=
\mu
+
\sigma\odot\epsilon
$$

其中：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

因此训练阶段的数据流可以理解成：

```text
Current Joint State
+
Expert Action Chunk
↓
CVAE Encoder
↓
μ, σ
↓
sample z
```

z 再作为 ACT 主策略的一个输入。

# 7. ACT 主策略的输入

ACT 的策略部分可以抽象为：

$$
\hat A_t
=
\pi_\theta(I_t,q_t,z)
$$

其中：

- I_t：当前视觉 observation；
- q_t：当前机器人 joint state；
- z：latent style representation；
- A_t：未来 action chunk。

因此完整输入由三类信息组成：

```text
视觉
+
机器人状态
+
latent style
```

而输出是：

```text
未来一段连续机器人动作
```

# 8. 图像如何进入 ACT？

原始 RGB image 无法直接作为普通 Transformer token 使用。

因此首先需要视觉 backbone：

```text
Image
↓
CNN / Visual Backbone
↓
Feature Map
```

可以抽象表示：

$$
I_t
\rightarrow
F_t
$$

假设视觉 backbone 得到空间 feature map：

```text
H' × W' × d_model
```

可以把空间位置展开成：

```text
N 个 Visual Tokens
```

其中：

```text
N = H' × W'
```

每一个视觉 token 都具有：

```text
d_model
```

维 feature。

于是：

```text
Image
↓
Visual Features
↓
Visual Tokens
```

这一过程与 VLM 中“图像先转化成 Transformer 可处理的表示”具有相似思想。

但 ACT 中没有 language tokens，也不存在 LLM backbone。

# 9. Joint State 为什么需要 Projection？

以 ALOHA 双臂机器人为例，可以把当前 joint state 表示为：

```text
14 维
```

即：

$$
q_t\in\mathbb R^{14}
$$

但假设 Transformer 的内部特征维度为：

```text
d_model = 512
```

那么原始 14 维状态不能直接与 512 维 Transformer representation 对齐。

因此需要线性投影：

$$
e_q
=
W_q q_t+b_q
$$

得到：

$$
e_q\in\mathbb R^{512}
$$

也就是：

```text
14D Joint State
↓
Linear Projection
↓
512D Representation
```

latent z 也需要进行类似投影，使不同模态最终都进入统一 feature space。

## 思考节点 C：为什么不能把 14 维 joint state 直接送进 512 维 Transformer？

### 解析

核心不是“Transformer 不认识 14 维数字”，而是：

> Transformer 中参与 token interaction 的 representation 需要具有统一 embedding dimension。

例如视觉 token 已经是：

```text
512 维
```

那么 joint state 也要投影到：

```text
512 维
```

才能方便参与统一的 attention 和 feature processing。

因此：

```text
Projection
```

本质是在完成：

```text
机器人原始状态空间
→
Transformer representation space
```

# 10. Observation Memory

视觉信息、机器人状态以及 latent representation 被编码后，可以抽象形成：

$$
M
=
\operatorname{Encoder}(X)
$$

这里的 M 可以理解为：

> Transformer 对当前 observation 的综合内部表示。

它并不是某个单独变量的物理含义，而是一组 feature representation。

其中可能隐式包含：

```text
目标物体位置
机械臂当前姿态
夹爪状态
场景几何信息
latent action style
```

可以把它直觉理解为：

```text
Observation
↓
Transformer Encoding
↓
Current Scene Memory
```

后续 action queries 会从这个 memory 中读取与各自未来动作位置有关的信息。

# 11. ACT 最关键的机制之一：Action Query

现在来到 ACT 中最值得理解的结构之一。

假设 chunk size 为：

```text
K = 90
```

ACT 希望一次产生：

```text
90 个未来动作
```

因此模型准备 90 个 learned action queries：

$$
q_1,q_2,\dots,q_{90}
$$

假设：

```text
d_model = 512
```

那么每个 query 都是一个 512 维向量：

$$
q_i\in\mathbb R^{512}
$$

于是整个 query tensor 的 shape 为：

```text
[90, 512]
```

而不是把两者相乘成：

```text
46080
```

Tensor shape 中的两个维度代表不同语义：

```text
90
=
未来 90 个动作位置

512
=
每个 query 的 feature dimension
```

这种 shape 意识对后续阅读 ACT、Diffusion Policy 和 VLA 代码非常重要。

# 12. Action Query 到底储存了什么？

Action Query 并不是：

```text
q₁ = 向左移动
q₂ = 抓紧
q₃ = 抬起
```

这种明确动作命令。

初始化时，它们通常只是模型中的 learned embeddings。

训练过程中，由于每个 query 始终对应一个固定 future slot，例如：

```text
q₁ → future step 1
q₂ → future step 2
...
q₉₀ → future step 90
```

它们逐渐形成不同的时间位置角色。

因此可以把 qᵢ 直觉理解为：

> “我要为未来第 i 个时间槽位读取 observation 信息并生成一个动作。”

而不能理解成：

> “qᵢ 本身已经保存了未来第 i 步应该执行什么动作。”

动作仍然取决于当前 observation。

# 13. 为什么一个 Observation 能生成 K 个不同动作？

这是理解 ACT Transformer 的核心。

所有 action queries 都读取同一个 observation memory：

$$
M
$$

但：

$$
q_1\neq q_2\neq\dots\neq q_K
$$

所以可以得到：

$$
h_i
=
\operatorname{Decoder}(q_i,M)
$$

其中：

- q_i 提供不同 future output slot；
- M 提供当前环境信息；
- h_i 表示对应未来位置的 decoder representation。

因此：

```text
同一个 Observation Memory
+
不同 Action Queries
↓
不同 Future Actions
```

直觉上：

```text
q₁：基于当前 observation，未来第 1 步应该怎么动？

q₂：基于当前 observation，未来第 2 步应该怎么动？

...

q₉₀：基于当前 observation，未来第 90 步应该怎么动？
```

这些“问题”并不是自然语言，而只是帮助理解 learned query 的功能。

# 14. Action Query 与 Attention 中的 Q 必须严格区分

这是学习 ACT 时最容易混淆的概念之一。

Transformer Attention 中有：

$$
Q=XW_Q
$$

这里：

- X 是输入 representation；
- W_Q 是可学习投影矩阵；
- Q 是由输入经过 W_Q 投影得到的 query representation。

因此：

```text
W_Q
=
模型参数

Q
=
根据当前输入计算得到的 Attention Query
```

而 ACT 的 Action Query：

```text
q₁, q₂, ..., q_K
```

本身就是 learnable embeddings。

所以必须区分：

```text
Action Query qᵢ
≠
Attention 中的 Q
≠
投影矩阵 W_Q
```

三者只是都使用了“query”这一术语，但概念不同。

## 思考节点 D：哪一个才是 Attention 的模型参数？

考虑：

$$
Q=XW_Q
$$

下面哪一个是直接训练得到的模型参数？

```text
A. Q
B. X
C. W_Q
```

### 解析

正确答案是 W_Q。

Q 不是一个固定模型参数，而是：

```text
当前输入 X
经过
可学习矩阵 W_Q
得到的动态 representation
```

ACT 的 Action Query 又是第三种情况：

```text
qᵢ 本身就是可学习参数
```

因此这三个概念不能混为一谈。

# 15. Transformer Decoder 如何产生 Action？

Decoder 得到每个 query 对应的 hidden representation：

$$
H
=
[
h_1,h_2,\dots,h_K
]
$$

假设每个 h_i 是：

```text
512 维
```

但机器人真正需要的 action 可能只有：

```text
14 维
```

因此最后通过 action head：

$$
a_i
=
W_a h_i+b_a
$$

实现：

```text
512
↓
Linear Action Head
↓
14
```

于是：

```text
[K, 512]
→
[K, 14]
```

如果加入 batch：

```text
[B, K, 512]
→
[B, K, 14]
```

例如：

```text
B = 32
K = 90
Action dimension = 14
```

最终得到：

```text
[32, 90, 14]
```

含义为：

```text
32 条训练样本
×
每条预测 90 个 future timesteps
×
每个 action 14 维
```

## 思考节点 E：Tensor Shape 不能直接“乘掉”

假设有：

```text
100 个 Action Queries

每个 Query 512 维
```

正确 shape 是什么？

### 解析

应写为：

```text
[100, 512]
```

虽然其中确实包含：

```text
100 × 512
```

个标量，但把它写成单独一个数字会丢失 tensor 的结构语义。

在机器人深度学习代码中：

```text
[B, T, D]
```

通常意味着：

```text
Batch
×
Time
×
Feature
```

这些维度代表不同物理含义，不能仅用元素总数代替。

# 16. ACT 不是 World Model

ACT 能一次预测未来很多步动作，因此很容易产生一种误解：

> 既然它能预测未来 90 步，那是不是已经在预测未来世界？

不是。

ACT 学习的是：

$$
o_t
\rightarrow
a_{t:t+K-1}
$$

而 World Model 更接近：

$$
(s_t,a_t)
\rightarrow
s_{t+1}
$$

或者预测未来 observation / state。

ACT 并没有显式计算：

```text
执行 aₜ 后世界会变成什么样
↓
再根据新世界预测下一状态
```

它只是从 demonstration 中学习：

> 在类似 observation 下，专家通常会执行怎样的未来动作序列。

因此 ACT 的本质仍然是：

```text
Policy
```

而不是：

```text
Dynamics Model / World Model
```

## 思考节点 F：ACT 为什么不能自动应对未来环境变化？

假设 ACT 根据当前画面预测未来 90 个动作。

在执行到第 30 步之前，有人突然把目标杯子移动了。

原先预测的第 31～90 步是否仍然可靠？

### 解析

不可靠。

因为这些动作是基于旧 observation：

```text
o_t
```

生成的。

ACT 没有显式预测未来环境变化。

因此：

```text
环境发生变化
↓
旧 Observation 失效
↓
旧 Action Chunk 可能变 stale
```

这也是为什么 Action Chunk Policy 必须与重新观察、重新规划和闭环执行机制结合。

# 17. ACT 的训练目标

ACT 仍然属于 Behavioral Cloning。

训练数据提供专家 action chunk：

$$
A_t^*
=
[
a_t^*,
a_{t+1}^*,
\dots,
a_{t+K-1}^*
]
$$

模型预测：

$$
\hat A_t
=
[
\hat a_t,
\hat a_{t+1},
\dots,
\hat a_{t+K-1}
]
$$

训练目标包括两类约束：

```text
动作重构
+
latent space 约束
```

可以抽象表示为：

$$
L
=
L_{\text{action}}
+
\beta L_{\text{KL}}
$$

其中：

- L_action 负责让预测动作接近专家动作；
- L_KL 负责约束 latent distribution；
- β 控制两部分 loss 的权重。

# 18. Action Reconstruction Loss

动作预测首先必须满足：

$$
\hat A_t\approx A_t^*
$$

因此可以把动作重构理解为：

$$
L_{\text{action}}
=
\sum_{i=0}^{K-1}
\left\|
\hat a_{t+i}
-
a_{t+i}^*
\right\|
$$

它检查的是：

> 模型生成的未来动作序列与专家 demonstration 是否接近。

如果预测：

```text
[0.12, 0.18, 0.31, 0.38]
```

专家为：

```text
[0.10, 0.20, 0.30, 0.40]
```

差异越小：

```text
Action Loss 越低
```

# 19. 为什么需要 KL Loss？

如果只有 reconstruction loss，CVAE Encoder 可以任意组织 latent space。

例如可能形成：

```text
某种轨迹 → z ≈ 100

另一种轨迹 → z ≈ -500

另一种轨迹 → z ≈ 7000
```

这种 latent space 虽然可能仍然可以帮助 reconstruction，却非常不规整。

于是 ACT 通过 KL divergence，让：

$$
q_\phi(z|q_t,A_t)
$$

尽量接近标准高斯：

$$
\mathcal N(0,I)
$$

对应的 KL 项为：

$$
L_{\text{KL}}
=
D_{\text{KL}}
\left(
q_\phi(z|q_t,A_t)
\|
\mathcal N(0,I)
\right)
$$

这一项的作用不是直接改善 action accuracy，而是整理 latent space。

因此可以记成：

```text
Action Loss
→
管动作是否像专家

KL Loss
→
管 latent space 是否规整
```

# 20. 为什么 KL Loss 与推理时 z = 0 有关系？

这是理解 ACT CVAE 最容易卡住的一步。

如果训练得到的 z 都分布在：

```text
100
300
-500
7000
```

附近，那么推理阶段突然使用：

```text
z = 0
```

就相当于输入模型几乎没见过的 latent。

但经过 KL regularization 后，CVAE 会被约束，使 latent distribution 更接近：

$$
\mathcal N(0,I)
$$

标准高斯的中心在：

```text
0
```

因此：

```text
KL Regularization
↓
Latent Distribution 接近 N(0, I)
↓
z = 0 成为合理的典型区域
↓
Inference 使用 z = 0 更合理
```

需要注意：

> 这并不是因为 z = 0 必然对应“最优动作”。

而是因为训练使 latent space 被组织在标准高斯附近，所以 0 成为了一个合理的默认 latent。

## 思考节点 G：为什么不能只说“z 要符合高斯分布”？

### 解析

“z 符合高斯分布”只描述结果，没有解释为什么。

完整逻辑应为：

```text
Inference 阶段没有专家 future action
↓
不能再通过 CVAE Encoder 得到当前 demonstration 的 z
↓
需要一个默认 latent
↓
ACT 选择 z = 0
↓
为了让 z = 0 成为训练分布中的合理位置
↓
训练时通过 KL Loss 将 latent distribution 拉向 N(0, I)
```

只有把这一整条因果链连接起来，才真正理解 KL loss 在 ACT 中的作用。

# 21. Training 与 Inference 的关键区别

## 21.1 Training

训练时有专家 future action：

```text
Joint State
+
Expert Action Chunk
↓
CVAE Encoder
↓
μ, σ
↓
sample z
```

随后：

```text
Images
+
Joint State
+
z
↓
ACT Policy
↓
Predicted Action Chunk
```

再与专家动作比较：

```text
Predicted Action
vs
Expert Action
↓
Action Loss
+
KL Loss
↓
Backpropagation
```

## 21.2 Inference

真正部署时没有：

```text
Expert Future Action
```

因此 CVAE Encoder 无法走训练阶段那条路径。

ACT 直接使用：

```text
z = 0
```

于是：

```text
Images
+
Joint State
+
z = 0
↓
ACT Policy
↓
Action Chunk
```

# 22. 一个关键误区：Inference 不会反向传播更新 z

容易出现这样的理解：

```text
推理时先令 z = 0
↓
执行后再通过反向传播逐步更新 z
```

这不是普通 ACT inference。

标准推理只有：

```text
Observation
↓
Forward Pass
↓
Action
↓
Robot Execution
```

不存在：

```text
Loss
↓
Backward
↓
Parameter Update
```

因此：

```text
Inference:
z = 0
并保持为推理输入

不是:
z = 0 后继续通过 backprop 在线学习
```

只有进入 online RL、test-time adaptation 等特殊研究范式时，部署期间才可能继续修改模型参数。

那已经不是普通 ACT inference。

## 思考节点 H：Training 和 Inference 如何快速区分？

看到下面结构：

```text
Forward
↓
Loss
↓
Backward
↓
Optimizer Step
```

这是 Training。

而：

```text
Observation
↓
Forward
↓
Action
```

这是 Inference。

ACT 中 CVAE Encoder 主要服务于训练阶段，而不是实时部署阶段持续运行的在线适应模块。

# 23. ACT 为什么仍然是 Behavioral Cloning？

ACT 使用了：

- Transformer；
- CVAE；
- learned action query；
- latent variable；
- KL loss；
- Action Chunking。

网络结构已经远比基础 BC 复杂。

但判断一个算法是否是 BC，不能看：

```text
网络复杂不复杂
```

而要看：

> 它究竟用什么监督信号学习策略？

ACT 的训练数据仍然是：

$$
(o,A^*)
$$

训练目标仍然是：

$$
\pi_\theta(o)\approx A^*
$$

也就是模仿专家 demonstration。

它没有依靠：

```text
Reward
Q Function
Value Function
Bellman Backup
Policy Gradient
```

来优化长期环境回报。

所以：

```text
ACT
=
复杂的 Behavioral Cloning Policy
```

更准确地说，它属于：

```text
Offline Supervised Imitation Learning
```

## 思考节点 I：复杂网络是否意味着 RL？

不意味着。

判断标准是学习信号：

```text
专家动作监督
→
Imitation Learning / BC

Environment Reward
→
RL
```

ACT 虽然比简单 BC 复杂很多，但优化目标仍然是模仿专家 action chunk，因此学习范式没有改变。

# 24. Temporal Ensemble：为什么同一个 timestep 会有多个预测？

假设：

```text
K = 4
```

在 t = 0 时预测：

```text
[a₀⁽⁰⁾, a₁⁽⁰⁾, a₂⁽⁰⁾, a₃⁽⁰⁾]
```

在 t = 1 时重新观察，又预测：

```text
[a₁⁽¹⁾, a₂⁽¹⁾, a₃⁽¹⁾, a₄⁽¹⁾]
```

在 t = 2 时再次预测：

```text
[a₂⁽²⁾, a₃⁽²⁾, a₄⁽²⁾, a₅⁽²⁾]
```

那么真正执行 timestep 2 时，已经存在三个候选：

```text
a₂⁽⁰⁾
a₂⁽¹⁾
a₂⁽²⁾
```

原因不是简单“因为重新观察了”。

完整因果关系是：

```text
不断重新观察
↓
每次都预测一个完整 Action Chunk
↓
不同 Chunk 在时间上 overlap
↓
同一个实际 timestep 被多个 Chunk 同时预测
```

因此才可以做 Temporal Ensemble。

# 25. Temporal Ensemble 的基本思想

如果同一实际 timestep 存在多个预测：

$$
\hat a_t^{(1)},
\hat a_t^{(2)},
\dots
$$

ACT 可以对它们进行加权融合：

$$
a_t
=
\frac{
\sum_i
w_i\hat a_t^{(i)}
}{
\sum_i w_i
}
$$

权重可以随预测时间差变化，例如采用指数形式。

直觉上：

```text
多个不同时间产生的未来预测
↓
针对同一实际 timestep 对齐
↓
加权融合
↓
得到最终执行 action
```

这使动作在 chunk 之间更加平滑。

# 26. Temporal Ensemble 为什么有用？

假设直接执行一个完整 chunk：

```text
Chunk A
────────────

Chunk B
            ────────────
```

两个 chunk 切换的位置可能出现不连续。

例如：

```text
Chunk A 最后一个动作
```

与：

```text
Chunk B 第一个动作
```

差别较大，就可能导致机器人产生突变或抖动。

而 Temporal Ensemble 中：

```text
Chunk 1 ────────────
Chunk 2   ────────────
Chunk 3     ────────────
Chunk 4       ────────────
```

大量 action prediction 在时间上重叠。

因此最终动作不是突然从一个 chunk 跳向另一个 chunk，而是融合多个预测。

可以把它理解成：

```text
Chunk hard switching
↓
Overlapping prediction fusion
```

# 27. Temporal Ensemble 与 Receding Horizon 的区别

这是本章最重要的概念辨析之一。

## Receding Horizon

假设一次预测 20 步：

```text
[aₜ, ..., aₜ₊₁₉]
```

但只执行前 5 步：

```text
执行 5 步
↓
重新观察
↓
预测新的 20 步
```

旧 chunk 未执行部分直接放弃。

本质是：

```text
旧计划截断
+
新计划替代
```

其主要目标是：

> 保持闭环、避免长期使用已经过时的计划。

## Temporal Ensemble

Temporal Ensemble 不把所有旧预测直接丢掉。

如果不同 chunk 对当前 timestep 都有预测：

```text
旧 Chunk 的预测
+
中间 Chunk 的预测
+
最新 Chunk 的预测
↓
加权融合
```

因此：

```text
Receding Horizon
=
旧计划截断 + 新计划替代

Temporal Ensemble
=
多个 Overlapping Chunks 融合
```

两者不是互斥的。

## 思考节点 J：二者到底分别解决什么？

一个容易出现的回答是：

> Temporal Ensemble 就是比 Receding Horizon 更灵活一些。

这种描述太模糊。

更准确的因果关系是：

```text
Receding Horizon
主要解决：
计划过时、闭环反馈问题

Temporal Ensemble
主要解决：
多个 chunk 的衔接和平滑问题
```

所以：

```text
一个偏重新规划

一个偏重叠预测融合
```

这是两者最核心的区别。

# 28. 一个完整 Pick-and-Place 实例

现在把 ACT 放进一个完整机械臂 Pick-and-Place 任务。

任务：

```text
将桌面药盒抓起
↓
移动到右侧篮子
↓
释放
```

假设机器人 observation 包含：

```text
RGB Camera
+
Current Joint State
```

策略需要预测：

```text
未来 90 步 joint targets
```

## 28.1 当前 Observation

相机看到：

```text
药盒位于机械臂右前方
篮子位于右侧
机械臂当前尚未接触药盒
```

机器人 joint state：

```text
qₜ
```

## 28.2 Vision Encoder

RGB image 经过视觉 backbone：

```text
Image
↓
CNN
↓
Visual Feature Map
↓
Visual Tokens
```

视觉 token 隐式表达：

```text
药盒位置
篮子位置
桌面结构
机械臂当前视觉状态
```

## 28.3 Robot State Projection

当前 joint state：

```text
14 维
```

经过 projection：

```text
14
↓
512
```

得到与 Transformer feature dimension 对齐的机器人状态 representation。

## 28.4 Latent z

训练阶段：

```text
Expert Action Chunk
+
Current Joint State
↓
CVAE Encoder
↓
z
```

推理阶段：

```text
z = 0
```

## 28.5 Transformer Observation Representation

视觉、机器人状态与 latent information 被 Transformer 编码成 observation memory。

## 28.6 90 个 Action Queries

准备：

```text
q₁
q₂
...
q₉₀
```

分别负责未来 90 个动作槽位。

它们与 observation memory 交互后得到：

```text
h₁
h₂
...
h₉₀
```

## 28.7 Action Head

每个 hidden representation：

```text
512D
```

通过 action head 转换为：

```text
14D Joint Action
```

最终：

```text
[90, 14]
```

动作序列可能隐式对应：

```text
未来早期：
接近药盒

中期：
降低机械臂
闭合夹爪
抬起

后期：
向篮子移动
准备释放
```

注意模型不会显式输出这些文字阶段。

它只输出连续 joint action 数值。

## 28.8 Closed-loop Execution

下一 timestep 获取新图像：

```text
Iₜ₊₁
```

重新得到：

```text
新的 Action Chunk
```

不同 chunk 重叠。

针对当前执行时刻产生多个 action prediction，再通过 Temporal Ensemble 融合。

因此：

```text
预测长 Chunk
```

与：

```text
持续闭环更新
```

可以同时存在。

# 29. ACT 在机器人控制系统中的位置

ACT 不是底层电机控制器。

典型链路可以写成：

```text
Camera / Robot State
↓
ACT Policy
↓
Joint Target / Action Chunk
↓
Robot Controller
↓
Motor / Actuator
↓
Physical Robot
```

也就是说 ACT 主要工作在：

```text
Learned Policy / Trajectory Generation
```

这一层。

它并不直接负责：

```text
PWM
Motor Current Loop
Torque Loop
Servo Control
```

等底层控制。

从控制视角看，可以把它理解为：

```text
Observation
↓
Learned Reference Generator
↓
Lower-level Controller
↓
Robot Dynamics
```

这一点有助于把现代 Robot Learning 与传统控制体系连接起来。

# 30. ACT 为什么有效？

到这里可以把 ACT 的核心优势总结为四部分。

## 30.1 Action Chunking

将：

```text
单步预测
```

提升为：

```text
动作序列预测
```

使模型能够直接建模 temporal correlation。

同时降低长任务的 effective horizon。

## 30.2 Transformer + Action Queries

Transformer 利用 observation representation，并通过多个 learned action queries 产生不同未来时间位置的动作。

核心结构是：

```text
Current Observation
+
K Action Queries
↓
K Future Actions
```

## 30.3 CVAE

处理 human demonstration 中存在的不同合理行为模式。

它不是增加 demonstration 数据，而是：

```text
建模已有 demonstration variation
```

## 30.4 Temporal Ensemble

利用：

```text
Overlapping Action Chunks
```

对同一实际 timestep 的多个 action prediction 进行融合。

主要改善：

```text
动作连续性
Chunk 切换平滑性
```

# 31. ACT 与普通 BC 的本质区别

普通 BC：

$$
o_t
\rightarrow
a_t
$$

ACT：

$$
o_t
\rightarrow
A_{t:t+K-1}
$$

因此两者最核心差异不是：

```text
一个使用 Transformer
一个不使用 Transformer
```

而是：

```text
普通 BC
→
单步 action mapping

ACT
→
sequence-level action mapping
```

ACT 仍然是 BC，只是改变了：

- action representation；
- prediction horizon；
- network architecture；
- multimodality modeling；
- execution aggregation。

# 32. ACT 仍然没有解决什么？

虽然 ACT 明显强于最基础的单步 BC，但它仍然存在许多局限。

## 32.1 Distribution Shift 仍然存在

ACT 训练仍然依赖 expert dataset。

如果机器人部署时进入训练数据很少覆盖的状态：

```text
OOD Observation
↓
错误 Action Chunk
↓
进一步偏离 Expert Distribution
```

ACT 仍然可能失败。

Action Chunking 缓解了 compounding error，但并没有从根本上消除 distribution shift。

## 32.2 长 Chunk 仍然可能过时

预测：

```text
未来 90 步
```

并不意味着未来环境不会变化。

如果：

```text
物体移动
机器人碰撞
夹爪滑脱
目标发生变化
```

旧 chunk 后半段就可能成为 stale action。

因此现代 action-chunk policy 必须与：

```text
Replanning
Receding Horizon
Asynchronous Inference
Reactive Correction
```

结合。

## 32.3 CVAE 对复杂 multimodal distribution 的建模能力有限

CVAE 已经比简单 regression 更适合处理多模态 demonstration。

但复杂机器人动作分布可能拥有：

```text
多个 mode
非线性结构
复杂 trajectory geometry
```

这也是后续 Diffusion Policy 出现的重要动机之一。

# 33. 为什么 ACT 之后自然走向 Diffusion Policy？

ACT 已经证明了一件非常关键的事情：

> 机器人策略可以把整个未来 Action Sequence 当成一个整体进行预测。

这一步非常重要。

从：

```text
Single Action Regression
```

走向：

```text
Action Trajectory Generation
```

以后，一个自然问题就出现了：

> 如果未来动作本身是复杂的 multimodal distribution，能否使用更强的生成模型来直接建模整个 action trajectory？

于是后续出现 Diffusion Policy。

ACT 的思路可以概括为：

```text
Observation
↓
Transformer
↓
Action Chunk
```

Diffusion Policy 则进一步转向：

```text
Observation
+
Noisy Action Trajectory
↓
Conditional Generative Model
↓
Denoising
↓
Action Trajectory
```

因此知识链可以理解为：

```text
Behavioral Cloning
↓
Action Chunking
↓
ACT
↓
Generative Action Sequence Modeling
↓
Diffusion Policy
↓
Flow-based Policy
↓
Modern VLA
```

这里不需要提前展开 Diffusion 的机制。

本章只需要建立一个动机：

> ACT 已经把“整段动作生成”变成核心建模对象，而 Diffusion Policy 会进一步研究“如何更好地表示复杂动作分布”。

# 34. 容易混淆的概念总整理

## 34.1 Action Chunking vs Open-loop Execution

Action Chunking：

```text
一次预测多个动作
```

Open-loop：

```text
执行期间不重新利用最新 observation
```

二者不是同一个概念。

## 34.2 CVAE vs Data Augmentation

CVAE：

```text
建模 demonstration 中已有的行为多样性
```

Data Augmentation：

```text
人为增加或转换训练样本
```

不能混淆。

## 34.3 Action Query vs Attention Q vs W_Q

```text
Action Query
=
ACT 中的 learned output embedding

Attention Q
=
XW_Q 得到的动态 query representation

W_Q
=
Attention 内的可学习投影矩阵
```

## 34.4 Action Chunk Prediction vs World Model

ACT：

```text
Observation
→
Future Actions
```

World Model：

```text
Current State + Action
→
Future State / Observation
```

预测未来动作不等于预测未来世界。

## 34.5 Training vs Inference

Training：

```text
Forward
↓
Loss
↓
Backward
↓
Parameter Update
```

Inference：

```text
Observation
↓
Forward
↓
Action
```

原始 ACT inference 中不会通过反向传播持续更新 z。

## 34.6 Receding Horizon vs Temporal Ensemble

Receding Horizon：

```text
旧计划截断
+
新计划替代
```

Temporal Ensemble：

```text
多个 overlapping chunks
+
对同一 timestep 的预测加权融合
```

## 34.7 Tensor Shape vs 元素数量

```text
[90, 14]
```

不能简单写成：

```text
1260
```

因为前者保留：

```text
Time × Action Dimension
```

的结构含义。

# 35. 本章在 VLA 知识体系中的位置

目前已经完成的知识链可以理解为：

```text
Robot Policy
↓
Transformer
↓
Vision Representation
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
```

ACT 是第一次把前面多个知识模块真正组合到一个完整机器人 policy 中：

```text
Visual Observation
↓
Feature Encoding

Robot State
↓
Projection

Action Chunking
↓
Future Sequence Representation

Transformer
↓
Multi-step Action Prediction

Behavioral Cloning
↓
Supervised Training
```

下一阶段继续研究：

```text
Diffusion
↓
Diffusion Policy
```

目的是进一步增强 action distribution 的生成能力。

# 36. 阅读 ACT 类论文时应主动检查什么？

以后再遇到类似 action-chunk policy，不应只看网络名字，而应该主动检查以下设计。

第一，Observation 是什么？

```text
RGB?
Multiple Cameras?
Robot State?
Depth?
```

第二，Action Space 是什么？

```text
Joint Position?
EEF Pose?
Delta Pose?
```

第三，Chunk Size 是多少？

```text
K = ?
```

第四，预测 horizon 与 execution horizon 是否一致？

第五，是否每个 timestep 都重新推理？

第六，多个 chunk 是否 overlap？

第七，是否使用 Temporal Ensemble？

第八，Action Distribution 如何建模？

```text
Regression?
CVAE?
Diffusion?
Flow?
```

第九，训练是否仍然是 Behavioral Cloning？

第十，部署时如何处理 stale actions 与 inference latency？

这套问题以后几乎可以直接迁移到现代 VLA 论文阅读中。

# 37. 综合思考

## 思考 1

如果 episode 长度为 1000，chunk size 为 100，为什么不能简单说：

```text
ACT 一共只需要推理 10 次？
```

### 解析

因为：

```text
1000 / 100 = 10
```

描述的是 chunk 对 effective horizon 的直觉压缩。

真实部署时为了保持闭环，可能：

```text
每个 timestep 都重新预测 Action Chunk
```

因此：

```text
Effective Horizon Reduction
≠
Inference Call Reduction
```

## 思考 2

为什么 q₅₀ 不能理解为：

```text
“未来第 50 步向左移动 2 cm”
```

### 解析

因为 q₅₀ 本身不包含具体动作。

它只是：

```text
未来第 50 个输出槽位的 learned representation
```

真正动作必须结合当前 observation memory：

```text
q₅₀
+
Observation
↓
Decoder
↓
a₅₀
```

## 思考 3

为什么 expert action tensor 必须和 model prediction 具有对应 shape？

假设模型输出：

```text
[B, 90, 14]
```

那么 expert target 也应组织成：

```text
[B, 90, 14]
```

因为 action reconstruction 要逐个对应：

```text
Batch
Future Timestep
Action Dimension
```

进行比较。

## 思考 4

为什么 ACT 不是 RL？

因为它没有通过环境 reward 优化策略。

训练信号仍然来自：

```text
Expert Demonstration
```

即：

```text
预测动作
vs
专家动作
```

因此它属于 Behavioral Cloning。

## 思考 5

为什么 ACT 不是 World Model？

因为它直接预测：

```text
Future Action
```

而不是预测：

```text
执行动作之后世界会发生什么
```

ACT 学的是 policy mapping，而不是环境动力学模型。

## 思考 6

Temporal Ensemble 为什么要求不同 Action Chunks 在时间上 overlap？

因为只有 overlap 后，同一真实 timestep 才可能拥有多个预测。

例如：

```text
Chunk 0 → 对 timestep 2 有预测
Chunk 1 → 对 timestep 2 有预测
Chunk 2 → 对 timestep 2 有预测
```

这样才有：

```text
多个 candidate predictions
↓
Weighted Ensemble
```

如果 chunk 完全不重叠，就没有 ensemble 的对象。

# 38. 本章核心心智模型

ACT 可以最终压缩成下面这张结构：

```text
                     Training Only
Expert Action Chunk ───────────────┐
                                   ↓
Joint State ───────────────→ CVAE Encoder
                                   ↓
                                   z

Camera Images
      ↓
Visual Backbone
      ↓
Visual Features ───────────────────┐
                                   │
Joint State → Projection ──────────┤
                                   │
z → Projection ────────────────────┤
                                   ↓
                        Observation Representation
                                   │
                  ┌────────────────┼────────────────┐
                  │                │                │
                 q₁               q₂              q_K
                  │                │                │
                  └──── Transformer Decoder ───────┘
                                   ↓
                         Future Representations
                                   ↓
                             Action Head
                                   ↓
                        Continuous Action Chunk
                                   ↓
                      Temporal Ensemble / Execute
```

训练阶段的本质是：

```text
Expert Demonstration
↓
Supervised Imitation
```

推理阶段的本质是：

```text
Current Observation
↓
ACT
↓
Future Action Chunk
↓
Closed-loop Replanning
```

# 39. 本章总结

ACT 最重要的意义，不只是提出了一个具体网络，而是把现代机器人 imitation policy 的几个关键思想整合在了一起。

第一，机器人策略不一定只预测一个瞬时动作。

ACT 将：

```text
oₜ → aₜ
```

扩展为：

```text
oₜ → Action Chunk
```

从而联合建模连续动作的 temporal correlation，并降低长任务的 effective horizon。

第二，一个 observation 能够同时产生多个 future actions，是因为 Transformer 使用了一组 learned action queries。

这些 query 并不保存具体动作，而是提供不同 future output slots。

第三，human demonstration 往往具有 multimodality。

ACT 使用 CVAE latent variable z 建模 demonstration 中原本存在的不同合理行为模式。

第四，KL regularization 与 inference 阶段使用 z = 0 是一条完整逻辑链。

```text
KL
↓
Latent Distribution 接近 N(0, I)
↓
0 成为合理 latent 区域
↓
Inference 使用 z = 0
```

第五，Action Chunking 不等于完全 open-loop。

ACT 可以不断使用最新 observation 重新产生 overlapping chunks。

Temporal Ensemble 再对同一 timestep 的多个预测进行融合，从而提高动作连续性。

第六，ACT 仍然只是 Behavioral Cloning Policy。

它没有环境 reward，没有 Q function，也没有显式环境动力学模型。因此它既不是 RL，也不是 World Model。

最终可以形成这样一条知识链：

```text
单步 Behavioral Cloning
↓
长 Horizon 与 Compounding Error
↓
Action Chunking
↓
Transformer + Action Queries
↓
ACT
↓
Action Sequence Generation
↓
更复杂 Multimodal Action Distribution
↓
Diffusion Policy
```

理解到这里，ACT 的学习目标就不应只是“知道论文用了 Transformer”。

真正应该建立的是：

> **机器人可以把未来动作轨迹作为一个整体生成对象，而 Transformer、Action Query、CVAE 和 Temporal Ensemble 分别解决序列生成、未来槽位、多模态 demonstration 和闭环动作融合问题。**

这才是 ACT 对后续现代 Robot Learning 与 VLA 学习最重要的铺垫。

# 40. 关键词

**Action Chunk — 动作块**  
一次预测的一段连续未来动作。

**Action Chunking — 动作分块预测**  
将单步动作预测改为未来动作序列预测。

**Effective Horizon — 有效决策时域**  
策略在学习过程中需要处理的长期决策跨度。

**Temporal Correlation — 时间相关性**  
相邻或远期机器人动作之间存在的结构关系。

**ACT — Action Chunking with Transformers**  
利用 Transformer 一次预测连续 Action Chunk 的 imitation learning policy。

**Behavioral Cloning, BC — 行为克隆**  
利用专家 observation-action 数据进行监督学习。

**CVAE — Conditional Variational Autoencoder**  
通过条件 latent variable 建模 demonstration 中的行为多样性。

**Latent Variable z — 潜变量**  
表示 demonstration 隐含动作模式的连续 representation。

**KL Divergence — KL 散度**  
用于约束 latent distribution 接近指定 prior。

**Action Query — 动作查询向量**  
ACT 中对应不同未来动作位置的 learned embedding。

**Attention Query Q — 注意力查询表示**  
由输入经过 W_Q 投影生成的 Attention 内部 representation。

**W_Q — Query Projection Matrix**  
Transformer Attention 中可学习的 Query 投影参数。

**Observation Memory — 观测表示**  
Transformer 对当前视觉、机器人状态等信息形成的内部 representation。

**Action Head — 动作输出头**  
将 Transformer hidden feature 映射为机器人 action dimension。

**Temporal Ensemble — 时间集成**  
融合多个 overlapping action chunks 对同一 timestep 的预测。

**Receding Horizon — 滚动时域执行**  
只执行计划的一部分，然后根据最新 observation 重新规划。

**Stale Action — 过时动作**  
基于旧 observation 生成、但当前环境已经发生变化的动作。

**Multimodal Action Distribution — 多模态动作分布**  
同一个 observation 下存在多种合理动作模式。

**World Model — 世界模型**  
预测环境状态或 observation 如何随 action 演化的模型。

**Inference — 推理**  
模型使用当前 observation 进行 forward pass 并输出 action，不进行普通训练反向传播。

**Training — 训练**  
通过 loss、backpropagation 和 optimizer 更新模型参数的过程。
