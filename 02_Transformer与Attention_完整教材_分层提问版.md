# 02｜Transformer 与 Attention：从 Token 表示到 VLA 多模态信息融合

## 1. 本章导言

Transformer 是现代视觉语言模型（Vision-Language Model, VLM）和视觉语言动作模型（Vision-Language-Action Model, VLA）的核心基础之一。对于机器人学习而言，学习 Transformer 的目的并不是系统研究传统自然语言处理，而是理解一种更一般的信息建模方式：图像、语言、机器人状态、历史观测乃至动作，都可以被转换成 token，再通过 Attention 在统一的表示空间中交换信息。

对于一个典型的机械臂任务，例如：

```text
Pick up the red cup.
```

机器人至少需要同时处理三类信息：

- 摄像头图像：红色杯子在哪里；
- 语言指令：目标对象是 red cup，任务是 pick up；
- 机器人状态：机械臂当前位姿、关节状态和夹爪状态。

如果这些信息彼此独立处理，就需要额外设计多个模块，把“red”与图像中的红色区域联系起来，把“cup”与具体物体联系起来，再把目标位置与机械臂当前状态联系起来。

Transformer 提供了一种统一的信息交互机制：

```text
Visual Tokens
+
Language Tokens
+
Robot-State Tokens
↓
Transformer
↓
Contextual Representation
↓
Action Prediction
```

因此，本章最重要的目标不是背诵 Transformer 的完整历史，而是建立几个以后阅读 VLA 论文和代码时必须具备的心智模型：

1. Token 是什么；
2. Embedding 和 Positional Information 在做什么；
3. Q、K、V 分别代表什么；
4. Attention 为什么能实现 token 间的信息交换；
5. Self-Attention、Cross-Attention 和 Multi-Head Attention 如何区分；
6. Residual、LayerNorm 和 MLP 在 Transformer Block 中分别做什么；
7. Transformer 最终如何进入 VLA，并服务于机器人动作预测。

本章将保持完整教材结构，同时在关键知识层之间插入思考节点。建议在看到问题时先独立回答，再继续阅读解析。

---

# 2. 问题从哪里来：为什么机器人需要 Transformer

传统机器人系统通常具有比较清晰的模块划分：

```text
Perception
↓
State Estimation
↓
Planning
↓
Control
↓
Robot
```

例如，视觉模块先检测杯子位置，规划模块再规划末端轨迹，控制器负责跟踪轨迹。

这种模块化方法的优点是结构清晰、可解释，但在开放环境中，任务语义、视觉目标、机器人状态和长期上下文之间的关系往往非常复杂。现代 VLA 尝试直接学习：

$$
a_t=\pi_\theta(I_t,l,s_t)
$$

其中：

- Iₜ：当前图像 observation；
- l：语言指令；
- sₜ：机器人 proprioception；
- aₜ：当前机器人动作。

这意味着模型内部需要解决一个核心问题：

> 如何让不同来源的信息彼此“理解”？

Transformer 的 Attention 正是围绕这个问题设计的。

可以先把 Transformer 理解为：

> 一个让 token 之间交换信息的神经网络。

以后看到任何 VLA Transformer 架构，都可以先问：

> 谁是 token？

> 谁在和谁做 Attention？

这两个问题通常比直接记模型名字更有用。

---

# 3. 第一层：Token 到底是什么

## 3.1 Token 不等于单词

在 NLP 中，token 常来自文本，因此容易形成：

> Token = 单词

的误解。

更准确地说：

> Token 是 Transformer 处理的一个基本信息单元。

文本可以形成 token，图像 patch 可以形成 token，机器人状态也可以形成 token。

例如：

```text
Pick up the red cup
```

为了建立直觉，可以近似看成：

```text
[Pick] [up] [the] [red] [cup]
```

这里共有 5 个 token。

但在 VLA 中，还可能出现：

```text
[visual token 1]
[visual token 2]
...
[language token 1]
...
[state token]
[action token]
```

因此，Token 更接近一种“统一的信息接口”。

---

## 3.2 为什么需要 Embedding

计算机不能直接对 `"cup"` 这样的字符串执行矩阵乘法。

因此需要把 token 转换成向量。

第 i 个 token 的表示可以写为：

$$
x_i\in\mathbb{R}^{d}
$$

其中：

- xᵢ：第 i 个 token 的 embedding；
- d：embedding dimension。

例如：

$$
x_{\text{cup}}
=
[0.3,\ 0.8,\ 0.6,\ 0.1]
$$

这只是一个极简示意。真实模型中的维度通常远大于 4。

Embedding 的本质不只是“把文字变成数字”，而是：

> 把离散 token 映射到一个可学习的连续特征空间。

模型训练以后，不同 token 的向量关系会逐渐携带语义结构。

---

## 3.3 Token Matrix 与 Shape

如果有 N 个 token：

$$
x_1,x_2,\dots,x_N
$$

可以把它们堆叠成：

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_N
\end{bmatrix}
$$

于是：

$$
X\in\mathbb{R}^{N\times d_{\text{model}}}
$$

这里：

```text
N
=
token 数量

d_model
=
每个 token 的 feature dimension
```

例如：

```text
200 tokens
×
768 dimensions
```

就得到：

$$
X\in\mathbb{R}^{200\times768}
$$

这个 shape 习惯非常重要，因为后续 Transformer 中几乎所有矩阵维度都可以从这里继续推导。

---

## 思考节点 1｜Token、Embedding 与 Shape

在继续阅读前，先回答：

1. 为什么 Transformer 不能直接处理 `"red cup"` 这样的字符串？
2. Embedding 除了“把文字变成数字”之外，还做了什么？
3. 如果输入有 200 个 token，每个 token 是 768 维，那么 X 的 shape 是什么？
4. 为什么在 VLA 中 image patch、language 和 robot state 都可以最终进入同一个 Transformer？

> 建议先独立回答，再继续阅读。

### 解析

第 1 题的最低层答案是：

> 神经网络需要数值输入。

更完整的理解是：

> Token 需要被映射到连续可学习的特征空间，神经网络才能通过矩阵运算学习语义与结构关系。

第 3 题：

$$
X\in\mathbb{R}^{200\times768}
$$

第 4 题的关键在于：

> Transformer 不关心原始数据到底是图像、文字还是机器人状态。只要它们最后都被转换成兼容维度的 token representation，就可以进入统一序列。

---

# 4. 第二层：为什么需要 Positional Encoding

Transformer 中还有一个问题：

> 如果只有 token 内容，模型是否知道 token 的顺序？

考虑：

```text
robot grasps cup
```

和：

```text
cup grasps robot
```

它们包含近似相同的 token，但语义完全不同。

因此，需要把位置信息加入 token 表示。

一种简化形式：

$$
z_i=x_i+p_i
$$

其中：

- xᵢ：token 内容表示；
- pᵢ：位置表示；
- zᵢ：最终输入 Transformer 的表示。

在语言中，位置表示词序。

在机器人中，位置的含义更加丰富。

### 图像空间位置

例如 ViT 把图像切成 patch：

```text
左上 patch
中心 patch
右下 patch
```

如果完全没有位置信息，模型难以知道某个物体处于图像哪个区域。

### 时间位置

机器人可能使用：

$$
o_{t-2},o_{t-1},o_t
$$

模型必须知道：

- 哪个 observation 更旧；
- 哪个 observation 是当前时刻。

因此可以把 Positional Encoding 的作用概括为：

> 告诉模型一个 token 不仅“是什么”，还“位于哪里、处于什么顺序或时间”。

---

## 思考节点 2｜为什么“位置”在机器人中尤其重要

思考：

1. 如果图像 patch 完全没有空间位置，模型会丢失什么信息？
2. 如果历史 observation 没有 temporal position，模型还能区分过去和现在吗？
3. 为什么机器人中的 position information 不只是语言中的词序问题？

### 解析

机器人系统不仅需要语义，还需要空间与时间。

例如：

```text
杯子在左侧
```

和：

```text
杯子在右侧
```

对控制动作的影响可能完全不同。

同样：

```text
机械臂正在靠近杯子
```

和：

```text
机械臂正在远离杯子
```

仅靠单帧状态可能难以判断，因此时序顺序也具有直接控制意义。

---

# 5. 第三层：Attention 到底在“注意”什么

现在已经有：

- token；
- embedding；
- position。

下一步的问题是：

> 一个 token 如何从其他 token 中获得信息？

以：

```text
Pick up the red cup
```

为例。

当模型更新 `"cup"` 的表示时，它可能特别需要：

- `"red"` 的颜色信息；
- `"Pick up"` 的任务语义；
- 图像中杯子所在区域；
- 当前机器人状态。

Attention 的核心思想就是：

> 对当前 token，动态计算其他 token 对它有多重要，然后按这个重要程度读取信息。

所以 Attention 并不是简单：

> “模型在看哪里”。

更准确地说：

> Attention 是一种基于相关性进行动态信息聚合的机制。

---

# 6. 第四层：Q、K、V 的直觉与数学

Q、K、V 是 Transformer 最容易让人产生错误直觉的地方。

给定：

$$
X\in\mathbb{R}^{N\times d_{\text{model}}}
$$

分别通过三个可学习矩阵：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

得到 Query、Key、Value。

---

## 6.1 Query：我想找什么

Query 可以理解为：

> 当前 token 想从上下文中寻找什么。

例如 `"cup"` 可能需要：

- 颜色；
- 空间位置；
- 与任务相关的动作语义。

所以 Query 是一种：

> 查询表示。

---

## 6.2 Key：我有什么特征可供匹配

Key 可以理解为：

> 一个 token 对外提供的“匹配标签”。

例如 `"red"` 的 Key 可能容易与 `"cup"` 当前需要的颜色信息匹配。

因此：

$$
q_{\text{cup}}^Tk_{\text{red}}
$$

如果很大，就说明两者在当前投影空间中匹配程度高。

---

## 6.3 Value：如果你关注我，我真正提供什么

Value 表示：

> 当某个 token 被关注以后，它真正向外传递的内容。

这与“价值大小”完全不是一回事。

Value 不是：

> 这个 token 有多重要。

重要程度由 Attention Weight 决定。

Value 是：

> 真正被读取、被加权、被聚合的信息。

可以形成一句非常重要的记忆：

> Q 和 K 决定“读谁”，V 决定“读什么”。

---

## 6.4 每个 Token 都通常有自己的 Q、K、V

另一个常见误解是：

```text
一些 token 是 Q
一些 token 是 K
一些 token 是 V
```

在 Self-Attention 中通常不是这样。

每个 token 都会产生：

```text
Query
Key
Value
```

只是三个表示来自不同的投影矩阵。

即：

$$
q_i=x_iW_Q
$$

$$
k_i=x_iW_K
$$

$$
v_i=x_iW_V
$$

因此一个 token 同时具有：

- “我想找什么”；
- “别人如何匹配我”；
- “别人关注我时我提供什么”。

---

## 思考节点 3｜Q/K/V 的第一次理解检查

请先回答：

1. Q、K、V 分别在做什么？
2. 如果 `"cup"` 的 Query 和 `"red"` 的 Key 内积很大，说明什么？
3. Value 是不是“这个 token 的价值比例”？

### 解析与纠偏

一个常见回答是：

> Q 是自己的信息，K 是对外信息，V 是这个 token 的价值比例。

这个说法有一定直觉基础，但并不准确。

更好的理解是：

```text
Q：我想找什么
K：别人如何判断我是否符合它的查询
V：别人关注我以后，我真正提供什么内容
```

尤其要纠正：

> V 不表示重要程度。

真正决定“关注多少”的，是后面的 Attention Weight。

---

# 7. 第五层：为什么 QKᵀ 可以表示相关性

假设：

$$
q_i\in\mathbb{R}^{d_k}
$$

$$
k_j\in\mathbb{R}^{d_k}
$$

计算：

$$
q_i^Tk_j
$$

也就是向量点积。

如果两个向量方向较一致，点积通常较大。

例如：

$$
q=[1,0]
$$

两个 Key：

$$
k_1=[1,0]
$$

$$
k_2=[0,1]
$$

那么：

$$
q^Tk_1=1
$$

而：

$$
q^Tk_2=0
$$

因此第一个 Key 与当前 Query 更匹配。

这就是为什么 Attention 使用 Q 与 K 的点积作为相似性 / 匹配程度的一种度量。

---

# 8. 第六层：Q、K、V 的矩阵 Shape

假设：

$$
X\in\mathbb{R}^{N\times d_{\text{model}}}
$$

例如：

$$
X\in\mathbb{R}^{200\times768}
$$

如果：

$$
W_Q\in\mathbb{R}^{768\times64}
$$

那么：

$$
Q=XW_Q
$$

因此：

$$
Q\in\mathbb{R}^{200\times64}
$$

同理：

$$
K,V\in\mathbb{R}^{200\times64}
$$

现在：

$$
QK^T
$$

的 shape 为：

$$
(200\times64)(64\times200)
$$

得到：

$$
QK^T\in\mathbb{R}^{200\times200}
$$

---

## 8.1 这个 200 × 200 矩阵到底是什么

第 i, j 个元素：

$$
(QK^T)_{ij}=q_i^Tk_j
$$

表示：

> 第 i 个 Query 对第 j 个 Key 的匹配分数。

所以整个矩阵可以理解成：

> 每一个 token 对所有 token 的匹配关系。

这也是为什么 Attention 能让一个 token 动态选择从其他 token 中读取信息。

---

## 思考节点 4｜Shape 推导

回答：

1. 如果 X 是 200 × 768，W_Q 是 768 × 64，Q 是什么 shape？
2. 如果 Q 和 K 都是 200 × 64，QKᵀ 是什么 shape？
3. QKᵀ 中一个元素具体表示什么？

### 解析

第一题：

$$
Q\in\mathbb{R}^{200\times64}
$$

第二题：

$$
QK^T\in\mathbb{R}^{200\times200}
$$

第三题：

> 一个元素表示某个 Query 与某个 Key 的匹配程度。

注意，此时还不能叫最终 Attention Weight，因为尚未经过 softmax。

---

# 9. 第七层：Scaled Dot-Product Attention

标准 Attention：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

这一公式可以拆成四步。

---

## 9.1 第一步：QKᵀ

作用：

> 计算 Query 与 Key 的匹配分数。

结果：

$$
QK^T\in\mathbb{R}^{N\times N}
$$

此时得到的是：

> Attention Score

还不是最终 Weight。

---

## 9.2 第二步：为什么要除以 √dₖ

如果 dₖ 很大，点积：

$$
q^Tk
=
\sum_{m=1}^{d_k}q_mk_m
$$

会累加很多维度。

随着 dₖ 增大，点积的数值尺度往往变大。

假设 score 变成：

```text
[2, 3, 25]
```

送入 softmax 后可能得到：

```text
[接近 0, 接近 0, 接近 1]
```

softmax 进入非常饱和的状态。

结果是：

- 梯度变小；
- 训练不稳定。

因此需要：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

进行缩放。

---

## 思考节点 5｜“为了梯度稳定”为什么不够

问题：

> 为什么 Attention 要除以 √dₖ？

如果只回答：

> 为了维护梯度健康。

方向正确，但不够完整。

完整因果链是：

```text
dₖ 增大
↓
QK 点积数值尺度增大
↓
Softmax 更容易过度饱和
↓
梯度变小
↓
训练变得不稳定
↓
除以 √dₖ 做 scaling
```

所以：

> “稳定梯度”是结果，而不是完整机制解释。

---

## 9.3 第三步：Softmax

假设一个 Query 对三个 Key 的 score：

$$
[1,\ 2,\ 4]
$$

经过 softmax：

$$
[0.042,\ 0.114,\ 0.844]
$$

可以理解为：

```text
token A：4.2%
token B：11.4%
token C：84.4%
```

这时候得到的才是：

> Attention Weight

因此必须区分：

```text
QKᵀ
↓
Attention Score
↓
Softmax
↓
Attention Weight
```

---

## 9.4 第四步：Attention Weight 乘 Value

最终：

$$
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

意味着：

> 根据 Q/K 算出来的 Weight，对 Value 做加权求和。

所以：

> Q/K 决定“关注谁以及关注多少”，V 决定“真正取回什么内容”。

---

## 思考节点 6｜Value 的二次检验

假设：

> `"red"` 对 `"cup"` 的最终 Attention Weight 是 0.8。

那么真正被乘以 0.8，并传递给 `"cup"` 的，是：

- `"red"` 的 Q；
- `"red"` 的 K；
- 还是 `"red"` 的 V？

### 解析

答案是：

> V。

因为：

```text
Q/K
↓
决定匹配关系与权重

V
↓
承载实际被加权和聚合的内容
```

这道题是检验 Q/K/V 是否真正建立起来的关键问题。

---

# 10. 一个完整的 Attention 手算例子

设：

$$
x_1=[1,0]
$$

$$
x_2=[1,1]
$$

为了简化：

$$
W_Q=W_K=W_V=I
$$

所以：

$$
Q=K=V=X
$$

现在只看 token 2。

Query：

$$
q_2=[1,1]
$$

Key：

$$
k_1=[1,0]
$$

$$
k_2=[1,1]
$$

---

## 10.1 计算匹配分数

$$
q_2k_1^T
=
1\times1+1\times0
=
1
$$

$$
q_2k_2^T
=
1\times1+1\times1
=
2
$$

得到：

$$
[1,\ 2]
$$

---

## 10.2 Scaling

因为：

$$
d_k=2
$$

所以：

$$
\sqrt{d_k}=\sqrt2
$$

缩放后：

$$
\left[
\frac1{\sqrt2},
\frac2{\sqrt2}
\right]
$$

约为：

$$
[0.707,\ 1.414]
$$

---

## 10.3 Softmax

得到大约：

$$
[0.33,\ 0.67]
$$

表示 token 2：

- 约 33% 关注 token 1；
- 约 67% 关注 token 2。

---

## 10.4 对 Value 加权

$$
v_1=[1,0]
$$

$$
v_2=[1,1]
$$

因此：

$$
z_2
=
0.33v_1+0.67v_2
$$

即：

$$
z_2
=
0.33[1,0]+0.67[1,1]
$$

得到：

$$
z_2=[1,0.67]
$$

新的 token 2 表示已经融合了 token 1 的信息。

这就是 Attention 最核心的结果：

> token 通过动态加权读取其他 token，从而形成上下文化表示。

---

# 11. 第八层：Self-Attention

Self-Attention 中：

> Q、K、V 来自同一个 token sequence。

即：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

都使用同一个 X。

所以它表示：

> 同一组 token 内部互相查询、互相匹配、互相传递信息。

---

## 11.1 VLA 中的 Self-Attention

例如：

```text
Visual Tokens
+
Language Tokens
+
Robot-State Token
↓
Concatenate
↓
Self-Attention
```

此时 `"red"` 可以关注图像中的红色区域；

`"cup"` 可以关注杯子视觉 token；

state token 可以关注当前任务相关信息。

因此：

> Self-Attention 并不意味着单模态。

只要不同模态已经被统一成 token，就可以一起做 Self-Attention。

---

# 12. 第九层：Cross-Attention

Cross-Attention 中：

> Q 来自 A，K/V 来自 B。

例如语言查询视觉：

$$
Q=X_LW_Q
$$

而：

$$
K=X_VW_K
$$

$$
V=X_VW_V
$$

其中：

- X_L：language token；
- X_V：visual token。

如果：

$$
X_L\in\mathbb{R}^{N_L\times d}
$$

$$
X_V\in\mathbb{R}^{N_V\times d}
$$

那么：

$$
QK^T\in\mathbb{R}^{N_L\times N_V}
$$

这个矩阵表示：

> 每个 language token 与每个 visual token 的匹配关系。

例如：

```text
"red cup"
↓
Query

image patches
↓
Key / Value
```

模型就可以找到与 `"red cup"` 最相关的视觉区域。

---

## 思考节点 7｜Self-Attention 与 Cross-Attention

问题：

> Self-Attention 和 Cross-Attention 最本质的区别是什么？

一个容易出现的简化回答是：

> 是否在一组 token 内进行计算。

这个方向基本正确。

更标准的判断方式是：

> 看 Q、K、V 分别来自哪里。

Self-Attention：

```text
Q / K / V
都来自同一组 token
```

Cross-Attention：

```text
Q 来自 A
K / V 来自 B
```

还要避免一个典型误区：

> Cross-Attention = 多模态  
> Self-Attention = 单模态

这并不成立。

---

# 13. 第十层：Multi-Head Attention

如果只使用一个 Attention Head，模型只能在一组 Q/K/V 投影空间中学习关系。

但真实机器人任务中的关系非常多样。

例如：

```text
Head 1：
red ↔ 红色区域

Head 2：
cup ↔ 杯子形状

Head 3：
pick ↔ 抓取任务语义

Head 4：
robot state ↔ 当前可达状态
```

第 i 个 head：

$$
\text{head}_i
=
\operatorname{Attention}(Q_i,K_i,V_i)
$$

然后：

$$
\operatorname{MultiHead}
=
\operatorname{Concat}
(\text{head}_1,\dots,\text{head}_h)W_O
$$

例如：

```text
d_model = 512
h = 8
```

常见情况下：

```text
每个 head
dₖ = 64
```

因为：

```text
8 × 64 = 512
```

重点是：

> Multi-Head 并不是把同一件事重复计算多次。

不同 head 有不同的 W_Q、W_K、W_V，因此可以学到不同关系子空间。

---

# 14. Transformer Block 不只有 Attention

Transformer 常见结构还包括：

- Multi-Head Attention；
- Residual Connection；
- LayerNorm；
- MLP / FFN。

一种常见的 Pre-LN 形式：

$$
X_1
=
X+
\operatorname{MHA}(\operatorname{LN}(X))
$$

接着：

$$
X_2
=
X_1+
\operatorname{MLP}(\operatorname{LN}(X_1))
$$

---

# 15. Residual Connection

Residual Connection 的基本形式：

$$
y=x+F(x)
$$

不是：

$$
y=F(x)
$$

它的核心作用是：

> 在保留原表示的基础上，叠加本层新学到的信息。

例如：

$$
X'=X+\operatorname{Attention}(X)
$$

这种结构有助于：

- 深层网络训练；
- 梯度传播；
- 避免每一层都完全覆盖原表示。

---

# 16. LayerNorm

LayerNorm 可以简化写成：

$$
\hat{x}
=
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

其中：

- μ：特征均值；
- σ²：特征方差；
- ε：防止除零的小常数。

本章不深入完整推导，只需要理解：

> LayerNorm 主要用于稳定特征尺度和深层网络训练。

---

# 17. MLP / FFN 到底在做什么

这是另一个容易模糊的地方。

Attention 已经让 token 之间交换信息，那么为什么还需要 MLP？

因为：

> Attention 负责“从谁那里获取信息”，但获取到的信息仍然需要在当前 token 内部进一步处理。

经典 FFN：

$$
\operatorname{FFN}(x)
=
W_2\sigma(W_1x+b_1)+b_2
$$

常见结构：

```text
d_model
↓
升维
↓
d_ff
↓
激活函数
↓
降维
↓
d_model
```

例如：

```text
1024
↓
4096
↓
GELU
↓
1024
```

---

## 思考节点 8｜Attention 与 MLP 的分工

问题：

> Attention 和 MLP 在 Transformer 中分别负责什么？

一个非常有用的心智模型是：

```text
Attention
=
token 与 token 之间 communication

MLP
=
每个 token 内部 computation
```

即：

> Attention 负责交流，MLP 负责消化。

这不是严格的数学定义，但对理解 Transformer Block 非常有效。

---

# 18. Encoder、Decoder 与 Autoregressive Generation

## 18.1 Encoder

Encoder 主要把输入转换成上下文化表示。

例如：

```text
Image
↓
ViT Encoder
↓
Visual Tokens
```

或者：

```text
Text
↓
Text Encoder
↓
Language Representation
```

---

## 18.2 Decoder

Decoder 常用于逐步生成输出。

例如：

```text
The robot should
↓
预测 move

The robot should move
↓
预测 left
```

数学上：

$$
p(x_t|x_{<t})
$$

即：

> 根据之前 token 预测下一个 token。

这叫：

> Autoregressive Generation。

---

## 18.3 Causal Mask

自回归训练时，预测当前位置不能看到未来 token。

因此：

```text
token 1 → 只能看 token 1
token 2 → 看 token 1,2
token 3 → 看 token 1,2,3
```

这就是 Causal Mask。

在 VLA 中需要注意：

> 并不是所有动作都通过 autoregressive token 生成。

后续还会学习连续回归、Diffusion、Flow Matching 等 action generation 方法。

---

# 19. 完整 VLA 机器人实例：Pick up the red cup

现在把本章所有概念放回完整机器人任务。

任务：

```text
Pick up the red cup.
```

场景：

```text
左边：蓝色盒子
右边：红色杯子
```

机器人状态：

```text
EEF position
joint state
gripper state
```

---

## 19.1 图像进入模型

摄像头图像经过视觉编码：

```text
Image
↓
Patchify / Vision Encoder
↓
196 Visual Tokens
```

每个 visual token 表示某个局部视觉区域。

---

## 19.2 语言进入模型

语言：

```text
Pick up the red cup
```

经过 tokenizer 和 embedding：

```text
Pick
up
the
red
cup
↓
Language Tokens
```

假设产生 5 个 language tokens。

---

## 19.3 Robot State 进入模型

机器人 proprioception：

```text
joint position
EEF pose
gripper state
```

经过线性层或 MLP：

```text
Robot State
↓
Embedding
↓
State Token
```

---

## 19.4 拼接 Token

例如：

```text
196 visual
+
5 language
+
1 state
=
202 tokens
```

假设：

```text
d_model = 1024
```

则：

$$
X\in\mathbb{R}^{202\times1024}
$$

---

## 19.5 Transformer 建立跨模态关系

多层 Attention 中可能形成：

```text
"red"
↔
红色视觉区域

"cup"
↔
杯子视觉区域

"pick up"
↔
任务语义

state token
↔
当前机械臂状态
```

逐层后，每个 token 都形成上下文化表示。

---

## 19.6 输出动作

最终 action head 可能输出：

$$
a_t=
[\Delta x,\Delta y,\Delta z,\Delta R,g]
$$

其中：

- Δx、Δy、Δz：末端位置增量；
- ΔR：姿态变化；
- g：夹爪动作。

需要特别纠正一个表述：

> Transformer 不是“调整机器人状态”。

更准确地说：

> Transformer 利用 robot state、vision 和 language 形成上下文表示，然后预测 action。

动作执行后：

```text
Observation_t
↓
VLA
↓
Action_t
↓
Robot / Environment
↓
Observation_{t+1}
```

整个系统仍然是闭环机器人 policy。

---

## 思考节点 9｜把 Attention 真正放回 VLA

假设 VLA 输入：

```text
196 个 visual tokens
10 个 language tokens
1 个 robot-state token
```

思考：

> `"red cup"` 这段语言信息，会怎样通过 Attention 与图像 token、state token 建立联系，并最终影响机器人动作？

### 解析

完整逻辑应该包括：

```text
"red"
↓
找到颜色相关视觉区域

"cup"
↓
找到杯子视觉特征

"pick up"
↓
确定任务类型

state token
↓
知道机械臂当前位置和状态

多模态融合
↓
形成任务相关 representation
↓
Action Head
↓
机器人动作
```

如果回答只停留在：

> 找到了红色杯子。

还不够，因为 VLA 最终必须把“语义目标”转换成“基于当前 robot state 的动作”。

---

# 20. 与传统控制 / Robot Learning 的联系

## 20.1 Transformer 不等于机器人控制器

一个完整机器人栈可以粗略表示：

```text
High-Level Task Planning
↓
Learned Policy / VLA
↓
Action / Trajectory Generation
↓
Robot Controller
↓
Actuator
```

Transformer 通常属于：

> Learned Policy / VLA 内部。

它不等于：

- PID；
- IK；
- Operational Space Controller；
- 电机 PWM；
- 驱动器。

---

## 20.2 与传统状态反馈的区别

传统控制常写：

$$
u_t=\pi(x_t)
$$

其中 xₜ 是明确状态。

VLA 可能学习：

$$
a_t=\pi_\theta(I_t,l,s_t)
$$

这里模型需要从高维、多模态 observation 中自己形成任务相关 representation。

因此：

> Transformer 的主要作用是 representation learning 和 information interaction，而不是显式动力学建模。

---

## 20.3 与 BC / RL 的区别

需要明确区分：

```text
Transformer
=
模型架构 / 信息交互机制

BC / SFT / RL
=
训练策略的方法

Action Representation
=
模型输出什么

Controller
=
真实机器人如何执行
```

这几个概念以后会在 VLA 论文中同时出现，不能混为一谈。

---

# 21. 容易混淆的概念

## 21.1 Token 与单词

Token 不等于单词。

视觉 patch、state、action 都可以 token 化。

---

## 21.2 Token 与 Embedding

Token 是离散信息单位。

Embedding 是 token 的连续向量表示。

---

## 21.3 Q、K、V

错误：

> V 表示 token 的重要程度。

正确：

```text
Q / K
→ 决定匹配和 Weight

V
→ 提供真正被聚合的信息
```

---

## 21.4 Attention Score 与 Attention Weight

$$
QK^T
$$

得到的是 Score。

经过 Softmax 后才是 Weight。

---

## 21.5 Self-Attention 与 Cross-Attention

Self：

> Q/K/V 来自同一组 token。

Cross：

> Q 与 K/V 来自不同组 token。

---

## 21.6 Multi-Head Attention 与多个 Transformer

Multi-Head 是一个 Attention 层内部的多个 head。

不是多个完整 Transformer。

---

## 21.7 Attention 与 MLP

Attention：

> token 之间交换信息。

MLP：

> token 内部继续做非线性处理。

---

## 21.8 Transformer 与 Attention

Transformer 不等于 Attention。

Transformer Block 还包括：

- Residual；
- LayerNorm；
- MLP。

---

## 21.9 多模态融合与 Cross-Attention

多模态融合不一定需要 Cross-Attention。

也可以把：

```text
Vision
+
Language
+
State
```

拼成同一个序列以后做 Self-Attention。

---

# 22. 本章内容在 VLA 知识体系中的位置

本章处于：

```text
Robot Policy
↓
Transformer / Attention
↓
ViT / Visual Representation
↓
Vision-Language Model
↓
Robot Learning / BC
↓
Action Representation
↓
Action Chunking
↓
Diffusion / Flow Matching
↓
VLA
```

本章主要解决：

> 多模态信息如何变成 token，并在模型中进行上下文交互。

后续会进一步回答：

- 图像如何具体变成 visual tokens；
- ViT、CLIP 如何工作；
- VLM 如何融合视觉和语言；
- Robot Learning 如何训练 policy；
- action 如何表示；
- Transformer 如何预测 action chunk；
- Diffusion 和 Flow Matching 如何生成动作。

---

# 23. 局限与研究视角

Transformer 很强，但仅仅“用了 Transformer”并不意味着机器人系统已经解决。

## 23.1 Token 数量与计算成本

Attention 关系矩阵：

$$
QK^T\in\mathbb{R}^{N\times N}
$$

随着 N 增加，计算和存储成本都会上升。

真实 VLA 可能有：

- 大量 image tokens；
- language tokens；
- history tokens；
- state tokens；
- action tokens。

因此必须考虑推理成本。

---

## 23.2 实时机器人还有 Latency 问题

即使 Transformer 能正确理解多模态信息，如果推理太慢：

```text
Observation_t
↓
长时间推理
↓
Action
```

真正执行时环境可能已经变化。

所以真实机器人还要考虑：

- inference frequency；
- control frequency；
- latency；
- stale observation。

这些问题会在后续专门学习。

---

## 23.3 Transformer 不能自动解决 Distribution Shift

Attention 负责信息交互，但不能自动保证：

- 新环境泛化；
- 新视角泛化；
- 新机器人适配；
- 数据外状态恢复。

这些问题会连接后续：

- BC；
- DAgger；
- VLA fine-tuning；
- Agent recovery；
- RL post-training。

---

## 23.4 阅读 VLA 论文时应该问什么

看到新的 VLA 模型，可以优先检查：

1. 输入有哪些 token？
2. Visual token 如何产生？
3. Language token 如何产生？
4. Robot state 如何进入？
5. 每类 token 的 shape 是什么？
6. Q/K/V 分别来自哪里？
7. 使用 Self-Attention 还是 Cross-Attention？
8. 是否有 action query？
9. Transformer 输出如何进入 action head？
10. 最终 action 是 token、continuous regression 还是其他形式？
11. Token 数量和推理成本是多少？
12. 是否满足真实机器人实时执行需求？

这些问题能帮助从“知道 Transformer”进入“能分析 VLA 架构”。

---

# 24. 综合思考

这一组问题用于检查是否已经形成完整心智模型。

### 问题 1

如果：

```text
N = 128
d_model = 512
```

那么 X 的 shape 是什么？

### 问题 2

如果：

```text
W_Q = 512 × 64
```

那么 Q 的 shape 是什么？

### 问题 3

为什么 QKᵀ 的 shape 是 N × N？

### 问题 4

为什么 V 不是“Attention Weight”？

### 问题 5

为什么除以 √dₖ？

要求完整说明：

```text
dimension
→ dot product scale
→ softmax
→ gradient
```

### 问题 6

Self-Attention 是否只能处理单模态？

为什么？

### 问题 7

Attention 和 MLP 的分工是什么？

### 问题 8

在机械臂抓红杯子的任务中：

```text
"red"
"cup"
visual tokens
state token
```

如何共同影响最终 action？

### 问题 9

Transformer 是否等于机器人 controller？

如果不是，它位于完整机器人系统哪一层？

---

# 25. 本章总结

本章最核心的不是某一个公式，而是一条完整信息流。

```text
不同模态原始信息
↓
Token
↓
Embedding
↓
Positional Information
↓
Q / K / V
↓
Attention
↓
Token 间动态信息交换
↓
MLP
↓
Contextual Representation
↓
Action Head
↓
Robot Action
```

需要真正形成以下几个心智模型。

第一：

> Token 是统一信息接口，不只是单词。

第二：

> Attention 的本质是根据 Query 与 Key 的匹配关系，动态决定从哪些 Value 中读取多少信息。

第三：

> Q 和 K 决定“读谁”，V 决定“读什么”。

第四：

> QKᵀ 是 Score，Softmax 后才得到 Weight。

第五：

> Self-Attention 和 Cross-Attention 的核心区别，是 Q/K/V 的来源。

第六：

> Attention 负责 token 间 communication，MLP 负责 token 内 computation。

第七：

> Transformer 是 VLA policy 内部的多模态表示与信息融合机制，不是完整机器人控制系统。

以后看到一个新的 VLA 架构时，如果能够自然追问：

```text
谁是 token？
谁和谁 Attention？
Q/K/V 来自哪里？
输出怎么变成 action？
```

就已经具备继续进入 ViT、VLM、ACT 和现代 VLA 的基础。

---

# 26. 关键词

**Token — 基本信息单元 — Transformer 接收和处理的序列单位，可以来自文本、图像、机器人状态或动作。**

**Embedding — 嵌入表示 — 把离散 token 映射到连续可学习向量空间。**

**Positional Encoding — 位置编码 — 给 token 加入顺序、空间或时间信息。**

**Query (Q) — 查询向量 — 表示当前 token 想寻找什么信息。**

**Key (K) — 键向量 — 表示一个 token 用于被其他 Query 匹配的特征。**

**Value (V) — 值向量 — 表示 token 被关注后真正传递的信息内容。**

**Attention Score — 匹配分数 — Q 与 K 点积得到的匹配程度。**

**Attention Weight — 注意力权重 — Attention Score 经过 Softmax 后得到的归一化权重。**

**Scaled Dot-Product Attention — 缩放点积注意力 — 使用 QKᵀ/√dₖ 计算匹配并对 V 加权。**

**Self-Attention — 自注意力 — Q、K、V 来自同一组 token。**

**Cross-Attention — 交叉注意力 — Q 与 K/V 来自不同 token 集合。**

**Multi-Head Attention — 多头注意力 — 使用多组 Q/K/V 投影同时学习不同关系。**

**Residual Connection — 残差连接 — 在保留原表示的同时加入本层的新变换。**

**LayerNorm — 层归一化 — 稳定 token 特征尺度和深层网络训练。**

**MLP / FFN — 多层感知机 / 前馈网络 — 对每个 token 的特征进行独立非线性处理。**

**Encoder — 编码器 — 把输入转换为上下文化 representation。**

**Decoder — 解码器 — 常用于根据已有上下文逐步产生输出。**

**Autoregressive Generation — 自回归生成 — 根据已有 token 逐步预测后续 token。**

**Causal Mask — 因果掩码 — 防止模型在自回归预测中看到未来 token。**

**Visual Token — 视觉 Token — 图像经过视觉编码后形成的 Transformer 输入单元。**

**State Token — 状态 Token — 由机器人 proprioception 等状态信息编码得到的 token。**

**Contextual Representation — 上下文化表示 — token 融合其他相关 token 信息后形成的表示。**

**Vision-Language-Action Model (VLA) — 视觉语言动作模型 — 以视觉、语言和机器人状态为输入，预测机器人动作或动作序列的策略模型。**
