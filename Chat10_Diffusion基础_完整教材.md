# Chat 10｜Diffusion 基础：从多模态动作分布到条件去噪策略

## 1. 本章导言

在现代机器人学习中，策略并不总是面对“一个状态对应唯一正确动作”的问题。很多 manipulation 任务天然存在多种合理解：机械臂可以从物体左侧接近，也可以从右侧接近；可以采用不同抓取姿态；可以绕过障碍物的不同一侧；也可以通过不同的中间轨迹完成同一个 Pick-and-Place 任务。

这种现象意味着，机器人策略真正面对的往往不是简单的确定性映射，而是一个**条件动作分布**。

Diffusion 的价值正是在这里出现。

本章不讨论图像生成，也不追求完整的扩散模型概率论推导，而是围绕机器人策略需要回答几个核心问题：

- 为什么机器人动作分布可能是多模态的？
- 为什么直接使用 MSE 回归动作可能产生“平均动作”？
- Diffusion 为什么要先把真实动作加成噪声？
- 模型为什么可以通过预测 noise 学会生成动作？
- Conditional Diffusion 如何结合图像、机器人状态和任务条件？
- 为什么 Diffusion 特别适合生成 Action Chunk？
- 它相比 ACT 又付出了什么代价？

在完整知识链中，本章处于：

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
现代 VLA
```

因此，本章真正需要建立的不是“会推扩散模型公式”，而是：

> 能够把 Diffusion 理解为一种条件动作分布建模方法，并知道它如何从 Gaussian noise 逐渐生成一段合理的机器人动作轨迹。

---

# 2. 为什么机器人策略需要生成式建模

## 2.1 确定性动作预测的隐含假设

最简单的 Behavioral Cloning 可以写成：

$$
A=f_\theta(o)
$$

其中：

- o 表示 observation；
- A 表示机器人动作，或者一整个 Action Chunk；
- θ 表示策略网络参数。

这种表达隐含了一个重要假设：

> 给定 observation 后，存在一个相对明确的“正确答案”。

如果这个假设成立，直接使用回归方法通常没有问题。

例如，机械臂当前末端位置略低于目标点，专家演示中对应动作几乎总是：

```text
向上移动一点
```

大量样本可能集中在：

```text
0.48
0.50
0.51
0.52
```

这种情况下，平均值仍然是一个合理动作。

问题出现在：

> 同一个 observation 下，数据中存在多个彼此差异很大、但都合理的动作模式。

---

## 2.2 什么是 Multimodal Action Distribution

考虑一个简单任务。

机械臂需要抓取障碍物后方的杯子：

```text
                 杯子
                  ●

             ┌────────┐
             │ 障碍物 │
             └────────┘

          ↗               ↖
       左侧绕行         右侧绕行

               机械臂
```

专家 A 可能从左侧绕过障碍物。

专家 B 可能从右侧绕过障碍物。

两条轨迹最终都能成功抓到杯子。

于是对于同一个 observation，存在：

$$
A^{(1)}
$$

以及：

$$
A^{(2)}
$$

两种不同但合理的动作轨迹。

因此，比下面这种表达：

$$
A=f(o)
$$

更准确的形式是：

$$
A\sim p(A\mid o)
$$

它表示：

> 给定 observation 后，机器人动作不是一个唯一值，而是从条件分布中产生的一个合理样本。

如果这个条件分布存在多个明显峰值，就称它具有 **Multimodality，多模态性**。

示意为：

```text
概率密度
   ↑
   │        /\                  /\
   │       /  \                /  \
   │______/    \______________/    \______
   │
   └──────────────────────────────────────→ Action
          模式 1               模式 2
```

两个峰都可能代表合理行为。

---

## 阶段思考 A：什么才算“多模态”？

### 问题 1

如果一个机器人面对同一个目标存在多种完成方式，是否就一定说明 action distribution 是 multimodal？

### 解析

这个说法方向正确，但还不够严谨。

真正讨论的是：

> 在相同或高度相似的 condition 下，数据是否存在多个明显不同的合理 action 或 action trajectory。

因此更准确地说是：

```text
相同 observation / condition
↓
对应多个彼此分离的合理动作模式
↓
Multimodal conditional action distribution
```

仅仅“最终目标一样”还不够。

例如两个机器人起始位置完全不同，即使最后都抓同一个杯子，它们对应的 observation 已经不同，就不能简单用这个例子证明同一个条件分布是多模态的。

---

# 3. 为什么直接 MSE 回归会产生“平均动作”

## 3.1 一个最简单的一维例子

假设在同一个 observation 下，专家动作只有两个可能值：

$$
a_1=-1
$$

表示向左绕。

以及：

$$
a_2=+1
$$

表示向右绕。

如果模型被要求输出唯一的确定性动作：

$$
\hat a=f_\theta(o)
$$

并使用 MSE：

$$
L=
(\hat a-a_1)^2
+
(\hat a-a_2)^2
$$

代入两个专家动作：

$$
L=
(\hat a+1)^2
+
(\hat a-1)^2
$$

展开：

$$
L=
2\hat a^2+2
$$

最小值出现在：

$$
\hat a=0
$$

于是模型学习到：

```text
专家模式 1：向左
专家模式 2：向右

MSE 最优解：
         ↓
       中间值
```

但是这个中间值可能恰好对应：

```text
直接撞向障碍物
```

于是出现一个非常典型的问题：

> 两个专家动作都是正确的，它们的平均值却可能是错误的。

---

## 3.2 MSE 为什么会这样

从统计意义上看，如果使用确定性函数并最小化平方误差，其最优预测倾向于条件均值：

$$
f^*(o)
=
\mathbb E[A\mid o]
$$

对于单峰分布，这通常没有问题。

但如果条件分布是：

```text
左模式            右模式
  /\                /\
 /  \              /  \
/    \____________/    \
          ↑
        条件均值
```

那么条件均值甚至可能落在数据密度很低的位置。

也就是说：

> 模型输出了一个“数学上折中”的答案，却不是一个真实专家会选择的答案。

---

## 3.3 MSE 并不是错误方法

需要特别强调：

> MSE 本身不是机器人策略中的错误方法。

如果动作分布接近 unimodal，直接回归往往非常有效。

例如：

```text
0.48
0.50
0.51
0.52
```

这种数据的均值依然是合理行为。

因此真正的问题不是：

```text
MSE 不适合机器人
```

而是：

```text
确定性单点回归
+
明显的多模态动作分布
↓
可能产生不合理平均动作
```

---

## 阶段思考 B：MSE 到底出了什么问题？

### 问题 1

假设同一个 observation 下，专家动作分别为：

$$
a=-2
$$

和：

$$
a=+2
$$

确定性 MSE 回归最容易预测什么？

### 解析

最优输出为：

$$
a=0
$$

但问题不在于数值 0 本身，而在于：

> 0 可能根本不是专家动作分布中的高概率区域。

---

### 问题 2

“Diffusion 不会产生平均动作，因为它不用 MSE。”

这个说法正确吗？

### 解析

不正确。

Diffusion 训练中非常常见的一个损失仍然是 MSE：

$$
L=
\left\|
\epsilon-
\epsilon_\theta
\right\|^2
$$

关键区别是：

> 它不是直接用 MSE 从 observation 回归最终 action。

它使用 MSE 学习不同噪声等级下的去噪信息，再通过一个随机生成过程建模整个分布。

所以应当区分：

```text
直接 Action Regression
≠
Noise Prediction
```

即使两者都可以使用 MSE，它们的建模对象和生成机制仍然完全不同。

---

# 4. Diffusion 的核心思想

如果复杂动作分布很难直接生成，一种思路是：

> 能不能先把复杂分布变成一种非常简单、非常容易采样的分布，然后再学习如何逆转这个过程？

Diffusion 采用的简单分布通常是 Gaussian distribution。

整体过程分成两个方向：

```text
真实动作
↓
Forward Process
↓
Gaussian Noise


Gaussian Noise
↓
Reverse Process
↓
真实动作分布
```

其中：

- Forward Process 是人为设计的；
- Reverse Process 是模型需要学习的。

可以把整个思想概括成：

$$
\text{Data}
\longrightarrow
\text{Noise}
\longrightarrow
\text{Data}
$$

真正有学习价值的是后半段。

---

# 5. Forward Process：为什么要故意破坏正确动作

## 5.1 从干净动作开始

设一个专家 Action Chunk 为：

$$
A_0
$$

例如：

$$
A_0\in\mathbb R^{20\times7}
$$

这里可以理解为：

- Action Chunk 长度 H = 20；
- 每个时间步 action dimension 为 7。

于是整个动作轨迹包含：

```text
20 个机器人时间步
×
每步 7 维动作
```

Forward Process 要做的是：

$$
A_0
\rightarrow
A_1
\rightarrow
A_2
\rightarrow
\cdots
\rightarrow
A_T
$$

随着 diffusion timestep 增大，动作中的真实结构逐渐被 Gaussian noise 破坏。

最终：

$$
A_T
\approx
\mathcal N(0,I)
$$

---

## 5.2 一步 Forward Diffusion

一步标准形式可以写成：

$$
q(A_t\mid A_{t-1})
=
\mathcal N
\left(
\sqrt{1-\beta_t}A_{t-1},
\beta_t I
\right)
$$

也可以从采样形式理解：

$$
A_t
=
\sqrt{1-\beta_t}A_{t-1}
+
\sqrt{\beta_t}\epsilon
$$

其中：

- A_{t-1} 是上一 diffusion step 的动作数据；
- A_t 是加入更多噪声后的动作数据；
- β_t 控制当前 step 加入多少噪声；
- ε 是 Gaussian noise；
- I 是单位协方差对应的单位矩阵。

并且：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

这个公式的直觉只有一句话：

```text
新的样本
=
保留一部分原信号
+
加入一部分随机噪声
```

---

## 5.3 βₜ 与 αₜ 的关系

定义：

$$
\alpha_t
=
1-\beta_t
$$

因此：

- βₜ 更直接表示当前 step 加多少 noise；
- αₜ 更接近当前 step 保留多少 signal。

一个常见混淆是：

> “βₜ 是原始信号保留比例。”

这个说法并不准确。

更准确的关系是：

```text
βₜ ↑
↓
当前 step 添加的噪声更多
↓
信号保留比例相应更低
```

而 αₜ 才直接满足：

$$
\alpha_t=1-\beta_t
$$

---

# 6. 不需要真的一步一步加噪

如果训练时必须：

```text
A₀
→ A₁
→ A₂
→ A₃
→ ...
→ Aₜ
```

才能得到某个噪声等级的样本，训练会非常低效。

Diffusion 有一个重要性质：

> 可以直接从 A₀ 构造任意 timestep 的 Aₜ。

定义：

$$
\bar\alpha_t
=
\prod_{s=1}^{t}\alpha_s
$$

则：

$$
A_t
=
\sqrt{\bar\alpha_t}A_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

其中：

- A₀ 是干净专家动作；
- Aₜ 是第 t 个 diffusion timestep 的 noisy action；
- ε 是随机采样的 Gaussian noise；
- ᾱₜ 决定原始信号保留程度。

这个公式可以直接看成：

$$
A_t
=
\text{Signal Component}
+
\text{Noise Component}
$$

其中：

$$
\sqrt{\bar\alpha_t}A_0
$$

代表保留下来的真实动作成分。

而：

$$
\sqrt{1-\bar\alpha_t}\epsilon
$$

代表噪声成分。

---

# 7. 两种完全不同的时间

Diffusion Policy 中最容易混淆的概念之一，是机器人时间和 diffusion timestep。

## 7.1 Robot Time

机器人动作序列可以写成：

$$
A
=
[a_\tau,a_{\tau+1},...,a_{\tau+H-1}]
$$

这里 τ 表示机器人真实执行过程中的时间。

例如：

```text
第 1 个动作
第 2 个动作
第 3 个动作
...
第 20 个动作
```

这属于机器人控制时间轴。

---

## 7.2 Diffusion Time

同一个 Action Chunk 又会经历：

$$
A_0
\rightarrow
A_1
\rightarrow
...
\rightarrow
A_T
$$

这个下标表示：

> 当前 Action Chunk 被 noise 污染到了什么程度。

它不是机器人运行了多少秒。

为了避免混乱，也可以写成：

$$
A^{(k)}
$$

其中 k 专门表示 diffusion timestep。

于是：

```text
Robot timestep
描述轨迹内部不同动作的位置

Diffusion timestep
描述同一整段轨迹当前被加噪或去噪到什么程度
```

两者完全不同。

---

## 阶段思考 C：Forward Process

### 问题 1

Forward diffusion 的方向是什么？

### 解析

正确方向是：

```text
Action
→
Noise
```

即：

$$
A_0
\rightarrow
A_T
$$

它不是生成过程，而是人为设计的数据破坏过程。

---

### 问题 2

对于：

$$
A_0\in\mathbb R^{20\times7}
$$

加噪之后 Aₜ 的 shape 会变化吗？

### 解析

通常不会。

仍然是：

$$
A_t\in\mathbb R^{20\times7}
$$

改变的是数值内容，而不是 tensor shape。

也就是说：

```text
干净轨迹：20 × 7
↓
轻度 noisy：20 × 7
↓
严重 noisy：20 × 7
↓
几乎纯噪声：20 × 7
```

---

### 问题 3

为什么最终希望 A_T 接近标准 Gaussian？

### 解析

因为推理时需要一个容易获得的起点。

标准 Gaussian：

$$
\mathcal N(0,I)
$$

非常容易随机采样。

因此可以：

```text
无需专家动作
↓
直接随机采 Gaussian noise
↓
启动 reverse generation
```

这也是 Diffusion 能作为生成模型工作的关键。

---

# 8. Reverse Process：从 Noise 重新得到 Action

Forward Process 本身不会产生机器人策略。

真正的目标是学习：

$$
A_T
\rightarrow
A_{T-1}
\rightarrow
...
\rightarrow
A_0
$$

这就是 **Reverse Process**。

最直观的理解是：

> 训练一个去噪器。

---

# 9. Noise Prediction：模型究竟学什么

由 Forward Process：

$$
A_t
=
\sqrt{\bar\alpha_t}A_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

训练时有三个重要量是已知的：

- 干净专家动作 A₀；
- 随机 timestep t；
- 自己随机采样的 noise ε。

因此，可以人为构造 Aₜ。

接下来训练神经网络：

$$
\epsilon_\theta(A_t,t)
$$

让它输出：

$$
\hat\epsilon
$$

并希望：

$$
\hat\epsilon
\approx
\epsilon
$$

也就是说：

> 模型学习识别当前 noisy action 中包含了什么噪声。

---

## 9.1 为什么 ground-truth noise 是已知的

因为 ε 并不是环境里未知产生的随机扰动。

训练时它是人为采样的：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

因此构造：

$$
A_t
=
\sqrt{\bar\alpha_t}A_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

之后：

- noisy sample 已知；
- timestep 已知；
- 真正加进去的 noise 也已知。

这使监督学习成为可能。

---

## 9.2 Noise Prediction Loss

最常见的一种训练目标为：

$$
L
=
\left\|
\epsilon
-
\epsilon_\theta(A_t,t)
\right\|^2
$$

这实际上仍然是 MSE。

但是这里预测的是 noise，而不是直接预测最终动作。

这也是理解 Diffusion 最关键的概念之一。

---

# 10. 为什么预测 Noise 可以恢复 Action

从：

$$
A_t
=
\sqrt{\bar\alpha_t}A_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

可以形式上解出 A₀：

$$
A_0
=
\frac{
A_t
-
\sqrt{1-\bar\alpha_t}\epsilon
}{
\sqrt{\bar\alpha_t}
}
$$

如果模型能够预测：

$$
\hat\epsilon
=
\epsilon_\theta(A_t,t)
$$

就可以利用预测 noise 推断干净数据的方向。

直觉上可以记成：

```text
当前 noisy action
-
模型识别出的 noise
↓
更接近 clean action
```

真实 Diffusion 的 reverse update 比简单减法更严格，但作为机器人学习阶段的心智模型，这个理解已经足够。

---

# 11. 为什么要多步去噪

如果 A_T 已经接近纯随机噪声，那么直接要求模型完成：

$$
A_T
\rightarrow
A_0
$$

是一个非常困难的问题。

Diffusion 将它拆解成：

$$
A_T
\rightarrow
A_{T-1}
\rightarrow
A_{T-2}
\rightarrow
...
\rightarrow
A_0
$$

于是每一步只需要完成一个较小的修正。

可以理解为：

```text
纯噪声
↓
出现非常粗略的动作结构
↓
轨迹方向逐渐明确
↓
时间连续性逐渐恢复
↓
形成合理动作
```

因此，与其说“多步去噪更稳定”，更准确的表述是：

> 多步 reverse process 把一个非常困难的全局生成问题拆成了一系列较容易的局部去噪问题。

---

## 阶段思考 D：Reverse Process

### 问题 1

Diffusion 模型训练时一定是在预测最终 action 吗？

### 解析

不是。

一种经典形式预测的是：

$$
\epsilon
$$

也就是 noise。

需要区分：

```text
传统 deterministic BC
Observation → Action

Diffusion noise prediction
Noisy Action + timestep → Noise
```

---

### 问题 2

为什么训练时可以直接监督 noise prediction？

### 解析

因为 noise 是人为采样并加进去的，因此 ground truth 已知。

完整因果链：

```text
从 dataset 取 A₀
↓
自己采样 ε
↓
自己构造 Aₜ
↓
当然知道真实 ε
↓
可以监督模型预测 ε
```

---

### 问题 3

为什么 Diffusion 中用了 MSE，却仍能表达 multimodal action distribution？

### 解析

关键不在损失函数名字，而在整个生成建模方式。

传统方法：

$$
o
\rightarrow
A
$$

直接把 observation 压成一个确定输出。

Diffusion 则是：

```text
Random Noise
+
Condition
↓
多步生成过程
↓
一个动作样本
```

因此最终建模的是：

$$
p(A\mid o)
$$

而不是单纯的条件均值。

---

# 12. 为什么机器人还需要 Conditional Diffusion

到目前为止，模型写成：

$$
\epsilon_\theta(A_t,t)
$$

仍然有一个严重问题：

> 它不知道机器人当前正在面对什么环境。

例如：

```text
场景 A：红杯在左侧
场景 B：红杯在右侧
```

即使 noisy action 完全一样，两种场景要求的正确去噪方向也完全不同。

因此机器人策略必须使用 **Conditional Diffusion**。

模型更合理的形式是：

$$
\epsilon_\theta(A_t,t,c)
$$

其中 c 表示条件信息。

---

# 13. Condition 可以包含什么

在机器人学习中，condition 常见包括：

$$
c=
\{I,s,l\}
$$

其中：

- I 表示 camera image；
- s 表示 robot state / proprioception；
- l 表示 language instruction。

因此真正希望学习的是：

$$
p(A\mid I,s,l)
$$

即：

> 在当前视觉、机器人状态和语言任务下，哪些动作轨迹是合理的？

这已经非常接近后续 VLA 的形式。

---

# 14. Condition 如何进入模型

具体架构因论文而异，但可以先建立通用数据流。

视觉：

```text
Image
↓
Visual Encoder
↓
Visual Feature
```

机器人状态：

```text
Proprioception
↓
MLP / State Encoder
↓
State Feature
```

语言：

```text
Language Instruction
↓
Language Encoder
↓
Language Feature
```

这些信息形成 condition：

$$
c=f(z_I,z_s,z_l)
$$

然后用于：

$$
\epsilon_\theta(A_t,t,c)
$$

也就是：

> 让 denoising model 不只是问“这里有哪些 noise”，而是问“在当前机器人任务条件下，这个 noisy trajectory 应该朝哪个方向去噪”。

---

# 15. 为什么 timestep 也必须告诉模型

如果只输入：

$$
A_t
$$

模型无法可靠判断当前数据的 noise level。

同一个数值形态可能对应：

- 很早期的轻度污染；
- 很后期的严重污染。

因此模型还需要 timestep t。

模型实际学习的是：

$$
\epsilon_\theta(A_t,t,c)
$$

直觉为：

```text
Aₜ
告诉模型：当前 noisy action 长什么样

t
告诉模型：当前脏到什么程度

c
告诉模型：最终应该变成什么类型的动作
```

三个信息缺一不可。

---

# 16. Random Noise 与 Condition 的分工

这是 Conditional Diffusion 中一个非常重要的心智模型。

Random noise 负责：

> 提供生成过程的随机性和 diversity。

Condition 负责：

> 把生成结果限制在符合当前任务和环境的合理区域。

可以简化成：

```text
Noise
↓
决定“从哪里开始采样”

Condition
↓
决定“允许往哪里收敛”
```

例如任务是：

```text
抓取右侧红杯
```

不同 noise 可能生成：

- 从左侧接近红杯；
- 从右侧接近红杯；
- 手腕姿态略有差异的抓取轨迹。

但是合理模型不应该因为 noise 不同，就突然生成：

```text
去抓左侧蓝杯
```

因为 condition 对生成结果施加了任务约束。

---

## 阶段思考 E：Conditional Diffusion

### 问题 1

如果模型只输入 Aₜ 和 timestep，而不输入 observation，会发生什么？

### 解析

模型只能学习：

> 一般意义上“什么轨迹像真实动作”。

却无法知道：

- 当前物体在哪里；
- 机器人目前是什么姿态；
- 任务要求抓哪个物体。

因此它无法成为一个真正可执行的机器人 policy。

---

### 问题 2

Random noise 为什么不会让机器人完全随机行动？

### 解析

因为 noise 不是唯一输入。

完整生成受 condition 引导：

$$
p(A\mid c)
$$

因此 noise 提供多样性，而 condition 限制行为必须符合当前任务。

---

### 问题 3

Diffusion Policy 更接近哪一种？

$$
A=f(o)
$$

还是：

$$
A\sim p(A\mid o)
$$

### 解析

更接近第二种。

其核心不是输出唯一的 point estimate，而是从条件动作分布中生成一个合理样本。

---

# 17. Noise Schedule 的作用

一整套：

$$
\beta_1,\beta_2,\dots,\beta_T
$$

称为 **Noise Schedule**。

它决定：

> 动作数据如何随着 diffusion timestep 从真实轨迹逐步过渡到 Gaussian noise。

例如可以粗略理解为：

```text
Diffusion timestep
0 -------------------------------- T

Noise level
低 ------------------------------- 高
```

Noise schedule 的具体形式会影响训练和生成质量。

但在本章阶段，无需深入比较 linear schedule、cosine schedule 等具体方案。

需要掌握的是：

> Noise schedule 决定不同 diffusion timestep 下 signal 与 noise 的混合程度。

---

# 18. Score 的定位

Diffusion 文献中还经常出现 **Score Function**。

形式通常与：

$$
\nabla_x \log p_t(x)
$$

有关。

本章不需要展开 score-based generative modeling 的完整概率论。

只需要建立一个直觉：

> Score 描述当前 noisy sample 朝哪个方向移动，会更接近高概率的数据区域。

因此可以粗略理解：

```text
当前位置
↓
Score
↓
告诉你朝什么方向更像真实数据
```

Noise prediction 与 score prediction 在不同 diffusion formulation 中存在紧密联系。

对于机器人策略学习阶段，重要的是知道：

> 它们都在服务于 reverse generation，即让 noisy trajectory 一步步回到合理动作分布。

---

# 19. 为什么 Diffusion 能表达 Multimodal Trajectory

现在可以把前面的所有知识连接起来。

假设：

$$
p(A\mid o)
$$

有两个主要模式。

一个是：

```text
左绕障碍物
```

另一个是：

```text
右绕障碍物
```

推理 1：

$$
A_T^{(1)}
\sim
\mathcal N(0,I)
$$

经过 conditional reverse process：

$$
A_T^{(1)}
\rightarrow
...
\rightarrow
A_0^{(1)}
$$

可能最终生成左绕轨迹。

推理 2：

$$
A_T^{(2)}
\sim
\mathcal N(0,I)
$$

经过同一个模型：

$$
A_T^{(2)}
\rightarrow
...
\rightarrow
A_0^{(2)}
$$

可能最终生成右绕轨迹。

因此：

> 模型不必求出两种动作的平均值，而可以采样其中某一个合理模式。

这就是 Diffusion 作为机器人生成式策略的核心价值。

---

# 20. 为什么特别适合 Action Chunk

考虑一个 Action Chunk：

$$
A
\in
\mathbb R^{H\times d_a}
$$

例如：

$$
A
\in
\mathbb R^{20\times7}
$$

这不是 140 个相互独立的数字。

它内部存在大量结构。

例如一个抓取过程可能是：

```text
接近物体
↓
调整末端姿态
↓
继续靠近
↓
闭合夹爪
↓
抬起
```

这些动作之间存在明显的 temporal correlation。

例如：

> gripper close 通常不能发生在机械臂尚未接近物体的时候。

因此 Action Chunk 是一种：

> 高维、连续、具有时间结构的数据。

Diffusion 的优势正是能够建模这种复杂连续联合分布。

---

# 21. 一个完整 Pick-and-Place 实例

现在把整个机制放进一个完整机器人任务。

任务：

```text
Pick up the red box and place it in basket B.
```

当前机器人观测：

```text
RGB image
+
joint state
+
language instruction
```

首先编码 condition：

$$
c=
\{I,s,l\}
$$

目标动作：

$$
A_0
\in
\mathbb R^{20\times7}
$$

---

## 21.1 Training

从真实 demonstration 中取一条专家动作轨迹：

$$
A_0
$$

随机采样 diffusion timestep：

$$
t
$$

随机采样 Gaussian noise：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

构造 noisy action：

$$
A_t
=
\sqrt{\bar\alpha_t}A_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

送入模型：

$$
\hat\epsilon
=
\epsilon_\theta(A_t,t,c)
$$

然后优化：

$$
L
=
\left\|
\epsilon-\hat\epsilon
\right\|^2
$$

完整数据流：

```text
Expert Action Chunk A₀
          +
Current Observation
          ↓

Random timestep t
          +
Random Gaussian noise ε
          ↓

Construct noisy action Aₜ
          ↓

Conditional Denoising Model
          ↓

Predict ε̂
          ↓

Compare ε̂ with ε
          ↓

Backpropagation
```

---

## 21.2 Inference

部署时没有：

$$
A_0
$$

因此首先采样：

$$
A_T
\sim
\mathcal N(0,I)
$$

然后结合当前 condition：

$$
c=
\{I,s,l\}
$$

执行：

$$
A_T
\rightarrow
A_{T-1}
\rightarrow
...
\rightarrow
A_0
$$

最后获得：

$$
A_0
=
[a_\tau,a_{\tau+1},...,a_{\tau+19}]
$$

这就是一段可以交给机器人控制系统执行的 Action Chunk。

---

# 22. Diffusion 在机器人控制系统中的位置

Diffusion Policy 并不等于整个机器人控制系统。

一个典型层级仍然可以写成：

```text
Task / Language
↓
Vision + Robot State
↓
Learned Policy
↓
Diffusion Action Generation
↓
Action Chunk
↓
Robot Controller
↓
Joint / EEF Command
↓
Robot
```

因此 Diffusion 主要解决的是：

> Learned Policy 内部如何生成复杂连续动作轨迹。

它通常不直接负责：

- 电机 PWM；
- 最底层电流环；
- 电机驱动；
- 高频关节稳定控制。

这些仍属于传统机器人控制系统。

从控制视角看，Diffusion 更接近：

```text
trajectory / command generation
```

而不是最底层 actuator control。

---

# 23. Diffusion 与 ACT 的关系

ACT 和 Diffusion 都可以生成 Action Chunk，但生成机制明显不同。

## 23.1 ACT

可以粗略理解为：

```text
Observation
↓
Transformer
↓
Action Chunk
```

一次主要 forward computation 即得到动作预测。

---

## 23.2 Diffusion

则更接近：

```text
Observation
+
Random Action Noise
↓
Denoising
↓
Denoising
↓
Denoising
↓
Action Chunk
```

因此二者的核心差异并不只是：

```text
Transformer vs Diffusion
```

更重要的是建模思想。

ACT 更接近：

> 直接预测动作序列。

Diffusion 更强调：

> 条件动作分布的生成。

---

# 24. Diffusion 的主要优势

对于机器人策略而言，本章已经可以总结出三个主要优势。

第一，它能自然处理 multimodal action distribution。

第二，它适合高维连续 Action Chunk。

第三，它能够联合建模整个轨迹的 temporal structure。

因此特别适合：

```text
Visuomotor Policy
+
Continuous Action Generation
+
Trajectory Generation
```

---

# 25. Diffusion 的核心代价：Inference Latency

Diffusion 的主要问题也非常明显：

> 它需要多次 neural network inference。

ACT 更接近：

$$
o
\rightarrow
A
$$

而 Diffusion 是：

$$
A_T
\rightarrow
A_{T-1}
\rightarrow
...
\rightarrow
A_0
$$

机器人控制却天然强调：

```text
感知
↓
快速决策
↓
执行
↓
重新感知
```

如果策略推理时间过长：

```text
观察已经过时
↓
机器人或环境已经变化
↓
生成出来的动作可能 stale
```

因此，生成质量和实时性之间存在天然矛盾。

这也是之后从 Diffusion Policy 继续发展到 Flow Matching、现代 VLA action expert 和 Real-Time VLA 时反复出现的问题。

---

# 26. 常见误区与阶段纠偏

## 26.1 “同一个目标有多种方法，所以一定是多模态”

不够准确。

更严谨的是：

> 同一个或高度相似的 condition 下，存在多个彼此分离的合理动作模式。

---

## 26.2 “MSE 不适合机器人”

错误。

MSE 在接近单峰的动作分布中完全可以工作得很好。

真正的问题是：

```text
Multimodal Target
+
Deterministic Point Regression
↓
Averaging Problem
```

---

## 26.3 “Diffusion 不用 MSE”

错误。

Noise prediction 常见损失就是：

$$
L=
\left\|
\epsilon-\epsilon_\theta
\right\|^2
$$

Diffusion 能表达多模态不是因为“没有 MSE”，而是因为：

> 它通过随机 latent/noise 和 iterative generation 建模整个条件分布。

---

## 26.4 “βₜ 表示保留多少 signal”

不准确。

βₜ 更直接表示：

> 当前 diffusion step 加多少 noise。

而：

$$
\alpha_t
=
1-\beta_t
$$

才对应 signal 保留比例。

---

## 26.5 “Diffusion timestep 就是机器人 timestep”

错误。

Robot timestep：

```text
描述轨迹里的第几个动作
```

Diffusion timestep：

```text
描述当前整条 trajectory 被污染到什么程度
```

两者属于完全不同的时间轴。

---

## 26.6 “Forward Process 是模型学出来的”

错误。

Forward Process 通常是人为规定的 noise corruption process。

模型真正学习的是：

> Reverse Process 所需要的去噪信息。

---

## 26.7 “Diffusion 一次直接从 noise 得到动作”

经典 Diffusion 通常不是。

它通过多步：

$$
A_T
\rightarrow
A_{T-1}
\rightarrow
...
\rightarrow
A_0
$$

逐步生成。

---

## 26.8 “Random noise 会导致机器人随机乱动”

不准确。

Noise 提供 diversity。

Condition 提供 task constraint。

真正生成的是：

$$
p(A\mid c)
$$

而不是无条件随机动作。

---

# 27. 在 VLA 知识链中的位置

本章建立的是“生成式动作策略”的基础。

完整关系可以理解为：

```text
Robot Policy
↓
Behavioral Cloning
↓
Distribution Shift
↓
Action Representation
↓
Action Chunking
↓
ACT
↓
Multimodal Action Distribution
↓
Diffusion
↓
Diffusion Policy
↓
Flow Matching
↓
π0 / π0.5 等现代 VLA
```

其中最重要的转折是：

```text
直接预测一个动作
↓
直接预测一段动作
↓
建模一整个动作分布
```

Diffusion 对应第三步。

---

# 28. 阅读机器人 Diffusion 论文时应关注什么

当后续看到 Diffusion Policy 或相关 generative policy 时，不应只问：

> 它是不是用了 diffusion？

而应该主动检查以下问题。

首先看生成对象：

```text
Single Action
还是
Action Chunk
```

其次看 condition：

```text
Image？
Robot State？
Language？
多帧 Observation？
```

再看 action shape：

```text
H × dₐ
```

然后检查：

```text
Prediction Horizon
Execution Horizon
Observation Horizon
```

还应关注：

```text
Diffusion steps 有多少？
Inference latency 多大？
使用什么 denoising network？
Condition 怎么注入？
Action 是否 normalization？
如何处理 multimodal demonstrations？
```

这些问题会在下一章 Diffusion Policy 中正式展开。

---

# 29. 综合思考

## 综合题 1

一个数据集里，针对几乎相同的 observation，有一半 demonstration 从左侧绕障碍物，另一半从右侧绕障碍物。

为什么 deterministic MSE policy 可能失败？

### 解析

因为它倾向于条件均值。

如果左、右两种行为形成明显分离模式：

```text
左轨迹
+
右轨迹
↓
平均轨迹
```

平均轨迹可能正好经过障碍物。

---

## 综合题 2

为什么 Diffusion 可以避免简单平均？

### 解析

因为它试图学习：

$$
p(A\mid o)
$$

并从该分布中采样。

不同 random noise 可以对应不同合理 mode。

因此：

```text
Noise 1
→ 左侧轨迹

Noise 2
→ 右侧轨迹
```

而不是：

```text
左 + 右
→ 中间平均
```

---

## 综合题 3

为什么训练时要先把正确 demonstration 故意破坏？

### 解析

因为我们想人为构造大量：

```text
Noisy Action
→
Correct Denoising Direction
```

的监督数据。

通过已知 noise：

$$
\epsilon
$$

训练网络学习 reverse generation 所需的信息。

---

## 综合题 4

如果没有 condition，Diffusion 是否仍然能生成“像机器人”的动作？

### 解析

可能可以生成动作分布的统计结构，但无法成为真正的任务策略。

因为它不知道：

```text
当前场景
机器人状态
目标物体
语言任务
```

因此它无法回答：

> 当前到底应该生成哪一种动作。

---

## 综合题 5

为什么 Diffusion 特别适合 Action Chunk，而不只是单步动作？

### 解析

因为 Action Chunk 是一个高维联合分布，内部存在：

- 时间相关性；
- 关节相关性；
- 位姿与夹爪动作协调关系；
- 多种合理轨迹模式。

Diffusion 可以直接生成整个结构化 trajectory。

---

## 综合题 6

Diffusion 最大的工程缺点是什么？

### 解析

典型问题是推理速度。

多步 denoising 意味着：

```text
一次控制决策
↓
需要多次网络前向
↓
Latency 增大
```

真实机器人环境变化后，旧 observation 可能变成 stale observation。

因此实时性成为 Diffusion Policy 和 VLA 部署中的重要研究问题。

---

# 30. 本章最终心智模型

完成这一章后，应当形成下面这一整套连续理解。

机器人 demonstration 并不总是满足：

$$
A=f(o)
$$

因为同一个 observation 下可能存在多个合理动作模式。

因此更自然的表示是：

$$
A\sim p(A\mid o)
$$

直接 deterministic MSE regression 容易趋向条件均值，在多模态情况下可能生成不存在于真实数据中的“平均动作”。

Diffusion 的处理方式是：

```text
复杂 Action Distribution
↓
人为 Forward Noise Process
↓
Gaussian Noise
```

然后训练模型学习 reverse generation。

训练时：

$$
A_t
=
\sqrt{\bar\alpha_t}A_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

模型预测：

$$
\hat\epsilon
=
\epsilon_\theta(A_t,t,c)
$$

优化：

$$
L
=
\left\|
\epsilon-\hat\epsilon
\right\|^2
$$

部署时：

$$
A_T
\sim
\mathcal N(0,I)
$$

然后：

$$
A_T
\rightarrow
A_{T-1}
\rightarrow
...
\rightarrow
A_0
$$

最终：

$$
A_0
$$

就是一段符合当前 condition 的机器人 Action Chunk。

所以整章最核心的一句话是：

> **Diffusion Policy 不只是预测“下一步应该做什么”，而是在当前 observation 的条件下，从复杂动作分布中生成一条合理的连续动作轨迹。**

---

# 31. 本章关键词

**Multimodal Distribution — 多模态分布 — 同一 condition 下存在多个彼此不同的合理动作模式**

**Deterministic Regression — 确定性回归 — 给定输入后直接输出唯一动作**

**Averaging Problem — 平均动作问题 — 多模态目标经过单点回归后可能得到不合理的条件均值**

**Gaussian Noise — 高斯噪声 — Diffusion 中用于构造简单起始分布的随机噪声**

**Forward Process — 前向过程 — 将真实数据逐渐破坏成 Gaussian noise**

**Reverse Process — 反向过程 — 从 noise 逐步恢复到真实数据分布**

**Denoising — 去噪 — 将 noisy sample 朝真实数据方向修正**

**Noise Prediction — 噪声预测 — 让模型预测人工加入的 Gaussian noise**

**Diffusion Timestep — 扩散时间步 — 描述当前数据被污染或去噪到什么程度**

**Robot Timestep — 机器人时间步 — 描述实际 trajectory 中动作执行的时间位置**

**Noise Schedule — 噪声调度 — 决定不同 diffusion timestep 的噪声强度**

**Conditional Diffusion — 条件扩散 — 在 observation、state、language 等条件下进行生成**

**Condition — 条件信息 — 用于约束当前应该生成哪一类机器人动作**

**Score — 分数函数 — 表示 noisy sample 朝高概率数据区域移动的方向信息**

**Action Chunk — 动作块 — 一次生成的一段连续机器人动作序列**

**Trajectory Distribution — 轨迹分布 — 对整段未来动作序列进行概率建模**

---

# 32. 本章结论

本章需要真正留下的不是一组公式，而是五个连续结论：

1. 机器人任务经常具有多种合理解，因此 action distribution 可能是 multimodal。
2. Deterministic MSE regression 在多模态条件下可能产生并不存在于真实数据中的平均动作。
3. Diffusion 先人为把真实 Action Chunk 加噪成 Gaussian noise，再学习反向去噪过程。
4. Conditional Diffusion 通过 image、robot state、language 等信息约束去噪方向，因此可以成为机器人 policy。
5. Diffusion 能够生成高维、连续、具有多种模式的动作轨迹，但 iterative denoising 会带来明显 inference latency。

至此，Diffusion 已经从“生成模型概念”转化为了一个机器人策略层面的完整心智模型。

下一阶段进入 **Diffusion Policy** 时，不再需要重新解释“为什么要加噪、为什么能去噪”。重点将转向：

```text
Observation Horizon
↓
Conditional Action Generation
↓
Prediction Horizon
↓
Action Horizon / Execution Horizon
↓
Receding-Horizon Control
↓
真实机器人闭环执行
```

也就是正式回答：

> **Diffusion 如何从一个生成模型，变成真正可以驱动机器人运行的 visuomotor policy。**
