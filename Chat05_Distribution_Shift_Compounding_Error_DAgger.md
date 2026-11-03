# 05｜Distribution Shift、Compounding Error 与 DAgger

## 1. 本章导言：为什么“会模仿”还不等于“会控制机器人”

Behavioral Cloning（BC）给出了机器人模仿学习中最直接的一条路线：收集专家演示数据，将 observation 与 expert action 配对，然后把策略训练成一个监督学习模型。

如果专家数据写成：

$$
D=
\left\{
(o_i,a_i^*)
\right\}_{i=1}^{N}
$$

那么 BC 希望得到一个策略：

$$
\pi_\theta(a\mid o)
$$

使得在专家数据中的 observation 上，模型输出尽量接近 expert action。

这一思路看起来非常自然，而且在训练集上往往能够得到很低的 loss。然而真实机器人部署之后，经常会出现一种反直觉现象：

> 模型在 demonstration dataset 上预测得很好，但一旦真正闭环运行，轨迹可能逐渐偏离，随后迅速恶化，最终失败。

问题并不只是“模型还有一点预测误差”，而是机器人策略具有一个普通静态监督学习中不存在的特性：

> **策略的输出会改变环境，而改变后的环境又会成为策略下一次的输入。**

因此机器人学习是一个 sequential decision making 问题。

一次动作预测误差并不会停留在当前 timestep，而可能改变后续整个 observation sequence。

本章由此围绕三条核心概念展开：

```text
Behavioral Cloning
↓
Distribution Shift
↓
Compounding Error
↓
DAgger
```

随后进一步分析 DAgger 的局限，并说明这一问题为什么会继续连接到：

```text
Human Intervention
Online RL
VLA Post-training
Agent Recovery
```

在整套 VLA 学习路线中，本章位于 Robot Learning 的核心基础阶段。学习手册将 Chat 05 的重点明确定位为 expert state distribution、learned policy state distribution、compounding error 与 DAgger，并要求进一步理解它们与 online RL、human intervention、VLA post-training 和 Agent recovery 的联系。

学完本章后，应当能够回答一个非常重要的问题：

> 为什么一个在离线专家数据上表现优秀的机器人策略，仍然可能在真实 closed-loop rollout 中迅速失败？

---

# 2. 问题从哪里来：BC 的训练世界和部署世界并不相同

## 2.1 从机械臂抓杯子开始

假设专家正在演示机械臂抓取一个杯子。

一条理想轨迹可能是：

```text
机械臂距离杯子较远
↓
移动到杯子上方
↓
末端下降
↓
调整夹爪位置
↓
闭合夹爪
↓
抬起杯子
```

用状态表示：

$$
s_0
\rightarrow
s_1
\rightarrow
s_2
\rightarrow
s_3
\rightarrow
s_4
$$

专家在每个状态给出动作：

$$
a_t^*
=
\pi_E(s_t)
$$

其中：

- π_E 表示 expert policy；
- s_t 表示时刻 t 的机器人状态；
- a_t* 表示专家在该状态下给出的正确动作。

收集大量这样的 demonstration 后，可以得到：

$$
D=
\left\{
(s_t,a_t^*)
\right\}
$$

训练 BC：

$$
\pi_\theta(s_t)
\approx
a_t^*
$$

如果机械臂使用 Cartesian delta action，动作可能类似：

```text
Δx
Δy
Δz
Δroll
Δpitch
Δyaw
gripper
```

此时 BC 做的事情非常明确：

> 给定专家曾经访问过的状态，预测专家当时采取的动作。

困难恰恰出现在“专家曾经访问过”这几个字上。

---

# 3. 一个小动作误差为什么会改变后面的所有问题

假设 expert 在某一时刻应该控制末端执行器向前移动：

```text
2.0 cm
```

而 learned policy 输出：

```text
1.8 cm
```

单看当前 timestep，只有 0.2 cm 的误差。

如果这是普通 regression，我们可能会认为：

> 误差很小，可以接受。

但是机器人并不是静态 regression。

专家执行：

$$
s_t
\xrightarrow{a_t^*}
s_{t+1}^*
$$

learned policy 执行：

$$
s_t
\xrightarrow{a_t}
s_{t+1}
$$

只要：

$$
a_t
\neq
a_t^*
$$

一般就可能得到：

$$
s_{t+1}
\neq
s_{t+1}^*
$$

因此下一次 policy 接收到的输入已经发生变化。

原本 expert 在下一时刻看到的可能是：

```text
夹爪位于杯子正上方
```

而 learned policy 看到的可能已经变成：

```text
夹爪略微偏离杯子中心
```

于是问题从：

> “在熟悉状态下动作有一点误差”

变成了：

> “下一步开始，模型可能正在处理训练数据中较少出现的状态。”

机器人策略由此形成一个闭环：

```text
Observation
↓
Policy
↓
Action
↓
Robot / Environment
↓
New Observation
↓
Policy
↓
...
```

也就是说：

> **Policy 不仅回答当前问题，它还参与制造下一道题。**

这是理解整个本章最重要的心智模型。

---

# 4. State Distribution：策略决定自己会访问哪些状态

为了准确描述这一现象，需要引入状态分布（state distribution）。

对于一个策略 π，可以用：

$$
d_\pi(s)
$$

表示：

> 使用策略 π 与环境交互时，机器人访问状态 s 的概率分布。

它不是一个单独的状态，而是描述：

> 这个 policy 在长期 rollout 中通常会走到哪里。

## 4.1 Expert State Distribution

专家策略运行时得到：

$$
d_{\pi_E}(s)
$$

专家通常会访问：

- 正确对准目标的状态；
- 正常接近物体的状态；
- 合理的抓取前姿态；
- 正常执行任务的轨迹附近。

如果 expert 很强，它访问的状态往往集中在一条比较窄的“成功走廊”中。

可以想象为：

```text
Expert trajectory

● → ● → ● → ● → ● → ●
```

专家很少主动访问：

```text
夹爪偏离目标 5 cm
杯子已经被碰歪
机械臂姿态错误
抓取失败之后的位置
```

因此这些状态在 demonstration dataset 中通常数量很少。

## 4.2 Learned Policy State Distribution

部署 BC policy 后，机器人访问的是：

$$
d_{\pi_\theta}(s)
$$

因为 learned policy 并不完美，所以它可能产生：

```text
正确状态
↓
轻微偏差
↓
更明显偏差
↓
训练数据较少覆盖的状态
↓
更大的策略误差
```

因此：

$$
d_{\pi_E}(s)
\neq
d_{\pi_\theta}(s)
$$

这就是 imitation learning 中最核心的 Distribution Shift 之一。

---

# 5. Distribution Shift：真正的问题不是“数据少”，而是“数据分布不同”

## 5.1 BC 实际优化的分布

BC 训练时的目标可以写成：

$$
L_{BC}
=
\mathbb{E}_{s\sim d_{\pi_E}}
\left[
\ell
\left(
\pi_\theta(s),
\pi_E(s)
\right)
\right]
$$

其中：

- d_πE 表示 expert 访问到的状态分布；
- π_θ 表示 learned policy；
- π_E 表示 expert policy；
- ℓ 表示单步 imitation loss。

例如连续动作下可能使用 MSE：

$$
\ell
=
\left\|
\pi_\theta(s)-a^*
\right\|_2^2
$$

注意 BC objective 中最关键的并不只是 loss，而是：

$$
s
\sim
d_{\pi_E}
$$

也就是说：

> 训练过程中模型主要在 expert 会访问的状态上接受监督。

## 5.2 部署时真正关心的分布

实际运行时，真正希望策略在以下分布上表现良好：

$$
s
\sim
d_{\pi_\theta}
$$

因此真正关心的是类似：

$$
\mathbb{E}_{s\sim d_{\pi_\theta}}
\left[
\ell
\left(
\pi_\theta(s),
\pi_E(s)
\right)
\right]
$$

于是训练与部署之间产生了 mismatch：

```text
Training

s ~ d_πE

Deployment

s ~ d_πθ
```

如果：

$$
d_{\pi_E}
\neq
d_{\pi_\theta}
$$

那么即使：

```text
Training Loss 很低
```

也不能保证：

```text
Closed-loop Rollout 很好
```

---

# 6. 阶段思考 A：为什么训练 loss 很低，robot rollout 仍然失败？

### 问题 1

d_πE(s) 与 d_πθ(s) 分别表示什么？

### 解析

前者表示 expert policy 运行时访问到的状态分布；后者表示 learned policy 自己 closed-loop rollout 时实际访问到的状态分布。

简单说：

```text
d_πE：老师通常走到哪里
d_πθ：学生自己走的时候通常走到哪里
```

### 问题 2

为什么一个在 demonstration dataset 上 loss 很低的 BC policy，部署后仍然可能失败？

### 解析

仅回答“因为模型仍然有预测误差”是不完整的。

完整因果链应为：

```text
存在小预测误差
↓
动作改变真实机器人状态
↓
下一时刻 observation 偏离 expert trajectory
↓
模型进入训练数据覆盖较弱的区域
↓
预测精度下降
↓
进一步产生状态偏移
```

因此问题的核心不是：

```text
单步误差不够小
```

而是：

```text
单步误差会改变未来输入分布
```

### 问题 3

为什么第 10 步只有 2 mm 的误差，并不能说明第 11 步仍然只会有 2 mm 的误差？

### 解析

因为第 11 步已经接收到新的状态。

第 10 步动作误差：

$$
a_{10}
\neq
a_{10}^*
$$

可能导致：

$$
s_{11}
\neq
s_{11}^*
$$

因此模型第 11 步面对的是新的输入条件。

预测误差不再是简单独立的：

```text
2 mm
+
2 mm
+
2 mm
```

而可能是一个不断变化的动态过程。

---

# 7. 为什么机器人学习比普通监督学习更容易受到 Distribution Shift 影响

考虑图像分类。

输入：

```text
Image
↓
Classifier
↓
Cat / Dog
```

如果模型将一只猫错误识别为狗，这个错误通常不会改变下一张输入图片。

下一张图片依旧由 dataset 或外部数据源提供。

因此通常不存在：

```text
当前预测
↓
改变下一次输入
```

这种闭环。

机器人 policy 不同。

机器人中：

$$
s_t
\rightarrow
\pi_\theta
\rightarrow
a_t
\rightarrow
\text{Environment}
\rightarrow
s_{t+1}
$$

所以：

$$
a_t
\rightarrow
s_{t+1}
$$

而下一次：

$$
a_{t+1}
=
\pi_\theta(s_{t+1})
$$

因此：

```text
当前 prediction
↓
改变 environment
↓
改变 next input
↓
改变 next prediction
```

这也是 sequential decision making 与普通 i.i.d. supervised learning 的关键区别之一。

---

# 8. Distribution Shift 与 Compounding Error 不是同一个概念

这是本章最容易混淆的一组概念。

## 8.1 Distribution Shift

Distribution Shift 强调：

> 训练时观察到的状态分布与部署时观察到的状态分布不同。

数学上：

$$
d_{\pi_E}(s)
\neq
d_{\pi_\theta}(s)
$$

它是一个“分布不一致”的概念。

## 8.2 Compounding Error

Compounding Error 强调：

> 早期动作误差改变了后续状态，而新的状态又使策略更容易产生新的误差，误差沿时间逐步传播和累积。

其典型链条为：

```text
Policy Error
↓
State Deviation
↓
Less Familiar State
↓
Larger Policy Error
↓
Further State Deviation
```

因此：

```text
Distribution Shift
```

更强调“模型现在遇到的输入和训练时不同”。

而：

```text
Compounding Error
```

更强调“错误如何随着 trajectory 演化不断放大”。

二者的关系通常可以理解为：

```text
初始动作误差
↓
状态开始偏移
↓
Distribution Shift
↓
模型进入不熟悉区域
↓
更容易犯错
↓
Compounding Error
```

---

# 9. 阶段纠偏：一个典型的不完整理解

一种常见表述是：

> “Distribution Shift 就是分布偏移带来的误差累积。”

这个回答方向相关，但把两个概念合并了。

更加准确的理解应该是：

```text
Distribution Shift
= 输入状态分布发生变化

Compounding Error
= 误差通过 sequential dynamics 持续累积
```

因此 Distribution Shift 可以成为 Compounding Error 的重要原因，但它们不是同义词。

阅读 Robot Learning 论文时，如果作者讨论：

```text
train-test state distribution mismatch
covariate shift
learner-induced states
```

重点通常在 Distribution Shift。

如果作者讨论：

```text
errors accumulate over horizon
long-horizon degradation
early mistakes affect future states
```

重点通常在 Compounding Error。

---

# 10. Compounding Error：为什么“小误差”能够演化成任务失败

假设一个 manipulation episode 共有 T 个 timestep。

BC policy 在 expert distribution 上每一步发生错误的概率为 ε。

例如：

```text
T = 100
ε = 1%
```

直觉可能会认为：

```text
100 × 1% ≈ 1
```

因此风险似乎不大。

但这个计算隐含了一个重要假设：

> 每一步错误彼此独立，而且当前错误不会影响未来状态。

这个假设对 closed-loop robot policy 通常并不成立。

## 10.1 第一次错误为什么特殊

假设前 20 步全部正确：

```text
1  2  3 ... 20
✓  ✓  ✓     ✓
```

第 21 步策略第一次明显出错：

```text
21
✗
```

那么第 22 步以后：

```text
22 23 24 ... T
?  ?  ?      ?
```

原因是：

$$
s_{22}
\neq
s_{22}^*
$$

此时 policy 可能已经离开 expert trajectory。

如果训练数据中对这一状态覆盖不足，那么下一步错误概率可能不再是原来的 ε。

因此第一次错误不仅产生：

```text
一个错误动作
```

还可能改变：

```text
之后所有动作的输入条件
```

## 10.2 早期错误为什么通常比晚期错误危险

假设 episode 长度为 100。

如果第 99 步出错，只剩下很短的未来 horizon。

而如果第 5 步出错：

```text
t = 5
```

后面还有约 95 个 timestep 可能受到影响。

因此可直觉理解为：

$$
\text{Damage}(t)
\propto
T-t
$$

这也是 long-horizon manipulation 特别困难的原因之一。

例如：

```text
拿起杯子
↓
移动到柜门
↓
打开柜门
↓
放入杯子
↓
调整位置
↓
关闭柜门
```

如果第一步抓取已经失败，那么后续所有 subtask 都失去了正常执行的前提。

---

# 11. O(Tε) 与 O(T²ε)：一个重要的二次检验

这里引入一个有助于理解 compounding error 的经典理论直觉。

## 11.1 如果错误完全不会影响未来

假设每一步独立发生错误，当前错误对下一状态完全没有影响。

那么整个 trajectory 的累计错误量级更接近：

$$
O(T\epsilon)
$$

原因很简单：

```text
T 个 timestep
×
每一步 ε 的错误概率
```

得到：

$$
T\epsilon
$$

## 11.2 如果早期错误能够污染未来

如果第 t 步发生错误，该错误可能影响余下：

$$
T-t
$$

个 timestep。

将不同时间发生错误的潜在影响加起来，可以得到类似：

$$
\epsilon
\left[
(T-1)
+
(T-2)
+
\cdots
+
1
\right]
$$

由于：

$$
1+2+\cdots+(T-1)
=
\frac{T(T-1)}{2}
$$

所以经典分析中会得到：

$$
O(T^2\epsilon)
$$

这一表达并不是说：

> 一个机器人一定会精确地产生 T²ε 个错误。

而是说明：

> **在经典 worst-case 分析中，BC 的 trajectory-level degradation 可能随着 horizon 呈现二次量级放大。**

真正重要的是它表达的机制：

```text
Error does not stay local in time.
```

---

# 12. 阶段思考 B：O(Tε) 还是 O(T²ε)？

### 问题

如果所有错误都只影响当前 timestep，完全不会改变未来状态，那么总误差更接近 O(Tε) 还是 O(T²ε)？

### 典型误区

容易因为刚刚学习 compounding error，就机械回答：

```text
O(T²ε)
```

但这是错误的。

### 正确解析

只有当错误能够通过系统动力学影响未来状态时，才存在 compounding。

如果每个 timestep 的错误完全独立：

$$
\text{Total Error}
\approx
T\epsilon
$$

因此是：

$$
O(T\epsilon)
$$

这道题真正检查的不是是否记住公式，而是是否理解：

> **T² 的来源是 temporal propagation，而不是“序列任务天然就是平方”。**

---

# 13. 从问题自然推导 DAgger

到这里，BC 的根本矛盾已经非常清楚：

训练：

$$
s
\sim
d_{\pi_E}
$$

部署：

$$
s
\sim
d_{\pi_\theta}
$$

那么最自然的解决思路就是：

> 为什么不直接让 learned policy 自己运行，观察它真正会访问什么状态，然后再让 expert 告诉它这些状态应该怎么处理？

这就是 DAgger。

DAgger 全称：

```text
Dataset Aggregation
```

核心流程是：

```text
Train
↓
Deploy
↓
Visit New States
↓
Expert Relabel
↓
Aggregate Dataset
↓
Retrain
```

---

# 14. DAgger 的核心：State 来自 Learner，Label 来自 Expert

假设最初拥有 expert demonstration：

$$
D_0
=
\left\{
(s_i,a_i^*)
\right\}
$$

在 D_0 上训练：

$$
\pi_1
=
\operatorname{Train}(D_0)
$$

接下来，不再只是继续让 expert 从正常初始状态反复演示，而是让：

$$
\pi_1
$$

自己 rollout。

于是采集：

$$
s
\sim
d_{\pi_1}
$$

这些 state 中会出现很多原始 expert dataset 中很少出现的状态，例如：

```text
夹爪偏右
夹爪下降过多
物体已经发生轻微移动
前一步动作执行不充分
抓取位置出现偏差
```

然后让 expert 对这些 learner-visited states 给出动作：

$$
a^*
=
\pi_E(s)
$$

得到新的训练样本：

$$
(s,a^*)
$$

最值得记住的一句话是：

> **DAgger 新数据中的 state 由 learner 产生，而监督动作由 expert 提供。**

---

# 15. Dataset Aggregation：为什么不是只训练最新数据

得到新的数据：

$$
D_1^{new}
$$

DAgger 并不是直接丢弃原始 expert dataset，而是做：

$$
D_1
=
D_0
\cup
D_1^{new}
$$

然后重新训练：

$$
\pi_2
=
\operatorname{Train}(D_1)
$$

再让 π₂ rollout：

$$
s
\sim
d_{\pi_2}
$$

expert 再标注：

$$
a^*
=
\pi_E(s)
$$

得到：

$$
D_2^{new}
$$

继续聚合：

$$
D_2
=
D_1
\cup
D_2^{new}
$$

不断重复：

```text
Train
↓
Learner Rollout
↓
Collect Learner States
↓
Expert Relabel
↓
Aggregate
↓
Retrain
```

这就是 DAgger 名称中的：

```text
Dataset Aggregation
```

---

# 16. 为什么要保留旧数据

阶段思考中一个很关键的问题是：

> 为什么 D_i = D_{i-1} ∪ D_i^new，而不是直接使用最新采集的数据？

原因是：

新数据主要解决：

```text
off-nominal states
recovery states
learner-induced states
```

但原始 expert demonstration 仍然包含大量：

```text
nominal behavior
correct task execution
normal task progression
```

如果完全丢掉旧数据，策略可能只剩下“如何处理错误状态”的经验，却损害正常任务行为。

因此 aggregation 的目标是：

> 同时保留 nominal expert behavior 与 learner-induced recovery states。

最终 dataset 会越来越接近：

```text
机器人真正部署时可能遇到的状态集合
```

---

# 17. 为什么“再收 1000 条完美专家演示”不等于 DAgger

这是理解 DAgger 最关键的概念检验之一。

假设已经拥有大量 expert demonstration。

再增加 1000 条 perfect demonstrations：

```text
Expert trajectory
Expert trajectory
Expert trajectory
...
```

虽然：

$$
N
\rightarrow
N+1000
$$

但数据仍然主要采样自：

$$
d_{\pi_E}
$$

也就是说：

> 数据量增加了，但 state distribution 可能几乎没有发生本质改变。

而 DAgger 的 learner rollout 会主动访问：

$$
d_{\pi_\theta}
$$

因此得到：

```text
模型自己真正会跑到的地方
```

所以 DAgger 的关键问题不是：

$$
\boxed{\text{Data Quantity}}
$$

而是：

$$
\boxed{\text{Data Distribution}}
$$

这是本章最值得长期保留的认识之一。

---

# 18. DAgger 如何带来 Recovery Ability

普通 BC 主要学习：

```text
正常状态下怎么做
```

这些状态可以称作 nominal states。

DAgger 会额外收集：

```text
偏左之后怎么修正
偏右之后怎么修正
抓取姿态略错之后怎么恢复
动作执行不足之后怎么补偿
```

这些可以理解为：

```text
off-nominal states
```

或者：

```text
recovery states
```

于是 policy 学到的不再只是：

> 如何沿着 expert trajectory 前进。

还包括：

> 离开 expert trajectory 之后，如何重新回到有效区域。

因此 DAgger 可以显著提升 recovery ability。

---

# 19. 一个完整的 Pick-and-Place DAgger 实例

假设任务是：

> 将桌面上的杯子抓起并放入篮子。

## 19.1 Expert Demonstration

expert trajectory：

```text
识别杯子
↓
移动到杯子正上方
↓
下降
↓
夹爪闭合
↓
抬起
↓
移动到篮子
↓
放下
```

原始 dataset 主要包含：

```text
正常接近
正常对准
正常抓取
正常搬运
正常放置
```

## 19.2 第一次 BC Rollout

训练得到 π₁。

π₁ 运行：

```text
成功接近
↓
末端偏右 1 cm
↓
继续下降
↓
夹爪已经偏离杯子中心
↓
抓空
```

注意：

> “夹爪偏右 1 cm”可能正是 expert demonstration 中很少出现的状态。

## 19.3 Expert Relabel

让 expert 对 learner 实际访问的状态进行标注：

```text
偏右 1 cm
→ 左移修正

偏右 3 cm
→ 明显左移并减速

即将抓空
→ 暂停下降，重新对准
```

于是形成新的：

$$
(s,a^*)
$$

样本。

## 19.4 Dataset Aggregation

将其加入：

$$
D_1
=
D_0
\cup
D_1^{new}
$$

重新训练得到 π₂。

π₂ 不仅知道：

```text
正常情况下如何抓
```

还逐渐知道：

```text
抓偏以后如何回来
```

## 19.5 第二次 Rollout

π₂ 可能不会再出现原来的偏右错误，却可能遇到：

```text
杯子被夹爪轻微碰动
```

于是再次：

```text
Learner rollout
↓
Collect state
↓
Expert relabel
↓
Aggregate
↓
Retrain
```

数据分布不断围绕 learner 真正的 failure modes 扩展。

---

# 20. Mixture Policy：为什么不能一开始就完全放手让 Learner 控制

经典 DAgger 还有一个现实问题：

> 初始 learned policy 可能非常差。

如果直接让一个刚训练出的 learner 完全控制机器人，可能出现：

```text
刚开始 rollout
↓
立刻严重偏离
↓
进入无意义甚至危险状态
```

因此可以使用 mixture policy：

$$
\pi_i
=
\beta_i\pi_E
+
(1-\beta_i)\pi_\theta
$$

这里：

- π_E 是 expert policy；
- π_θ 是 learner；
- β_i 控制 expert 介入比例。

早期可以令：

```text
β_i 较大
```

即更多依赖 expert。

随着 learner 变强：

```text
β_i 逐渐减小
```

最终：

```text
β_i → 0
```

让 learned policy 独立运行。

其直觉并不是复杂数学，而是：

> **先在 expert 的保护下让 learner 逐渐接管，避免初期 policy 直接进入完全失控的状态。**

---

# 21. 阶段思考 C：DAgger 真正改变了什么？

### 问题 1

DAgger 中新增样本：

$$
(s,a^*)
$$

里的 s 和 a* 分别来自谁？

### 解析

通常：

```text
s：来自 learner rollout
a*：来自 expert annotation
```

这是 DAgger 与普通 expert demonstration 最核心的区别。

### 问题 2

为什么 DAgger 不是“收更多 expert demonstration”？

### 解析

更多 expert demonstration 可能只是：

$$
N
\uparrow
$$

但：

$$
d_{\text{data}}
\approx
d_{\pi_E}
$$

没有明显变化。

DAgger 的目标则是：

$$
d_{\text{data}}
\rightarrow
d_{\pi_\theta}
$$

也就是让训练 distribution 更接近实际 deployment distribution。

### 问题 3

“DAgger 的核心是增加训练数据量”是否准确？

### 解析

不准确。

更准确的说法是：

> **DAgger 的核心是改变训练数据的 state distribution。**

数据量增加只是伴随现象。

---

# 22. 从控制系统角度重新理解 BC 与 DAgger

对控制背景的读者，可以把这个问题理解为：

一个 learned policy 本身位于闭环系统中。

```text
Reference / Task
↓
Learned Policy
↓
Action Target
↓
Low-level Controller
↓
Robot Dynamics
↓
State
↓
Observation
└────────────→ Learned Policy
```

BC 主要训练：

```text
nominal trajectory 上的映射
```

因此很像只在正常 operating region 中学习了一个 controller。

但真实系统存在：

```text
initial state variation
actuation error
perception error
delay
disturbance
object displacement
```

这些因素都会让真实状态离开 demonstration manifold。

从这个角度看，DAgger 做的事情可以理解为：

> 主动采集 closed-loop system 会进入的偏离状态，并学习对应的 corrective action。

因此它与控制中的：

```text
robustness
region of attraction
disturbance recovery
```

存在非常自然的思想联系。

但 DAgger 本质仍然是 imitation learning，而不是传统 model-based robust control。

---

# 23. 与无人机 BC → DAgger → PPO 路线的联系

对于已经做过无人机 imitation learning 的学习者，可以把这一过程重新理解为三层递进。

## 23.1 BC

目标：

```text
先学会 expert 的基本行为
```

例如：

```text
正常情况下应该怎样沿轨迹飞行
```

## 23.2 DAgger

目标：

```text
让 learner 自己暴露容易访问的偏离状态
```

然后学习：

```text
偏离以后应该怎么修正
```

例如：

```text
无人机偏离中心线后如何恢复
```

其主要处理：

```text
learner-induced distribution shift
```

## 23.3 PPO / RL

目标发生进一步变化。

不再要求：

$$
a
\approx
a^*
$$

而开始直接考虑：

```text
任务最终做得好不好
```

例如：

```text
是否碰撞
是否成功到达
轨迹误差大小
速度
控制平滑性
```

因此：

```text
BC
→ 学习基本 expert behavior

DAgger
→ 解决 learner state distribution

RL
→ 直接优化 task performance
```

这三者并不是简单堆叠，而具有清晰的逻辑递进。

---

# 24. DAgger 的局限：为什么它不是机器人学习的终点

DAgger 解决了 BC 的关键 distribution shift 问题，但仍然保留了 imitation learning 的本质：

$$
\pi_\theta(s)
\approx
\pi_E(s)
$$

因此它存在一系列根本局限。

## 24.1 Expert Query Cost

DAgger 需要 learner rollout 后继续请求 expert：

$$
a^*
=
\pi_E(s)
$$

真实机器人中 expert 可能是：

- 人类 teleoperator；
- 高质量 planner；
- MPC；
- 更强的 teacher policy。

如果 learner rollout 产生大量新状态，那么 expert labeling 成本可能非常高。

这使得经典 DAgger 很难无限迭代。

## 24.2 Safety Problem

为了知道 learner 会在哪些状态失败，DAgger 通常需要让 learner 自己 rollout。

但对于真实机器人：

```text
机械臂
无人机
自动驾驶
人机协作系统
```

某些失败状态可能不是“一个无害的数据点”，而是：

```text
碰撞
硬件损坏
物体破坏
安全风险
```

形成矛盾：

```text
想学会 failure recovery
↓
需要看到 failure-like states
```

但：

```text
真实系统又不希望真正进入危险状态
```

## 24.3 Expert Upper Bound

DAgger 的监督信号仍然来自：

$$
a^*
$$

也就是说，其目标仍然是：

> 模仿 expert。

但 expert 并不一定是全局最优。

例如 expert 选择：

```text
从杯子左边绕过去
```

而 learner 可能存在：

```text
从右边走
```

这种同样成功甚至更高效的行为。

如果 imitation loss 只衡量：

$$
\pi_\theta(s)
\approx
a^*
$$

那么一个“与 expert 不同但任务结果很好”的动作仍可能被惩罚。

这就引出了 RL。

---

# 25. Human Intervention：只在真正需要帮助的地方收集数据

如果 learner 已经能完成大部分正常 trajectory，那么继续收大量 perfect demonstration 的边际价值会越来越低。

真正有价值的状态可能集中在：

```text
即将失败
出现明显偏差
需要 recovery
处于成功与失败边界附近
```

Human Intervention 的核心思路是：

```text
Learner controlling
↓
Human detects imminent failure
↓
Human takes over
↓
Recover to safe / useful state
```

同时记录：

$$
(s,a_{\text{human}})
$$

这些数据具有很高的信息密度，因为它们通常恰好位于 learner 的 failure boundary。

与经典 DAgger 相比：

```text
DAgger
Learner rollout
→ Expert relabel

Human Intervention
Learner rollout
→ Human takes over when needed
```

二者共同的核心思想仍然是：

> **由 learner 的实际 state distribution 决定应该在哪里收数据。**

---

# 26. 从 Imitation Learning 自然走向 RL

BC 和 DAgger都在问：

> Expert 在这个 state 会做什么？

数学目标大致为：

$$
\min_\theta
\ell
\left(
\pi_\theta(s),
a^*
\right)
$$

而 RL 的问题变成：

> 这个 action 最终能够带来多大的任务收益？

其目标可以写为：

$$
J(\pi)
=
\mathbb{E}_\pi
\left[
\sum_{t=0}^{T}
\gamma^t r_t
\right]
$$

其中：

- r_t 表示 reward；
- γ 表示 discount factor；
- T 表示 horizon。

于是监督信号发生了根本变化。

DAgger：

```text
State
↓
Ask Expert
↓
Correct Action
```

RL：

```text
State
↓
Take Action
↓
Environment
↓
Reward
```

---

# 27. 为什么 BC 与 RL 的目标不同

假设 expert action 是：

```text
Δx = 2.0 cm
Δy = 0
```

learner 输出：

```text
Δx = 1.8 cm
Δy = 0.2 cm
```

从 imitation loss 看：

```text
learner action ≠ expert action
```

因此产生误差。

但如果机器人最终仍成功抓住杯子，那么从 task outcome 看：

```text
Success = 1
```

这个动作未必是坏动作。

因此：

```text
BC / DAgger
关注 action imitation
```

而：

```text
RL
关注 task return
```

这也是机器人基础模型常出现：

```text
Pretraining / SFT
↓
RL Post-training
```

这一结构的重要原因。

---

# 28. 一个统一视角：DAgger 与 RL 都在处理 Learner Distribution

一个非常重要的高级理解是：

> 进入 RL 以后，Distribution Shift 并没有消失。

RL policy 同样会：

$$
\pi_\theta
\rightarrow
d_{\pi_\theta}(s)
$$

不同的是反馈信号发生变化。

DAgger：

$$
s
\sim
d_{\pi_\theta}
$$

然后：

$$
a^*
=
\pi_E(s)
$$

RL：

$$
s
\sim
d_{\pi_\theta}
$$

然后获取：

$$
r
$$

或者进一步学习：

$$
Q(s,a)
$$

和：

$$
V(s)
$$

因此二者可以统一理解为：

```text
Policy Rollout
↓
Visit Learner Distribution
↓
Collect Training Signal
```

区别在于：

```text
DAgger：training signal = expert action
RL：training signal = reward / return / value
```

---

# 29. Distribution Shift 如何继续进入现代 VLA

VLA 虽然模型规模远大于普通 BC policy，但本质仍然是 policy。

可以写为：

$$
\pi_\theta
\left(
a
\mid
I,l,s
\right)
$$

其中：

- I 表示视觉 observation；
- l 表示 language instruction；
- s 表示 robot state；
- a 表示 robot action。

只要 VLA 与真实机器人形成 closed loop：

```text
Image + Language + Robot State
↓
VLA
↓
Action
↓
Robot
↓
New Image + New State
```

它就仍然存在：

```text
Policy Error
↓
State Shift
↓
New Observation
↓
Potential OOD
```

因此“大模型”并不会自动消灭 distribution shift。

例如任务：

```text
Pick up the red cup and place it in the basket.
```

VLA 可能：

```text
识别杯子
↓
接近
↓
抓取
```

但如果抓取时轻微碰到杯子：

```text
杯子位置变化
↓
visual observation 改变
↓
进入训练数据覆盖较弱的状态
```

后续仍然可能失败。

所以现代 VLA post-training 中一个长期问题依然是：

> 如何让 pretrained VLA 在真实 rollout distribution 上持续提高能力？

---

# 30. 从 Policy Recovery 到 Agent Recovery

并不是所有 failure 都必须要求低层 VLA 自己恢复。

可以把 recovery 分成三个层级。

## 30.1 Policy-level Recovery

通过 BC、DAgger、intervention data 等方法，让 policy 自己学习：

```text
偏离以后如何修回来
```

例如：

```text
夹爪偏右
→ 左移重新对准
```

## 30.2 RL-level Adaptation

通过 reward 学习：

```text
哪一种 recovery 真正提高 success rate
```

此时优化的不再只是 imitation accuracy，而是长期 task outcome。

## 30.3 Agent-level Recovery

如果当前技能已经明显失败，高层 Agent 可以：

```text
Detect Failure
↓
Stop Current Skill
↓
Re-observe
↓
Re-plan
↓
Retry / Re-stage
```

例如：

```text
杯子已经从夹爪掉落
```

此时继续执行原 action sequence 已经没有意义。

更合理的做法是：

```text
重新定位杯子
↓
重新规划抓取
↓
再次调用 VLA skill
```

这与后续 Harness VLA 中的 Retry、Re-staging、Semantic Re-grounding 与 Failure Recovery 形成自然联系。

---

# 31. 三层 Recovery 的统一框架

因此真实机器人系统的鲁棒性可以从三个层级理解：

```text
Policy-level
↓
局部动作偏差自己纠正

RL-level
↓
通过环境反馈优化策略

Agent-level
↓
任务级失败检测、重规划与重试
```

三者并不冲突。

一个未来更完整的 embodied system 很可能同时需要：

```text
Policy Robustness
+
RL Adaptation
+
Agent Recovery
```

这也是为什么“训练一个足够大的 VLA，然后把所有事情全部交给它”并不一定是现实系统的最佳设计。

---

# 32. 容易混淆的概念

## 32.1 Distribution Shift vs Compounding Error

### Distribution Shift

关注：

```text
训练输入分布
≠
部署输入分布
```

核心表达：

$$
d_{\pi_E}
\neq
d_{\pi_\theta}
$$

### Compounding Error

关注：

```text
一个早期错误
↓
改变未来状态
↓
未来继续犯错
↓
误差不断积累
```

二者相关，但不相同。

## 32.2 More Demonstrations vs DAgger

更多 expert demonstration：

```text
增加样本数量
```

但很可能仍然来自：

$$
d_{\pi_E}
$$

DAgger：

```text
增加 learner-visited states
```

核心是改变训练 distribution。

## 32.3 Error Rate vs Task Failure Rate

单步错误率 ε 很低，不代表整个长时任务 failure probability 很低。

因为：

```text
single-step accuracy
```

和：

```text
closed-loop trajectory success
```

是两个不同层面的指标。

## 32.4 DAgger vs RL

DAgger：

```text
向 expert 问正确动作
```

RL：

```text
向 environment 要 reward
```

DAgger 仍然是 imitation learning。

## 32.5 Recovery Data vs Ordinary Expert Data

ordinary expert data：

```text
大多在成功轨迹附近
```

recovery data：

```text
集中在偏离、边界和 failure-like states
```

后者往往对 closed-loop robustness 尤其重要。

---

# 33. 本章内容在 VLA 知识链中的位置

整套学习路线可以写为：

```text
Robot Policy
↓
Transformer
↓
VLM
↓
Robot Learning
↓
BC
↓
Distribution Shift / Compounding Error
↓
DAgger
↓
Robot Dataset
↓
Action Representation
↓
Action Chunking
↓
Diffusion / Flow Matching
↓
VLA
↓
Fine-tuning / Agent / RL
```

本章位于 Robot Learning 中非常基础、但又会贯穿后续 VLA 研究的问题层。

如果不能真正理解：

```text
learner-induced state distribution
```

后面很多内容都会显得像彼此无关的技术堆叠。

理解之后则会发现：

```text
DAgger
Human Intervention
Online RL
VLA Post-training
Failure Recovery
Agent Re-planning
```

在某种程度上都在回答一个共同问题：

> **机器人进入训练数据没有充分覆盖的状态以后，系统应该怎么办？**

---

# 34. 研究视角：以后读论文时应该问什么

当一篇 Robot Learning 或 VLA 论文声称：

```text
提高 robustness
提高 closed-loop success
减少 failure
提高 real-world generalization
```

应主动检查以下问题。

### 训练数据来自哪里？

是：

```text
纯 expert demonstrations
```

还是包含：

```text
policy rollouts
intervention data
failure trajectories
recovery trajectories
```

### Policy 是否真正 closed-loop evaluation？

只看 offline action prediction accuracy 不足以证明 closed-loop robot performance。

需要关注：

```text
rollout success rate
failure modes
recovery behavior
long-horizon performance
```

### 是否覆盖 learner-induced states？

如果所有数据都来自 perfect expert trajectories，那么需要警惕：

```text
训练 distribution 与部署 distribution 的 mismatch
```

### Failure 如何处理？

论文是依赖：

```text
policy 自己恢复
```

还是：

```text
human intervention
RL adaptation
high-level replanning
```

### 数据效率如何？

如果需要大量人工 relabel：

```text
real robot scalability
```

可能成为主要瓶颈。

### Safety 如何保证？

如果方法需要大量失败 rollout，应关注：

```text
危险状态是否真的被允许访问
是否有 intervention
是否先在 simulation 中训练
是否存在 safety controller
```

---

# 35. 综合思考

## 思考 1

为什么说：

> “BC 的核心问题不是模型不会拟合 expert，而是 supervised training distribution 与 closed-loop deployment distribution 不一致。”

### 关键思路

模型完全可能在：

$$
d_{\pi_E}
$$

上拟合得很好。

真正的问题发生在：

$$
d_{\pi_\theta}
$$

上。

所以 offline prediction accuracy 与 online rollout performance 必须区分。

## 思考 2

一个 BC policy 的单步错误率只有 0.5%，是否可以据此判断长时任务一定非常稳定？

### 关键思路

不能。

必须考虑：

```text
horizon T
错误是否改变未来状态
policy 是否具备 recovery
state distribution 是否逐步漂移
```

## 思考 3

如果一个机器人从未在 training dataset 中见过“杯子被碰倒后的画面”，DAgger 为什么可能帮助它？

### 关键思路

如果 learner rollout 本身容易产生这种状态，那么 DAgger 可以采集：

```text
learner-induced state
```

再由 expert 提供 recovery action。

于是数据集开始覆盖这一 failure state。

## 思考 4

为什么 Human Intervention 可以看作 DAgger 思想的一种更有针对性的延伸？

### 关键思路

两者都由 learner distribution 触发数据采集。

区别在于 Human Intervention 通常只在：

```text
high-value failure boundary
```

附近介入，因此减少大量不必要的 expert labeling。

## 思考 5

为什么 Agent Recovery 不能简单等同于 DAgger recovery？

### 关键思路

DAgger 主要增强：

```text
policy-level action recovery
```

Agent Recovery 则可以做：

```text
任务级 failure detection
重新观察
重新规划
重新调用技能
```

它们作用在不同层级。

---

# 36. 本章总结：应形成怎样的最终心智模型

本章不应该被记成“DAgger 是一种数据增强算法”，而应该形成一条完整的因果链。

第一，BC 的训练数据主要来自 expert state distribution：

$$
s
\sim
d_{\pi_E}
$$

但部署时机器人运行在 learned policy 自己产生的分布中：

$$
s
\sim
d_{\pi_\theta}
$$

因此：

$$
d_{\pi_E}
\neq
d_{\pi_\theta}
$$

形成 Distribution Shift。

第二，由于机器人是 closed-loop sequential system，当前动作会改变下一时刻状态。因此一个小的 early-stage error 可能改变未来所有输入条件，引发 Compounding Error。

第三，DAgger 的核心并不是单纯增加数据量，而是：

> **让 learner 自己暴露会访问的状态，再让 expert 对这些状态提供正确动作。**

其关键结构是：

```text
Learner State
+
Expert Action
```

并不断进行 Dataset Aggregation。

第四，DAgger 可以提高 recovery ability，但它仍然受 expert query cost、安全风险以及 expert upper bound 限制。

第五，这一问题会自然继续演化到：

```text
Human Intervention
↓
更高效地收集 failure / recovery data

Online RL
↓
直接利用 reward 优化任务结果

VLA Post-training
↓
让基础 VLA 在真实 rollout distribution 上继续改进

Agent Recovery
↓
在任务级执行 retry、re-stage 和 re-plan
```

因此，本章真正想建立的并不是一个孤立算法，而是下面这个长期有效的研究视角：

> **机器人策略最大的困难之一，是它的行为会主动改变未来输入分布。一个可靠的 embodied system 必须能够处理由自身行为产生的新状态。**

这是从经典 imitation learning 一直延伸到现代 VLA、RL post-training 和 agentic robotics 的共同核心问题。

---

# 37. 关键词

**Behavioral Cloning，BC — 行为克隆 — 使用 expert demonstration 通过监督学习拟合 robot policy。**

**Expert Policy — 专家策略 — 提供高质量 demonstration 或动作标签的策略。**

**State Distribution — 状态分布 — 某一 policy 运行时访问不同状态的概率分布。**

**Expert State Distribution — 专家状态分布 — expert policy 正常运行时访问的状态分布。**

**Learned Policy State Distribution — 学习策略状态分布 — learner 自己 rollout 时实际访问的状态分布。**

**Distribution Shift — 分布偏移 — 训练时与实际部署时输入状态分布发生不一致。**

**Compounding Error — 误差累积 — 早期动作错误通过环境动力学影响未来状态，并进一步产生新的策略错误。**

**Trajectory Horizon — 轨迹长度 — 一个 episode 中需要连续决策的 timestep 数量。**

**DAgger — Dataset Aggregation — 让 learner rollout，收集其访问状态，由 expert relabel，再不断聚合训练数据的方法。**

**Expert Relabel — 专家重新标注 — expert 对 learner 实际访问状态给出正确动作。**

**Dataset Aggregation — 数据集聚合 — 将新采集的 learner-state/expert-action 数据持续加入已有训练集。**

**Mixture Policy — 混合策略 — 在 DAgger 早期混合 expert 与 learner 控制，逐渐过渡到 learner 独立执行。**

**Nominal State — 正常状态 — expert 成功轨迹附近常见的状态。**

**Off-Nominal State — 偏离状态 — 由于误差、扰动或执行偏差产生的非标准状态。**

**Recovery Ability — 恢复能力 — policy 从偏离正常轨迹的状态重新回到有效执行区域的能力。**

**Human Intervention — 人类介入 — 在 learner 即将失败时由人类接管并产生高价值 recovery data。**

**Online RL — 在线强化学习 — 通过与环境真实交互获取 reward 并继续更新 policy。**

**VLA Post-training — VLA 后训练 — 在 pretrained VLA 基础上利用机器人数据、反馈或 RL 进一步适配和提升策略。**

**Agent Recovery — Agent 级恢复 — 在任务级发现失败后执行重新观察、重新规划、重试或重新布置。**

**Closed-loop Policy — 闭环策略 — 策略动作改变环境，新环境状态继续反馈给策略形成连续决策闭环。**

---

## 本章最终知识链

```text
Expert Demonstration
↓
Behavioral Cloning
↓
Train on d_πE
↓
Closed-loop Deployment
↓
Run on d_πθ
↓
Distribution Shift
↓
Early Action Error
↓
State Deviation
↓
Compounding Error
↓
Learner Rollout
↓
Expert Relabel
↓
Dataset Aggregation
↓
DAgger
↓
Recovery Ability
↓
Expert Cost / Safety / Expert Upper Bound
↓
Human Intervention
     +
RL Post-training
     +
Agent Recovery
↓
More Robust Embodied Policy
```

如果能够从这条链自然解释每一个箭头，就说明 Chat 05 的核心知识已经真正建立起来。
