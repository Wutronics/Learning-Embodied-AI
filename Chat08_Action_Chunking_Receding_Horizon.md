# 08｜Action Chunking 与 Receding Horizon：从单步动作预测到实时滚动执行

## 1. 本章导言

前面的学习已经回答了一个重要问题：**VLA 究竟输出什么？**

一个机器人策略可以输出关节位置、关节速度、末端执行器位姿、Cartesian Delta、夹爪状态等动作表示。但仅仅知道 action representation 还不够。真实机器人系统还有第二个同样重要的问题：

> **这些动作应该以什么时间结构被预测和执行？**

最简单的机器人策略可以表示为：

$$
o_t \rightarrow a_t
$$

即当前时刻获得 observation，然后预测当前时刻需要执行的一个 action。

这种形式在数学上非常自然，却没有充分考虑真实 VLA 系统的一个基本事实：

> **VLA 的推理频率通常远低于机器人底层控制频率。**

大型视觉语言动作模型可能几十毫秒、几百毫秒才能完成一次推理，而底层机械臂控制器可能以几十 Hz、几百 Hz 甚至更高频率运行。如果要求 VLA 在每一个底层控制周期重新完成一次视觉编码、语言理解和动作预测，系统往往根本无法实时运行。

因此，现代机器人策略越来越倾向于：

> **一次预测未来的一段动作，而不是只预测一个动作。**

这就是 Action Chunking。

Action Chunking 又会自然带来一系列新的问题：

```text
为什么一次预测一串动作？
↓
这一串动作应该多长？
↓
是不是全部执行完以后才重新推理？
↓
如何根据新的 observation 不断更新动作？
↓
多个重叠的 action chunk 如何融合？
↓
旧 chunk 为什么可能变得过时？
↓
VLA 推理延迟为什么会产生 stale observation 和 stale action？
```

因此，本章真正研究的并不只是“如何一次预测多个 action”，而是：

> **慢速学习策略如何与高速、连续、闭环的机器人控制系统连接起来。**

在整套 VLA 学习路线中，本章位于 Action Representation 之后，并直接为 ACT、Diffusion Policy、π0 以及后续实时 VLA 推理打基础。

---

## 2. 问题从哪里来：为什么不能每个控制周期只预测一个 Action？

### 2.1 最简单的单步策略

考虑机械臂执行一个 Pick-and-Place 任务：

> 将桌上的盒子抓起并放到右侧篮子中。

假设 VLA 的 observation 包含：

```text
RGB 图像
+
机器人 proprioception
+
语言指令
```

最简单的 policy 是：

$$
a_t=\pi_\theta(o_t)
$$

其中：

- oₜ：时刻 t 的 observation；
- πθ：参数为 θ 的机器人策略；
- aₜ：当前时刻预测的 action。

如果使用 Cartesian Delta，动作可能表示为：

```text
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
```

于是系统似乎可以不断：

```text
获得 observation
↓
VLA 推理
↓
执行一个 action
↓
获得新 observation
↓
再次推理
```

这种设计的优点非常明显：

> 每一个动作都依据最新 observation 产生，因此闭环程度非常高。

但真实系统中马上会遇到频率问题。

### 2.2 Policy Frequency 与 Control Frequency 不一致

假设：

```text
机器人底层控制频率：100 Hz
VLA 推理频率：5 Hz
```

那么底层控制器每隔：

$$
\frac{1}{100}\text{ s}=10\text{ ms}
$$

就需要新的控制目标。

而 VLA 每隔：

$$
\frac{1}{5}\text{ s}=200\text{ ms}
$$

才能完成一次推理。

也就是说：

```text
底层控制器：
10 ms → 需要一次控制更新

VLA：
200 ms → 才能产生一次新预测
```

两个频率相差 20 倍。

如果坚持要求：

$$
o_t\rightarrow a_t
$$

并且每执行一个底层 action 就重新运行一次 VLA，那么机器人将不得不频繁等待模型推理。

这在大型 VLA 中通常是不现实的。

### 2.3 与传统分层控制的联系

如果已经学习过无人机、机械臂或 MPC 系统，这个问题其实并不陌生。

传统机器人系统本来就是多时间尺度的：

```text
任务规划器
频率很低
↓
轨迹规划器
↓
运动控制器
频率较高
↓
关节控制器
↓
驱动器
频率很高
```

VLA 通常也不是直接替代所有底层控制器。

更常见的是：

```text
Camera / Robot State
        ↓
       VLA
        ↓
EEF Target / Joint Target / Action Sequence
        ↓
IK / Operational Space Controller
        ↓
Joint Controller
        ↓
Robot
```

因此，VLA 本质上需要解决的一个接口问题是：

> **我算得慢，但下面的机器人必须连续运动。**

Action Chunking 正是在这个背景下变得重要。

---

## 3. 阶段思考 A：为什么不能只预测一个 Action？

### 问题

如果底层机器人控制频率是 100 Hz，而 VLA 只能以 5 Hz 推理，为什么不能简单地让 VLA 每个控制周期输出一个 action？

### 解析

只回答“VLA 推理频率和底层控制频率不同”已经抓住了核心，但还可以进一步建立完整因果链：

```text
底层控制需要高频更新
↓
大型 VLA 推理较慢
↓
一个 VLA inference 无法对应一个底层控制周期
↓
机器人会出现等待、停顿或控制目标缺失
↓
需要一次推理产生多个未来 action
```

因此 Action Chunking 不是单纯为了“让动作更长”，而是一个**跨时间尺度接口机制**。

---

## 4. Action Chunk：一次预测一段动作

### 4.1 基本定义

单步 action：

$$
o_t \rightarrow a_t
$$

Action Chunk：

$$
o_t
\rightarrow
[a_t,a_{t+1},\dots,a_{t+H-1}]
$$

这里的 H 表示一次预测的 action 数量。

这一串未来动作：

```text
a_t
a_t+1
a_t+2
...
a_t+H-1
```

称为：

> **Action Chunk**

而 H 常被称为：

> **Chunk Horizon**

或者根据具体论文语境称为：

> **Prediction Horizon**

### 4.2 H 的实际含义

假设：

```text
H = 20
```

并且 action frequency 为：

```text
20 Hz
```

那么相邻 action 的时间间隔是：

$$
\Delta t=\frac{1}{20}\text{ s}=50\text{ ms}
$$

一个 chunk 覆盖的时间范围为：

$$
T_{\text{chunk}}
=
20\times50\text{ ms}
=
1000\text{ ms}
=
1\text{ s}
$$

也就是说，一次 VLA inference 可以生成未来大约 1 秒的运动计划。

---

## 5. 为什么一次预测一串 Action？

Action Chunking 至少有两个重要动机：

```text
Inference Efficiency
+
Temporal Correlation
```

前者解决计算频率问题，后者利用机器人动作本身的时间结构。

---

## 6. Inference Efficiency：减少昂贵的模型调用

假设机器人接下来需要执行 20 个 action。

如果每次只预测一个 action：

```text
Inference
↓
Action 1

Inference
↓
Action 2

Inference
↓
Action 3
...
```

需要 20 次 VLA inference。

如果使用：

```text
H = 20
```

那么可以：

```text
一次 inference
↓
预测 20 个 action
↓
连续执行
```

从推理调用次数来看：

```text
Single Action：
20 actions → 20 inference

Action Chunk：
20 actions → 1 inference
```

对于大型视觉语言模型，这个差别非常重要。

因此通常可以得到一个基本趋势：

$$
H\uparrow
\quad\Rightarrow\quad
\text{单位 action 所需 inference 次数下降}
$$

也就是说，更长 chunk 通常意味着更高的 inference efficiency。

但这并不是 Action Chunk 的全部价值。

---

## 7. Temporal Correlation：机器人动作具有时间结构

### 7.1 相邻 Action 并不是独立样本

机器人连续运动中，相邻 action 通常具有很强的时间相关性。

以抓取盒子为例：

```text
aₜ：
向目标靠近

aₜ₊₁：
继续靠近

aₜ₊₂：
略微下移

aₜ₊₃：
继续下移

aₜ₊₄：
闭合夹爪

aₜ₊₅：
向上抬起
```

这些 action 并不是六个互相独立的决策。

它们共同组成一个连续的 motor behavior：

```text
Approach
↓
Descend
↓
Grasp
↓
Lift
```

所以：

> **一个 action 的合理取值通常与其前后动作有关。**

这就是 temporal correlation。

### 7.2 为什么预测整段动作有价值？

如果每次独立预测一个 action：

```text
当前图像
↓
预测一个位移

下一帧图像
↓
重新预测一个位移

下一帧
↓
重新预测
```

即使每一次预测单独来看都合理，相邻预测之间也可能出现抖动。

例如：

```text
+2.0 mm
+3.2 mm
+1.7 mm
+3.4 mm
+1.9 mm
```

而 Action Chunking 让模型学习：

> **这一小段连续运动整体应该是什么形状。**

因此模型不只是学习：

```text
当前该动多少？
```

还隐式学习：

```text
这一段 reaching / grasping / lifting 动作应该如何连续展开？
```

### 7.3 一个直观类比

如果画一条圆弧，每移动 1 mm 都完全重新决定一次方向，那么轨迹可能出现：

```text
↗
 →
  ↘
   →
    ↗
```

而如果一次规划接下来的一小段圆弧，就更容易得到连续轨迹。

Action Chunk 的思想类似：

> **预测局部动作片段，而不是把动作序列完全看成互不相关的瞬时决策。**

---

## 8. 阶段思考 B：什么叫 Temporal Correlation？

### 问题

为什么机器人动作具有 temporal correlation？

### 容易出现的不完整理解

一种常见回答是：

> “因为这是机器人系统的特点。”

方向没有错误，但信息不足。

### 完整理解

机器人执行一个物理任务时，连续 action 共同组成一个具有动态连续性的运动过程：

```text
前一个动作改变机器人状态
↓
新的状态约束下一个动作
↓
多个连续动作共同完成一个运动 primitive
```

例如：

```text
Approach
→ Align
→ Descend
→ Close Gripper
→ Lift
```

所以机器人动作并不是 IID 样本。

Action Chunking 的一个重要意义，就是显式建模这种局部时间结构。

---

## 9. Chunk Horizon 不等于 Execution Horizon

这是本章最容易混淆的概念之一。

假设：

```text
H = 20
```

模型一次预测：

$$
A_t=
[a_t,a_{t+1},\dots,a_{t+19}]
$$

一个自然但错误的理解是：

> 预测 20 步，就必须把 20 步全部执行完以后才能重新推理。

实际上并不是。

真实系统需要区分两个 horizon。

---

## 10. Chunk Horizon / Prediction Horizon

H 描述：

> **模型一次预测多少个未来 action。**

例如：

```text
H = 20
```

意味着：

$$
\pi(o_t)
=
[a_t,\dots,a_{t+19}]
$$

模型获得的是未来 20 步动作。

---

## 11. Execution Horizon

再定义一个执行长度 K。

K 表示：

> **当前 action chunk 实际执行多少步以后，重新获取 observation 并进行新的 planning。**

例如：

```text
H = 20
K = 5
```

整个系统不是：

```text
预测 20
↓
执行 20
↓
再预测
```

而是：

```text
预测 20
↓
执行前 5
↓
重新观察
↓
再预测新的 20
↓
再执行前 5
```

因此通常：

$$
K\leq H
$$

---

## 12. 阶段思考 C：H = 20，K = 5 到底是什么意思？

### 问题

如果 chunk horizon 为 20，而 execution horizon 为 5，机器人怎样运行？

### 解析

正确的数据流应该是：

```text
Observation oₜ
↓
预测未来 20 个 action
↓
[aₜ ... aₜ₊₁₉]
↓
执行前 5 个
↓
到达新的机器人状态
↓
获得新 observation
↓
重新预测新的 20 个 action
```

需要特别纠正一个表述：

> K 不是“重推理的步数”。

更准确地说：

> **K 是两次 replanning 之间实际执行的 action 数量。**

因此：

```text
H：
看多远

K：
真正往前走多远再重新看
```

这个区别非常重要。

---

## 13. Replanning：为什么不能长期相信旧 Chunk？

考虑一个动态抓取场景。

在 t 时刻，VLA 看到杯子的位置，预测：

```text
向前
向前
向前
向下
向下
闭夹爪
抬起
...
```

如果环境完全静态，整段执行可能没有问题。

但机器人系统存在大量不确定性：

```text
杯子被碰动
抓取时发生滑动
机械臂实际运动与预测不同
目标物位置估计存在误差
夹爪第一次没有夹住
外部人员改变环境
```

如果仍然坚持执行最初预测的全部 action：

> 后半段动作可能已经不再适用于当前真实状态。

因此：

> **不能把 Action Chunk 当作一次性完全开环轨迹。**

需要在执行一部分以后重新观察。

这就是 replanning。

---

## 14. Receding Horizon：不断向前滚动的规划窗口

### 14.1 基本思想

Receding Horizon 可以表示为：

```text
预测未来 H 步
↓
只执行前 K 步
↓
获得新 observation
↓
再预测未来 H 步
↓
再执行前 K 步
↓
……
```

随着时间前进，预测窗口不断向前移动：

```text
第一次：

t -------------------- t+H


第二次：

     t+K -------------------- t+K+H


第三次：

          t+2K -------------------- t+2K+H
```

所以称为：

> **Receding Horizon，滚动时域。**

---

## 15. 与 MPC 的联系

对于有控制基础的读者，一个非常有用的心智模型是 MPC。

经典 MPC 会：

```text
当前状态
↓
预测未来一段时间
↓
求解未来控制序列
↓
只执行前面的控制
↓
获取新状态
↓
重新优化
```

例如得到：

$$
[u_t,u_{t+1},\dots,u_{t+H-1}]
$$

但不会简单地把整个控制序列完全开环执行。

Action Chunk + Replanning 具有非常相似的思想：

```text
Observation
↓
Learned Policy
↓
Action Chunk
↓
执行前一部分
↓
新 Observation
↓
重新预测
```

因此可以建立：

$$
\text{Action Chunking + Replanning}
\approx
\text{Learned Policy 版本的 Receding-Horizon 思想}
$$

但需要注意：

> 二者思想类似，不代表数学机制完全相同。

经典 MPC 通常在线求解优化问题。

VLA 则可能直接通过神经网络生成 action chunk。

所以真正共同的是：

> **预测未来，但只信任近期部分，并不断根据新反馈更新未来。**

---

## 16. 阶段思考 D：Action Chunk + Replanning 为什么像 MPC？

### 问题

Action Chunking 与 MPC / Receding Horizon Control 在思想上有什么相似之处？

### 解析

仅仅回答“都是分层控制”还不够。

更准确的共同点是：

```text
都预测未来一段控制行为
↓
都不会长期完全开环执行整个预测结果
↓
都只执行近期部分
↓
都根据最新状态重新预测
```

因此核心不是“分层”本身，而是：

> **滚动预测 + 局部执行 + 状态反馈。**

---

## 17. Horizon 的核心 Trade-off

现在可以看到 H 和 K 并不是越大越好或越小越好。

### 17.1 较大的 Chunk Horizon

H 较大意味着一次预测更远。

优势：

```text
一次 inference 产生更多 action
↓
模型调用次数减少

能够建模更长运动片段
↓
temporal coherence 增强
```

但问题是：

```text
后半段 action
↓
距离原始 observation 时间更远
↓
未来不确定性增加
↓
更容易失效
```

### 17.2 较小的 Execution Horizon

K 较小意味着更频繁 replanning。

优势：

```text
执行少量动作
↓
快速获得新 observation
↓
快速修正未来动作
↓
更 reactive
```

但代价是：

```text
replanning 更频繁
↓
VLA inference 更频繁
↓
计算需求增加
```

因此系统实际上存在两个基本权衡：

$$
\text{Inference Efficiency}
\leftrightarrow
\text{Reactivity}
$$

以及：

$$
\text{Temporal Consistency}
\leftrightarrow
\text{Freshness}
$$

这两个 trade-off 是后续实时 VLA 系统设计中的核心问题。

---

## 18. Temporal Aggregation：多个 Chunk 为什么会同时预测同一个 Action？

Action Chunking 配合频繁 replanning 后，又会出现一个新的现象：

> **不同时间生成的 action chunk 在时间轴上发生重叠。**

假设 H = 5。

在时刻 t，模型预测：

$$
\hat A_t
=
[
\hat a_t^{(t)},
\hat a_{t+1}^{(t)},
\hat a_{t+2}^{(t)},
\hat a_{t+3}^{(t)},
\hat a_{t+4}^{(t)}
]
$$

这里：

```text
上标 (t)
```

表示：

> 这个 action 是在时刻 t 预测出来的。

到了 t+1 时刻，重新获得 observation，又产生一个新 chunk：

$$
\hat A_{t+1}
=
[
\hat a_{t+1}^{(t+1)},
\hat a_{t+2}^{(t+1)},
\dots
]
$$

于是对于真实时间：

```text
t+1
```

存在两个候选预测：

$$
\hat a_{t+1}^{(t)}
$$

和：

$$
\hat a_{t+1}^{(t+1)}
$$

时间轴可以表示成：

```text
t 时刻预测：

aₜ   aₜ₊₁   aₜ₊₂   aₜ₊₃   aₜ₊₄

t+1 时刻预测：

     aₜ₊₁   aₜ₊₂   aₜ₊₃   aₜ₊₄   aₜ₊₅
```

所以：

> 同一个未来时刻可能被多个 chunk 重复预测。

---

## 19. Temporal Aggregation 的基本思想

Temporal Aggregation 要解决的问题是：

> **多个 chunk 对同一个时间点预测出不同 action 时，该用哪一个？**

一个最简单的思路是加权融合：

$$
a_{t+1}
=
w_1\hat a_{t+1}^{(t)}
+
w_2\hat a_{t+1}^{(t+1)}
$$

其中：

$$
w_1+w_2=1
$$

例如：

$$
a_{t+1}
=
0.3\hat a_{t+1}^{(t)}
+
0.7\hat a_{t+1}^{(t+1)}
$$

通常可以让更新预测拥有更高权重。

这样可以避免 action 在 chunk 边界处突然跳变。

---

## 20. Temporal Aggregation 为什么可以提高平滑性？

假设：

```text
旧 chunk：
目标位置 = 0.102

新 chunk：
目标位置 = 0.118
```

如果直接硬切：

```text
0.102
↓
0.118
```

下一次新的 chunk 又可能预测：

```text
0.105
```

于是控制目标可能：

```text
0.102
→ 0.118
→ 0.105
→ 0.123
```

出现明显抖动。

经过 aggregation：

```text
旧预测
+
新预测
↓
加权
↓
平滑目标
```

可能得到：

```text
0.110
→ 0.112
→ 0.114
```

因此：

> Temporal Aggregation 的一个主要作用，是改善频繁 replanning 造成的 chunk-to-chunk discontinuity。

---

## 21. 阶段思考 E：为什么同一个 aₜ 会出现多个预测？

### 问题

连续 replanning 后，为什么同一个实际时刻的 action 会存在多个预测值？

### 常见不完整回答

> “因为不同时刻输入的 observation 不一样。”

这说明已经意识到了 observation 在变化，但还缺少最关键的一层。

### 完整因果链

真正原因是：

```text
一次预测一整段未来 action
↓
下一时刻又预测一整段未来 action
↓
两个预测窗口在时间轴上重叠
↓
同一个真实时间点同时出现在多个 chunk 中
↓
因此得到多个预测值
```

输入不同解释了：

> 为什么这些预测值可能不同。

但：

> **Chunk overlap**

才解释了：

> 为什么同一个时刻会被预测多次。

---

## 22. 阶段思考 F：Temporal Aggregation 到底在解决什么？

### 问题

Temporal Aggregation 本质上是在做什么？

### 解析

只说“让 chunk 更加平滑”基本正确，但可以进一步区分：

Action Chunking 解决：

> 如何一次预测一段动作。

Receding Horizon 解决：

> 如何不断重新规划。

Temporal Aggregation 解决：

> 多个重叠 chunk 对同一时刻的 action 不一致时，如何融合。

所以三者不能混为一谈。

---

## 23. Temporal Aggregation 的隐藏代价：Smoothness 与 Reactivity

Temporal Aggregation 看起来能够提高平滑性，但它也带来新的 trade-off。

考虑当前时刻 t。

旧 chunk 可能是在：

```text
t-2
```

时刻根据：

```text
oₜ₋₂
```

生成的。

新 chunk 则根据：

```text
oₜ
```

生成。

如果环境变化不大，旧预测可以帮助减少抖动。

但如果目标物突然移动：

```text
旧预测：
仍认为杯子在旧位置

新预测：
已经看到杯子的新位置
```

如果 aggregation 给旧预测过高权重，就会出现：

```text
Smoothness ↑
但
Reactivity ↓
```

因此存在：

$$
\text{Smoothness}
\leftrightarrow
\text{Reactivity}
$$

新的 observation 更 reactive。

旧预测更平滑，但更可能 stale。

---

## 24. 从 Action Chunk 自然进入 Latency 问题

现在进入本章最后一层。

Action Chunking 已经解决了：

> VLA 不能和底层控制器同频运行的问题。

但这并没有消除 inference latency。

假设机器人在时刻 t 获得：

```text
oₜ
```

VLA 推理需要 d 个控制周期。

那么动作真正被用于机器人时已经到了：

```text
t+d
```

可以表示为：

$$
a_{t+d}
=
\pi(o_t)
$$

注意这里存在一个非常重要的时间错位：

```text
模型看到的是：
oₜ

动作实际作用时：
机器人已经到了 t+d
```

也就是说，当前真实状态已经是：

$$
s_{t+d}
$$

而不是：

$$
s_t
$$

这就是实时 VLA 中最基础的 delay mismatch。

---

## 25. Stale Observation：输入已经过时

假设：

```text
t = 0 ms
Camera 获得图像 o₀

t = 0 ~ 200 ms
VLA inference

t = 200 ms
得到 action
```

到了 200 ms 时：

> 模型得到的 action 仍然基于 0 ms 的图像。

但机器人、目标物和环境在这 200 ms 内可能已经变化。

因此：

```text
o₀
```

相对于实际执行时刻已经是：

> **Stale Observation，过时观测。**

可以把它理解成：

```text
Observation
↓
Inference Delay
↓
Action
```

由于 inference 不是瞬间完成，因此信息在计算过程中已经老化。

---

## 26. Stale Action：动作本身已经不适合当前状态

Stale Observation 和 Stale Action 不是完全相同的概念。

### Stale Observation

强调的是：

> **决策使用的输入信息过时。**

例如：

$$
a_{t+d}=\pi(o_t)
$$

执行时已经到了 t+d，但模型仍然依据 oₜ。

### Stale Action

强调的是：

> **当前正在执行的 action 是根据过去状态生成的，因此可能已经不适用于现在。**

例如，VLA 在 t 时刻产生：

$$
A_t
=
[a_t,a_{t+1},\dots,a_{t+19}]
$$

如果执行到：

```text
aₜ₊₁₅
```

这个 action 仍然源自：

```text
oₜ
```

而不是最新 observation。

所以通常：

$$
k\uparrow
\quad\Rightarrow\quad
\text{a}_{t+k}\text{ 的 staleness 风险增加}
$$

可以简单记忆：

```text
Stale Observation
=
信息旧

Stale Action
=
动作旧
```

---

## 27. 阶段思考 G：为什么“推理延迟”和“输出延迟”不是最准确的区分？

### 问题

Stale observation 和 stale action 的区别是什么？

### 容易出现的回答

一种直觉回答是：

```text
Stale observation：
推理的滞后

Stale action：
输出的滞后
```

这种回答捕捉到了“都有时间滞后”，但没有真正指出对象。

### 更准确的区分

应该问：

> **到底是什么东西旧了？**

如果旧的是模型使用的信息：

```text
old image
old robot state
old observation
```

就是 stale observation。

如果旧的是当前仍在执行的控制行为：

```text
old planned action
old action chunk
```

就是 stale action。

这种区分以后阅读 real-time VLA 论文时非常重要。

---

## 28. 为什么 Action Chunk 会放大 Staleness？

假设：

```text
Action Frequency = 20 Hz
H = 20
```

每个 action 持续：

```text
50 ms
```

那么整个 chunk 覆盖：

$$
20\times50\text{ ms}
=
1\text{ s}
$$

如果把整个 chunk 完全执行完：

```text
oₜ
↓
预测未来 1 秒
↓
整个 1 秒不重新观察
```

那么最后一个 action 实际上依据的是：

> 接近 1 秒以前的 observation。

如果是静态桌面，可能还能工作。

但如果发生：

```text
杯子移动
机械臂碰撞
抓取失败
目标滑动
环境变化
```

后面的 action 很可能全部失效。

所以：

$$
H\uparrow
\Rightarrow
\text{Inference Efficiency}\uparrow
$$

但与此同时：

$$
H\uparrow
\Rightarrow
\text{潜在 Staleness}\uparrow
$$

---

## 29. 完整机器人实例：Pick-and-Place

### 29.1 任务

机器人需要：

> 将盒子从篮子 A 抓起并放入篮子 B。

Observation：

```text
RGB image
+
joint / EEF state
+
语言指令
```

Action：

```text
Cartesian Delta
+
Gripper State
```

设：

```text
Action Frequency = 20 Hz
H = 16
K = 4
```

### 29.2 一次 Action Chunk 覆盖多久？

20 Hz 表示：

$$
\Delta t=50\text{ ms}
$$

H = 16，因此：

$$
T_{\text{chunk}}
=
16\times50\text{ ms}
=
800\text{ ms}
$$

即一次预测未来：

> **0.8 秒动作。**

### 29.3 每隔多久 Replanning？

K = 4，因此：

$$
T_{\text{execute}}
=
4\times50\text{ ms}
=
200\text{ ms}
$$

于是整个控制过程可能是：

```text
t = 0

Camera:
看到盒子

↓ VLA

预测未来 16 个 action

↓ Execute

执行前 4 个 action
持续 200 ms

↓ Observe

获得新的图像和 robot state

↓ Replan

重新预测未来 16 个 action
```

### 29.4 第一阶段：Approach

VLA 第一次预测：

```text
Chunk 1：

向前
向前
向前
向左
向左
向下
向下
...
```

机器人只执行前 4 步。

### 29.5 第二阶段：Replanning

200 ms 后获得新的 observation。

此时机器人已经更靠近盒子。

新的 VLA 预测可能变成：

```text
Chunk 2：

向左
向下
向下
向下
闭夹爪
抬起
...
```

于是 policy 不再盲目坚持 Chunk 1 的全部计划。

### 29.6 突发环境变化

假设此时盒子被轻轻碰了一下：

```text
原位置
   ↓
[BOX]

移动后
       [BOX]
```

如果机器人仍然执行最早生成的全部 chunk：

```text
继续向旧位置下降
↓
闭合夹爪
↓
抓空
```

这就是 stale action 导致的失败。

### 29.7 Receding Horizon 的作用

因为 K = 4，而不是 K = 16：

```text
只执行 200 ms
↓
重新观察盒子位置
↓
重新预测
```

所以机器人有机会根据盒子的移动修正轨迹。

这就是：

> **Reactivity。**

---

## 30. 为什么 Receding Horizon 仍不能彻底解决 Latency？

一个自然的问题是：

> 既然频繁 replanning 就能获得最新 observation，为什么还需要研究 real-time VLA？

原因是：

> **VLA inference 本身需要时间。**

例如，在执行完前 4 个 action 后获得：

```text
oₜ
```

然后启动 VLA inference。

如果 VLA 需要：

```text
300 ms
```

才能得到新 chunk，机器人不能每次都：

```text
停住
↓
等待 300 ms
↓
得到 chunk
↓
继续动
```

否则运动会非常不连续。

因此实际系统自然会想到：

> **Asynchronous Inference。**

---

## 31. Asynchronous Inference 的基本直觉

同步系统可能是：

```text
执行完旧 chunk
↓
停止
↓
拍图
↓
等待 VLA
↓
得到新 chunk
↓
继续执行
```

异步系统则是：

```text
旧 chunk 正在执行
        +
VLA 同时计算新 chunk
```

例如：

```text
t = 0
获得 o₀
↓
VLA 开始 inference

同时
↓
机器人继续执行旧 action queue

t = 300 ms
新 chunk 产生
↓
切换 / 融合
```

这样可以减少：

> **机器人因为等待模型而停顿。**

但 asynchronous inference 并没有自动消除 stale observation。

---

## 32. 阶段思考 H：为什么 Asynchronous Inference 仍然会有 Stale Observation？

### 问题

如果机器人一边执行旧 chunk，一边后台计算新 chunk，为什么新 chunk 仍然可能 stale？

### 解析

关键是：

```text
t = 0
拍到 o₀
↓
VLA 开始推理

与此同时
机器人继续运动

t = 300 ms
VLA 推理完成
```

虽然机器人没有停下来等模型，但是：

> 新 chunk 仍然是根据 o₀ 计算的。

到了 300 ms 时，机器人已经到了：

```text
s₃₀₀ms
```

所以：

```text
Inference Blocking
```

减少了，但：

```text
Observation Age
```

仍然存在。

因此：

> **Asynchronous inference 解决的是等待问题，而不是天然解决信息新鲜度问题。**

---

## 33. 从传统控制角度重新理解本章

如果从控制系统而不是机器学习术语出发，本章其实研究的是一个经典的：

> **多时间尺度闭环控制问题。**

可以建立如下层次：

```text
高层任务：
Pick and Place
↓
VLA Policy
低频、计算昂贵
↓
Action Chunk
中间时间尺度
↓
Robot Controller
高频
↓
Joint Command
↓
Actuator
```

本章中不同技术各自解决一个问题。

### Action Chunking

解决：

> VLA 太慢，不能每一个底层控制周期都重新推理。

### Temporal Correlation

说明：

> 连续动作本身具有结构，因此预测一段动作不仅节约推理，也可能提高动作一致性。

### Receding Horizon

解决：

> 长期完全执行旧 chunk 会失去反馈。

### Replanning

实现：

> 获得新 observation 后重新生成未来动作。

### Temporal Aggregation

解决：

> 不同 chunk 重叠时，对同一时刻 action 的预测可能不连续。

### Latency

暴露：

> 从获得 observation 到动作真正作用之间存在时间延迟。

### Stale Observation / Stale Action

描述：

> 延迟和长 chunk 导致模型使用的信息或执行动作逐渐过时。

---

## 34. 容易混淆的概念

### 34.1 Chunk Horizon 与 Execution Horizon

Chunk Horizon：

```text
一次预测多少步
```

Execution Horizon：

```text
实际执行多少步后重新规划
```

例如：

```text
H = 20
K = 5
```

不是“预测 20 步然后执行 20 步”。

而是：

```text
预测 20
↓
执行 5
↓
重新预测 20
```

### 34.2 Action Chunking 与 Receding Horizon

Action Chunking：

> 一次预测多个 action。

Receding Horizon：

> 预测未来一段，但只执行一部分，然后重新规划。

Action Chunking 本身并不保证 closed-loop。

如果预测一个超长 chunk 并一次性全部执行，仍然接近开环。

### 34.3 Replanning 与 Temporal Aggregation

Replanning：

> 根据新的 observation 生成新的 chunk。

Temporal Aggregation：

> 对多个重叠 chunk 的 action prediction 进行融合。

所以：

```text
Replanning
产生多个新 chunk

Temporal Aggregation
处理 chunk 之间的重叠与不连续
```

### 34.4 Stale Observation 与 Stale Action

Stale Observation：

> 模型使用的 observation 已经过时。

Stale Action：

> 当前执行的 action 来自过去状态，因此已经可能不适用于当前真实环境。

### 34.5 Temporal Correlation 与 Temporal Aggregation

这两个词非常容易因为都有“temporal”而混淆。

Temporal Correlation：

> 机器人相邻动作之间天然存在时间相关性。

Temporal Aggregation：

> 对多个时间上重叠的预测结果进行融合。

前者是：

```text
数据 / 动作本身的性质
```

后者是：

```text
一种执行或预测融合机制
```

---

## 35. 本章中的核心数学关系

虽然本章数学并不复杂，但几个量之间的关系必须清晰。

### 单步动作策略

$$
a_t=\pi_\theta(o_t)
$$

### Action Chunk Policy

$$
\pi_\theta(o_t)
=
[a_t,a_{t+1},\dots,a_{t+H-1}]
$$

其中：

```text
H
=
Chunk Horizon
```

### Chunk 覆盖的实际时间

如果 action frequency 为 f_action：

$$
\Delta t
=
\frac{1}{f_{\text{action}}}
$$

那么：

$$
T_{\text{chunk}}
=
H\Delta t
$$

### Execution Horizon 对应时间

若执行 K 步：

$$
T_{\text{execution}}
=
K\Delta t
$$

### 带推理延迟的策略

如果 inference delay 为 d：

$$
a_{t+d}
=
\pi(o_t)
$$

这个式子最重要的不是函数本身，而是下标：

```text
Observation：
t

Action 真正作用：
t+d
```

这个时间差就是 real-time VLA 问题的来源之一。

### Temporal Aggregation 的简化表达

假设同一 action 有两个预测：

$$
a_t
=
w_1\hat a_t^{(t-1)}
+
w_2\hat a_t^{(t)}
$$

并满足：

$$
w_1+w_2=1
$$

这只是一个用于建立直觉的简化形式。

真正论文中的具体 aggregation 方法可能不同，但本章需要掌握的重点是：

> **同一时刻可能由多个 chunk 提供预测，因此需要决定如何融合这些预测。**

---

## 36. 综合思考：为什么 H 越大不一定越好？

可以把 H 增大的影响写成两个方向。

第一条：

```text
H 增大
↓
一次 inference 产生更多 action
↓
模型调用次数减少
↓
Inference Efficiency 提高
```

第二条：

```text
H 增大
↓
Chunk 后半段距离生成 observation 更远
↓
环境与机器人状态可能变化更多
↓
Action Staleness 增加
↓
Reactivity 降低
```

所以：

> Action Chunking 不是简单地把 H 尽可能设置得大。

真正的问题是：

> **在模型推理速度、机器人动态、环境变化速度以及任务特征之间找到合理时间尺度。**

---

## 37. 综合思考：为什么 K 越小也不一定越好？

K 越小：

```text
更频繁获得新 observation
↓
更频繁 replanning
↓
更强 closed-loop ability
```

但同时：

```text
更多 VLA inference
↓
计算负担更高
↓
可能跟不上实时执行
```

所以 K 同样存在 trade-off。

这也说明 Action Chunking 的研究不能只看：

```text
模型预测精度
```

还必须看：

```text
Inference Frequency
Control Frequency
Chunk Length
Execution Length
Latency
Environment Dynamics
```

---

## 38. 本章的完整逻辑链

将所有内容串联起来，可以得到：

```text
VLA 推理较慢
↓
无法每个底层控制周期重新预测
↓
Action Chunking
↓
一次预测多个未来 action
↓
减少 inference 次数
+
利用 temporal correlation
↓
但长 chunk 后半段越来越依赖旧 observation
↓
Replanning
↓
只执行一部分动作
↓
获得新 observation
↓
重新预测
↓
Receding Horizon
↓
不同 chunk 时间范围重叠
↓
同一时刻出现多个 action prediction
↓
Temporal Aggregation
↓
提高 chunk 之间的连续性
↓
但旧 chunk 可能已经 stale
↓
Smoothness 与 Reactivity 发生冲突
↓
进一步考虑 inference latency
↓
Stale Observation
↓
Stale Action
↓
引出真正的 Real-Time VLA 问题
```

如果能够独立解释这条因果链，就说明已经真正掌握本章。

---

## 39. 本章在 VLA 知识体系中的位置

目前已经走过：

```text
Robot Policy
↓
Transformer
↓
Vision-Language Model
↓
Behavioral Cloning
↓
Distribution Shift / DAgger
↓
Robot Dataset
↓
Action Representation
↓
Action Chunking
```

本章解决的是：

> **动作在时间维度上的组织方式。**

接下来不同机器人策略会在此基础上选择不同的 action generation mechanism：

```text
Action Chunking
↓
ACT
↓
Diffusion Policy
↓
Flow Matching
↓
现代 VLA
```

以后阅读 ACT、Diffusion Policy 或 π0 时，不应该只问：

```text
模型输出什么 action？
```

还要问：

```text
一次输出多少步？
Prediction Horizon 是多少？
Execution Horizon 是多少？
多久 Replanning？
Chunk 如何衔接？
是否使用 Temporal Aggregation？
Inference Latency 多大？
```

这已经从“网络架构问题”进入了真正的：

> **Robot Policy System Design。**

---

## 40. 研究视角：阅读论文时应该关注哪些设计选择？

本章虽然属于基础知识，但已经开始出现明显的研究问题。

阅读机器人 policy 论文时，可以重点关注以下设计。

### 第一类：Action Chunk Length

论文一次预测多长的 chunk？

更长 chunk 是否真的带来更高成功率？

还是只提高了推理效率？

### 第二类：Execution Horizon

模型预测 H 步以后：

```text
执行 H 步？
执行 H/2？
只执行几步？
每一帧都重新预测？
```

这些选择直接决定系统的 closed-loop 程度。

### 第三类：Action Frequency

同样是 H = 20：

```text
10 Hz → 覆盖 2 秒
20 Hz → 覆盖 1 秒
50 Hz → 覆盖 0.4 秒
```

所以不能脱离 action frequency 单独讨论 H。

### 第四类：Temporal Aggregation

论文是否融合多个重叠 chunk？

如何确定不同预测的权重？

是更强调：

```text
Smoothness
```

还是：

```text
Reactiveness
```

### 第五类：Inference Latency

大型 VLA 如果每次推理需要几百毫秒，那么即使 policy 在 offline benchmark 上很好：

> 真机执行时也可能因为 stale observation 而严重退化。

因此模型质量不能只看离线 action prediction error。

### 第六类：Static Task 与 Dynamic Task

如果任务是：

```text
静态桌面上的固定物体抓取
```

长 chunk 可能问题不大。

但如果是：

```text
人机协作
移动目标
动态环境
高速 manipulation
```

staleness 会更加严重。

所以 optimal horizon 与任务动态特性密切相关。

---

## 41. 综合思考题

### 思考 1

机器人 action frequency 为 20 Hz，H = 16。

Action Chunk 覆盖多少时间？

#### 解析

每个 action：

$$
\Delta t
=
\frac{1}{20}
=
0.05\text{ s}
$$

因此：

$$
T
=
16\times0.05
=
0.8\text{ s}
$$

### 思考 2

如果 H = 16，K = 4，每隔多久进行一次 replanning？

#### 解析

K = 4：

$$
T_{\text{execution}}
=
4\times50\text{ ms}
=
200\text{ ms}
$$

因此每执行约 0.2 秒重新规划一次。

### 思考 3

为什么 H 越大通常 inference efficiency 越高？

#### 解析

因为一次 inference 能产生更多 action：

```text
H ↑
↓
每次 inference 覆盖更多控制周期
↓
单位 action 所需 inference 数下降
```

但这个结论不能孤立理解，因为同时还会增加后半段 action 的 staleness。

### 思考 4

为什么只说“长 Chunk 有状态滞后”还不够？

#### 解析

因为 H 增大同时带来两个方向的效果：

```text
正面：
Inference Efficiency ↑
Temporal Coherence ↑

负面：
Future Uncertainty ↑
Staleness Risk ↑
Reactivity ↓
```

研究中要分析的是 trade-off，而不是只指出一个缺点。

### 思考 5

为什么 asynchronous inference 不能天然消除 stale observation？

#### 解析

因为：

```text
机器人继续运动
```

和：

```text
模型使用最新 observation
```

是两个不同问题。

异步只解决：

> 推理期间不要让机器人完全停住。

但模型从拍摄 oₜ 到推理完成仍然需要时间，因此最终 chunk 仍然可能建立在旧 observation 上。

### 思考 6

用一句话分别解释以下三个概念。

#### Action Chunking

一次预测未来的一串 action。

#### Receding Horizon

预测未来一段，但只执行近期部分，再根据新 observation 重规划。

#### Temporal Aggregation

融合多个重叠 action chunk 对同一时刻产生的多个预测。

### 思考 7

从控制系统角度看，本章真正解决的是什么矛盾？

#### 解析

最核心的矛盾不是“如何预测多个 action”，而是：

$$
\boxed{
\text{低频、昂贵的高层 Policy Inference}
\leftrightarrow
\text{高频、连续、闭环的机器人控制}
}
$$

Action Chunking、Replanning、Receding Horizon 和 Temporal Aggregation 都是在处理这个多时间尺度矛盾。

---

## 42. 本章总结

本章最重要的不是记住几个术语，而是建立一套完整的时间维度心智模型。

第一，现代 VLA 通常不能与底层机器人控制器同频运行，因此需要一次预测多个连续 action。Action Chunking 首先是解决**推理频率与控制频率不匹配**的问题，同时也利用了机器人动作的 temporal correlation。

第二，模型一次预测 H 步，并不意味着必须执行 H 步。Chunk Horizon 决定“看多远”，Execution Horizon 决定“走多远以后重新看”。二者分离以后，系统就能够采用 Receding Horizon：

```text
预测未来
↓
执行近期部分
↓
获得反馈
↓
重新规划
```

第三，频繁 replanning 会让不同 action chunk 在时间轴上发生重叠。Temporal Aggregation 可以融合多个预测，提高动作连续性，但旧 chunk 权重过大会损害 reactivity，因此存在 smoothness 与 freshness 的权衡。

第四，Action Chunk 并没有消除实时问题。大型 VLA 的 inference latency 会导致 observation 在推理完成之前已经老化，从而产生 stale observation；而过去基于旧 observation 生成、现在仍在执行的动作则可能成为 stale action。

最终，本章可以压缩成一条最重要的因果链：

```text
VLA 很慢
↓
Action Chunking
↓
减少 inference 次数
↓
但动作逐渐过时
↓
Replanning / Receding Horizon
↓
恢复闭环反馈
↓
重叠 Chunk
↓
Temporal Aggregation
↓
进一步暴露 Latency / Staleness 问题
```

这个心智模型会直接成为后续理解 ACT、Diffusion Policy、π0 和 Real-Time VLA 的基础。

---

## 43. 关键词

**Action Chunk — 动作块 / 动作序列 — 一次 policy inference 产生的一组未来连续动作**

**Action Chunking — 动作分块预测 — 一次预测多个未来 action，而不是只预测当前一步**

**Chunk Horizon — 动作块预测长度 — 一个 action chunk 包含多少个未来 action**

**Prediction Horizon — 预测时域 — 模型向未来预测的动作范围**

**Execution Horizon — 执行时域 — 一个 chunk 实际执行多少步后重新规划**

**Temporal Correlation — 时间相关性 — 相邻机器人动作在运动过程上具有连续依赖关系**

**Replanning — 重规划 — 根据新的 observation 重新生成未来动作**

**Receding Horizon — 滚动时域 — 预测未来一段，只执行近期部分，再不断重新预测**

**Temporal Aggregation — 时间聚合 — 融合多个重叠 chunk 对同一时刻产生的动作预测**

**Inference Frequency — 推理频率 — policy 每秒能够完成多少次新预测**

**Control Frequency — 控制频率 — 底层控制器更新控制命令的频率**

**Inference Latency — 推理延迟 — 从获得 observation 到模型生成 action 所经历的时间**

**Stale Observation — 过时观测 — 模型决策依据的信息相对于实际执行时刻已经老化**

**Stale Action — 过时动作 — 过去基于旧状态生成、当前已经可能不适用的动作**

**Asynchronous Inference — 异步推理 — 机器人继续执行已有动作的同时计算新的 action chunk**

**Reactivity — 反应能力 — 策略根据最新环境变化及时调整动作的能力**

**Temporal Consistency — 时间一致性 — 连续动作在时间上保持平滑、连贯的性质**
