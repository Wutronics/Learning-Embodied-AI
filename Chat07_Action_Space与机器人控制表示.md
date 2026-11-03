# 07｜Action Space 与机器人控制表示

VLA 的前半部分解决“机器人看到了什么、理解了什么”，而真正让机器人产生物理运动，还必须回答一个更加工程化的问题：

> VLA 最终到底输出什么？

这个问题看似只是输出向量的定义，实际上直接决定了策略学习的对象、模型与机器人控制系统的接口、对不同机器人本体的依赖程度，以及后续跨机器人泛化的难度。

一个 VLA 可以输出关节角、关节速度、末端位姿、末端相对位移，也可以输出一段连续动作序列。即使两个模型网络结构完全相同，只要 action representation 不同，它们实际解决的问题就可能明显不同。

因此，理解 Action Space 不能停留在“输出有几维”，而需要把它放进完整控制链中理解：

```text
Image + Language + Robot State
                ↓
             VLA Policy
                ↓
       Action Representation
                ↓
       IK / Jacobian / Controller
                ↓
      Joint Command / Torque
                ↓
              Robot
```

本章的目标，是建立这样一个完整心智模型：

> VLA 通常负责决定机器人“接下来应该怎样运动”，传统机器人控制器负责把这个运动目标稳定、快速、安全地执行出来。

---

# 1. 为什么必须先理解 Action Space

设机器人策略为：

$$
a_t=\pi(o_t)
$$

其中：

- oₜ 表示当前 observation；
- π 表示机器人策略；
- aₜ 表示策略在时刻 t 输出的 action。

对于 VLA，observation 通常还会进一步包含：

$$
o_t=(I_t,l,s_t)
$$

其中：

- Iₜ：视觉输入；
- l：语言指令；
- sₜ：机器人自身状态，例如 joint position、joint velocity 等。

问题在于：

$$
a_t
$$

究竟是什么？

它可能是：

```text
7 个关节的目标角度
```

也可能是：

```text
末端执行器向右移动 2 cm
```

也可能是：

```text
末端执行器目标位姿 + 夹爪闭合
```

还可能更加底层：

```text
每个关节应该施加多少 torque
```

这些表示在数学维度上有时甚至相同，但物理含义完全不同。

例如，一个 7 DoF 机械臂的 joint position action 可以是 7 维：

$$
a_t=
[q_1,q_2,q_3,q_4,q_5,q_6,q_7]
$$

而一个 Cartesian delta action 也可能是 7 维：

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

前者的七维是七个关节。

后者的七维是：

```text
3D translation
+
3D rotation
+
1D gripper
```

因此：

> Action 的维度本身并不足以说明机器人在控制什么，必须结合 action representation 一起理解。

---

# 2. VLA 通常不会直接输出 PWM

从传统自动控制或嵌入式系统出发，很容易产生一个直觉：

> 既然策略控制机器人，那么策略是不是应该直接产生电机控制量？

在大多数现代 VLA 系统中，并非如此。

VLA 更常见的输出是：

```text
EEF Target
Joint Target
Cartesian Delta
Joint Velocity Target
```

而不是直接输出：

```text
PWM
Motor Current
Motor Voltage
```

其核心原因是控制频率层级不同。

大型 VLA 的推理通常相对较慢，而底层电机控制往往需要非常高的实时性。

因此实际系统更接近：

```text
VLA
↓
较低频更新运动目标
↓
底层控制器
↓
高频跟踪目标
↓
Motor
```

这与无人机系统非常类似。

例如：

```text
RL Policy
↓
Velocity / Position Reference
↓
PX4
↓
Attitude / Rate Controller
↓
Motor
```

机械臂中可以对应为：

```text
VLA
↓
EEF / Joint Target
↓
IK / Robot Controller
↓
Joint Command
↓
Motor
```

二者背后的系统思想一致：

> 学习策略产生 reference，传统控制器完成 tracking。

---

# 3. Joint Position Action

最直接的一种 action representation，是让策略输出每个关节的目标位置。

对于一台 7 DoF 机械臂：

$$
q=
[q_1,q_2,\ldots,q_7]
$$

如果策略输出：

$$
a_t=q_t^*
$$

则表示：

> 策略直接告诉机器人，每一个关节应该转到什么目标角度。

例如：

$$
q_t^*=
[0.2,-0.4,0.1,-1.5,0.3,1.2,0.5]
$$

这里每一个分量都对应一个具体 joint。

随后，底层位置控制器可以根据目标位置与当前状态之间的误差计算控制量。

例如，一个简化的 PD 控制形式可以写成：

$$
\tau
=
K_p(q^*-q)
+
K_d(\dot q^*-\dot q)
$$

其中：

- q：当前关节位置；
- q*：目标关节位置；
- q̇：当前关节速度；
- q̇*：目标关节速度；
- Kₚ：位置误差增益；
- K_d：速度误差增益；
- τ：关节控制力矩。

这里非常重要的一点是：

> VLA 不需要亲自完成所有底层控制计算。

VLA 只需要产生：

$$
q^*
$$

后面的 tracking 可以继续交给传统控制器。

---

## 阶段思考 A

### 问题 1

Joint Position Action 和 Joint Velocity Action 在物理意义上的区别是什么？

### 解析

Joint Position 解决的是：

> 每个关节最终应该到哪里。

其目标通常写成：

$$
q^*
$$

而 Joint Velocity 解决的是：

> 当前每个关节应该朝哪个方向、以多快的速度运动。

表示为：

$$
\dot q^*
$$

因此，不能把二者仅理解成“两个不同数组”。

它们对应的是不同的控制接口。

---

# 4. Joint Velocity Action

如果策略不输出目标角度，而输出关节速度，则：

$$
a_t=\dot q_t^*
$$

例如：

$$
\dot q^*
=
[0.1,-0.2,0,0.15,0,0.05,0]
$$

这表示：

- 第 1 个关节正向旋转；
- 第 2 个关节反向旋转；
- 第 3 个关节当前不动；
- 其他关节分别按照给定速度运行。

从控制语义上看：

```text
Joint Position：
去哪里

Joint Velocity：
现在怎样运动
```

Joint velocity 比 joint position 更具有局部运动含义，但仍然明显依赖机器人的具体关节结构。

---

# 5. Joint Torque Action

更加底层的一种方式，是直接让策略输出每个关节的力矩：

$$
a_t=\tau_t
$$

对于 7 DoF 机器人：

$$
\tau_t=
[\tau_1,\tau_2,\ldots,\tau_7]
$$

机械臂动力学通常可以抽象成：

$$
M(q)\ddot q
+
C(q,\dot q)\dot q
+
g(q)
=
\tau
$$

其中：

- M(q)：惯性矩阵；
- C(q,q̇)：科氏力与离心力相关项；
- g(q)：重力项；
- τ：关节力矩。

如果 policy 直接输出 τ，那么它已经进入了非常底层的控制层。

这类输出对模型和控制频率提出更高要求。

大型 VLA 往往不适合直接承担高频 torque control，原因包括：

```text
推理频率较低
实时性要求高
动力学敏感
安全约束更严格
控制稳定性要求更高
```

因此工程系统更常采用：

```text
Large VLA
↓
Position / Velocity / EEF Target
↓
Fast Low-level Controller
↓
Torque / Motor
```

---

## 阶段思考 B

### 问题 2

为什么大型 VLA 一般不会直接输出 PWM，甚至也较少直接承担高频 torque control？

### 解析

只回答“因为模型推理比较慢”方向是正确的，但还不完整。

完整因果关系是：

```text
大型 VLA 推理频率有限
↓
Motor / Torque Control 通常要求高频闭环
↓
VLA 无法在每一个底层控制周期重新计算控制量
↓
直接控制容易降低稳定性和安全性
↓
采用分层控制
↓
VLA 产生低频 reference
底层 controller 高频 tracking
```

因此，Action Representation 实际上还隐含了：

> 学习策略和传统控制系统到底在哪一层完成接口。

---

# 6. Joint Space 的核心问题：与机器人本体强绑定

Joint-space action 有一个非常明显的特征：

> Action 的每一个维度都与具体机器人的机械结构直接对应。

对于 Franka：

```text
7 DoF
```

可能有：

$$
q=
[q_1,\ldots,q_7]
$$

对于 UR5：

```text
6 DoF
```

则可能是：

$$
q=
[q_1,\ldots,q_6]
$$

二者的 joint index、关节轴方向、机械结构都不同。

因此，即使两个机器人完成完全相同的任务：

> 末端向右移动 5 cm。

它们内部对应的 joint action 也可能完全不同。

所以：

$$
\boxed{
Joint\ Action
\text{具有很强的 embodiment dependence}
}
$$

这将成为后续 Cross-Embodiment Learning 中的重要难点。

---

# 7. End-Effector Position

对于 manipulation 任务，人真正关心的通常不是：

> 第 3 个关节应该转多少度。

而是：

> 机械手应该到哪里。

于是可以定义末端执行器位置：

$$
p=
[x,y,z]^T
$$

如果 VLA 输出：

$$
p^*=
[0.55,0.12,0.30]
$$

则含义是：

> 让机械臂末端移动到这个空间位置。

这种 representation 比 joint space 更接近任务本身的物理语义。

但机器人最终仍然依赖关节运动，因此需要一个中间转换：

```text
EEF Position
↓
Inverse Kinematics
↓
Joint Configuration
```

即：

$$
q^*=IK(p^*)
$$

---

# 8. End-Effector Pose

只有 position 通常还不足以完成抓取。

例如，机器人夹爪即使到了杯子旁边，如果朝向错误，仍然无法正确抓取。

因此需要：

```text
Position
+
Orientation
```

完整的末端状态称为 End-Effector Pose。

可以用齐次变换表示：

$$
T=
\begin{bmatrix}
R & p\\
0 & 1
\end{bmatrix}
$$

其中：

- p：末端位置；
- R：末端姿态旋转矩阵。

机器人系统中也经常使用 quaternion 表示 orientation：

```text
[x, y, z, qx, qy, qz, qw]
```

因此，一个 7 维 EEF pose 与一台 7 DoF 机器人对应的 7 维 joint position 虽然维度一致，但意义完全不同。

---

## 阶段思考 C

### 问题 3

End-Effector Position 与 End-Effector Pose 有什么区别？

### 解析

Position 只回答：

> 末端在哪里？

即：

$$
p=[x,y,z]
$$

Pose 还必须回答：

> 末端朝什么方向？

因此：

```text
Pose = Position + Orientation
```

这是 manipulation 中非常重要的区别。

抓取一个箱子的侧面和从上方抓取，即使末端 position 相同，orientation 也可能完全不同。

---

# 9. Absolute Action

假设当前机械臂末端位置为：

$$
p_t=
[0.40,0.10,0.30]
$$

策略直接输出：

$$
p^*=
[0.55,0.20,0.25]
$$

这种 action 的含义是：

> 直接给出目标状态在某个坐标系中的绝对值。

这称为 Absolute Action。

数学形式可以写成：

$$
a_t=p^*
$$

它回答：

> 应该去哪里？

而不关心当前位置与目标之间相差多少。

---

# 10. Relative Action 与 Cartesian Delta

另一类非常常见的方式，是输出相对于当前状态的变化量。

例如：

$$
\Delta p=
[0.02,0,-0.01]
$$

表示：

```text
x 方向移动 +2 cm
y 方向不变
z 方向下降 1 cm
```

新的目标位置为：

$$
p_{t+1}^*
=
p_t+\Delta p_t
$$

假设：

$$
p_t=
[0.40,0.10,0.30]
$$

同时：

$$
\Delta p_t=
[0.02,0,-0.01]
$$

则：

$$
p_{t+1}^*
=
[0.42,0.10,0.29]
$$

如果位置与姿态变化同时输出，则可能得到：

$$
a_t=
[
\Delta x,
\Delta y,
\Delta z,
\Delta r_x,
\Delta r_y,
\Delta r_z
]
$$

再加入夹爪：

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

这种 7D Cartesian Delta Action 在现代 robot learning 和 VLA 中十分常见。

---

# 11. Absolute 与 Relative 的真正区别

这里存在一个很容易产生的误区。

一种常见理解是：

```text
Absolute = 最终目标
Relative = 瞬时控制
```

这个直觉只能算部分正确。

Relative action 确实经常用于短周期的闭环运动，因此看起来更像“当前瞬时动作”。

但是，它的本质定义不是“瞬时”。

真正的区别在于 reference frame。

Absolute Action 描述：

> 目标绝对值是多少。

Relative Action 描述：

> 相对于当前状态改变多少。

因此：

```text
Absolute：
目标是多少

Relative：
从当前状态变化多少
```

例如：

$$
p^*=
[0.50,0.20,0.30]
$$

属于 absolute target。

而：

$$
\Delta p=
[0.02,-0.01,0]
$$

属于 relative target。

一个 delta action 完全可以描述未来一小段运动，而不一定是严格意义上的瞬时 velocity command。

---

## 阶段思考 D：第一次纠偏

### 问题

Absolute Action 和 Relative Action 的核心区别是什么？

### 一个容易出现的不完整回答

```text
Absolute 是最终目标；
Relative 是当前瞬时控制。
```

### 纠偏

这个回答抓住了使用场景的一部分特点，但混淆了：

```text
Relative Action
```

和：

```text
Velocity Action
```

Relative 描述的是“相对当前状态的变化”。

Velocity 描述的是“单位时间内状态变化的速度”。

二者不能直接画等号。

例如：

$$
\Delta x=0.03
$$

表示：

> 相对当前位置移动 3 cm。

而：

$$
\dot x=0.03
$$

如果单位是 m/s，则表示：

> 以 0.03 m/s 的速度运动。

这是两个不同物理量。

---

# 12. Gripper Action

Manipulation action 通常不仅需要控制机械臂的运动，还需要控制夹爪。

可以定义：

$$
g_t\in\{0,1\}
$$

例如：

```text
0 = open
1 = close
```

也可以采用连续形式：

$$
g_t\in[-1,1]
$$

用于表示不同的 opening width 或 command。

因此，一个常见的 manipulation action 可以写成：

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

前六维：

```text
末端如何运动
```

最后一维：

```text
夹爪如何动作
```

---

# 13. 从 VLA 输出到机器人执行：完整控制链

Action Representation 只有放入真实机器人控制链中才真正有意义。

假设 VLA 输入：

```text
RGB Image
+
Language Instruction
+
Robot State
```

语言指令为：

```text
Pick up the cup.
```

模型输出：

$$
a_t=
[
0.02,
-0.01,
-0.03,
0,
0.05,
0,
1
]
$$

它可以解释为：

```text
EEF 向某方向移动 2 cm
侧向移动 -1 cm
下降 3 cm
发生少量旋转
Gripper close
```

此时 VLA 并没有直接告诉 7 个机械臂关节各自如何转动。

真正的数据流是：

```text
VLA
↓
Cartesian Action
↓
EEF Target
↓
IK / Controller
↓
Joint Command
↓
Robot
```

---

# 14. 正运动学与逆运动学

机械臂的正运动学解决：

> 已知 joint configuration，末端在哪里？

即：

$$
x=f(q)
$$

其中：

- q：关节配置；
- x：末端状态。

逆运动学（Inverse Kinematics, IK）反过来：

> 已知希望末端到达的位置或位姿，关节应该处于什么状态？

即：

$$
q^*=f^{-1}(x^*)
$$

实际机器人存在多解、无解、约束等问题，因此这里的 f⁻¹ 主要用于表达概念，而不是说明它一定是普通意义上的唯一解析逆函数。

因此，如果 VLA 输出 EEF target：

$$
x^*
$$

机器人控制链可以写成：

$$
x^*
\rightarrow
IK
\rightarrow
q^*
$$

随后：

$$
q^*
\rightarrow
Joint\ Controller
\rightarrow
\tau
\rightarrow
Robot
$$

---

## 阶段思考 E

### 问题

如果 VLA 已经输出 Cartesian EEF Target，为什么不能通常直接把这个数值发送给电机？

### 解析

因为电机并不知道：

```text
x = 0.43 m
```

意味着自己应该如何转动。

末端空间状态与各个关节之间存在机器人运动学关系。

必须经过：

```text
EEF Target
↓
Kinematic / Control Mapping
↓
Joint Command
↓
Motor
```

这正是机器人软件与控制系统存在的重要原因。

---

# 15. Cartesian Velocity 与 Jacobian

如果 VLA 输出的不是 EEF position，而是 Cartesian velocity：

$$
\dot x^*
$$

此时不能简单使用位置 IK 来理解转换。

机器人微分运动学满足：

$$
\dot x
=
J(q)\dot q
$$

其中：

- ẋ：末端 Cartesian velocity；
- q̇：joint velocity；
- J(q)：当前 joint configuration 下的 Jacobian。

如果希望根据目标 Cartesian velocity 求 joint velocity，可以使用：

$$
\dot q^*
=
J^\dagger(q)\dot x^*
$$

其中：

- J†：Jacobian pseudoinverse；
- ẋ*：VLA 或上层控制器给出的 Cartesian velocity target；
- q̇*：得到的 joint velocity command。

因此应区分：

```text
EEF Position Target
→ IK
→ Joint Position

Cartesian Velocity Target
→ Jacobian Mapping
→ Joint Velocity
```

---

## 阶段思考 F：第二次纠偏

### 问题

已知：

$$
\dot x
=
J(q)\dot q
$$

如果 VLA 输出 Cartesian velocity，应该如何获得 joint velocity？

### 一个常见的不精确回答

```text
通过逆运动学解算。
```

### 纠偏

从广义机器人学语境看，它确实属于从 task space 到 joint space 的逆映射问题。

但在当前数学形式下，更准确的说法是：

> 使用 Jacobian inverse 或 pseudoinverse 完成 velocity mapping。

即：

$$
\dot q^*
=
J^\dagger(q)\dot x^*
$$

因此阅读机器人论文时，要根据变量类型判断：

```text
Pose / Position
还是
Velocity
```

不要把所有 Cartesian → Joint 的过程都笼统称为同一种 IK。

---

# 16. Policy Frequency 与 Control Frequency

这是连接 VLA 与传统控制最重要的系统概念之一。

假设：

```text
VLA inference frequency = 10 Hz
```

意味着模型大约每 100 ms 更新一次 action。

但底层机器人控制器可能运行在：

```text
100 Hz
500 Hz
1000 Hz
```

甚至更高。

因此：

```text
VLA 10 Hz
≠
Robot Motor Control 10 Hz
```

例如，VLA 在某一时刻产生：

$$
q^*
$$

随后，在下一次 VLA inference 到来之前，底层控制器仍然可以持续计算：

$$
e_q=q^*-q
$$

并不断更新：

$$
\tau
=
K_p(q^*-q)
+
K_d(\dot q^*-\dot q)
$$

因此完整时间尺度可以理解为：

```text
VLA:
|------100 ms------|------100 ms------|

Controller:
|1ms|1ms|1ms|1ms|1ms|1ms|1ms|...

Motor:
持续执行底层控制
```

这构成了现代机器人中典型的 hierarchical control。

---

## 阶段思考 G

### 问题

为什么 VLA inference = 10 Hz，并不意味着机器人只能以 10 Hz 控制电机？

### 解析

因为 VLA 只是在低频更新 reference。

底层 controller 会在两次 VLA inference 之间继续高速 tracking。

因此：

```text
Policy Frequency
```

和：

```text
Control Frequency
```

不是同一个概念。

这个区别在后续研究 VLA latency、asynchronous inference、Action Chunking 时非常重要。

---

# 17. 与无人机控制系统的对应关系

对于具有控制背景的读者，可以将机械臂 VLA 与无人机强化学习系统直接类比。

无人机：

```text
RL Policy
↓
Velocity / Position / Yaw Target
↓
PX4
↓
Attitude / Rate Controller
↓
Motor
```

机械臂：

```text
VLA
↓
EEF / Joint Target
↓
IK / Robot Controller
↓
Joint Command
↓
Motor
```

因此：

```text
VLA ≈ High-level Learned Policy

PX4 / Robot Controller ≈ Low-level Controller
```

它们共同构成完整 closed-loop robot system。

---

# 18. Action Representation 对学习本身的影响

Action representation 不是单纯的数据格式选择，因为 policy 学习的是：

$$
\pi(o_t)\rightarrow a_t
$$

所以 aₜ 的定义直接决定模型要拟合什么规律。

如果 aₜ 是：

$$
q^*
$$

模型学习：

> 当前 observation 下，各个 joint 应该到哪里。

如果 aₜ 是：

$$
\Delta x
$$

模型学习：

> 当前 observation 下，EEF 应该相对当前位置怎样变化。

如果 aₜ 是：

$$
\tau
$$

模型甚至需要直接学习：

> 当前状态下，各 joint 应施加什么力矩。

这些学习目标具有不同的：

```text
物理语义
任务抽象层级
对 robot embodiment 的依赖
控制稳定性要求
训练难度
泛化能力
```

因此，阅读 VLA 论文时，Action Representation 应当是最先确认的设计之一。

---

# 19. Absolute Action 与 Delta Action 的学习差异

假设机器人在两个不同场景中抓杯子。

第一次杯子位置为：

```text
x = 0.4
```

第二次：

```text
x = 0.6
```

如果使用 absolute action，模型需要学习：

> 在不同 observation 下应该输出不同的绝对位置。

如果使用 delta action，则可以学习更局部的行为模式：

```text
杯子在右边
→ 向右移动一点

杯子在下方
→ 向下移动一点
```

例如：

$$
\Delta x=0.02
$$

这种局部 action pattern 在不同绝对位置下可能具有相似语义。

因此 relative action 往往具有更强的局部 closed-loop 特性。

但它不是没有缺点。

如果连续多步：

$$
\Delta x_1,\Delta x_2,\ldots,\Delta x_T
$$

都包含微小误差，那么 cumulative error 也可能逐渐积累。

因此不能简单总结为：

```text
Delta 一定优于 Absolute
```

更准确的理解是：

> 不同 action representation 向模型引入了不同的 inductive bias。

也就是：

> 它让模型更容易学习某一类运动规律，同时可能牺牲另一些性质。

---

# 20. Joint Space 与 Cartesian Space 的泛化差异

Joint-space action 与具体 robot morphology 强绑定。

例如，模型学习：

```text
Joint 3 +0.1 rad
```

这句话只在 joint 3 的机械意义已经确定时才有意义。

换成另一个 embodiment：

```text
Joint 数量不同
Joint axis 不同
Link length 不同
Kinematic chain 不同
```

原来的 action 可能完全失去意义。

相比之下：

```text
EEF 向右移动 2 cm
```

具有更加通用的任务空间语义。

Franka 可以执行：

$$
\Delta x=0.02
$$

UR5 同样可以执行：

$$
\Delta x=0.02
$$

尽管二者内部对应的 joint motion 完全不同。

因此：

$$
\boxed{
Cartesian\ Action
\text{通常比 Joint Action 更容易跨 embodiment 对齐}
}
$$

---

# 21. Cartesian Action 仍不能彻底解决 Cross-Embodiment

这里不能进一步得出：

> 只要所有机器人都使用 Cartesian action，cross-embodiment 问题就解决了。

因为 embodiment difference 远不止 joint space。

不同机器人可能拥有：

```text
不同自由度
不同 workspace
不同 link length
不同最大速度
不同 payload
不同 gripper
不同坐标系
不同 reachability
单臂 / 双臂
固定机械臂 / 移动机械臂
平行夹爪 / 灵巧手
```

例如：

> EEF 向右移动 30 cm。

对于一个机器人可能可达。

对于另一个机器人可能已经超过 workspace。

再例如：

> 关闭 gripper。

平行夹爪和五指灵巧手的动作表示完全不同。

因此：

```text
Cartesian Representation
↓
降低 embodiment dependence
```

但不能：

```text
完全消除 embodiment gap
```

后续 Cross-Embodiment Learning 还需要解决 action normalization、robot-specific interfaces、heterogeneous datasets 等问题。

---

## 阶段思考 H

### 问题

为什么 Cartesian action 比 joint action 更适合 cross-embodiment，但仍然无法统一所有机器人？

### 解析

第一层原因是：

```text
Cartesian action 描述 task-space motion
```

因此与具体 joint index 的绑定较弱。

但更深入的一层是：

```text
不同机器人虽然可以共享“末端运动”的语义，
却不一定共享完全相同的物理能力。
```

因此需要区分：

```text
Semantic Alignment
```

和：

```text
Physical Executability
```

前者可能通过 Cartesian space 得到改善。

后者仍受 embodiment 强烈影响。

---

# 22. 一个完整 Pick-and-Place 示例

现在将本章所有内容放进一个完整机械臂抓取任务中。

任务：

```text
Pick the box from basket A and place it into basket B.
```

当前输入：

```text
RGB Image
Language Instruction
Robot Joint State
Gripper State
```

VLA policy：

$$
a_t=\pi(I_t,l,s_t)
$$

假设采用 7D Cartesian Delta Action：

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

某一时刻输出：

$$
a_t=
[
0.03,
-0.02,
-0.01,
0,
0.05,
0,
1
]
$$

假设当前 EEF position：

$$
p_t=
[0.40,0.10,0.30]
$$

则新的位置目标：

$$
p_{t+1}^*
=
p_t+\Delta p_t
$$

得到：

$$
p_{t+1}^*
=
[0.43,0.08,0.29]
$$

再结合 orientation delta，得到新的 EEF pose target。

随后：

```text
EEF Pose Target
↓
IK / Operational Space Controller
↓
Joint Target
↓
Joint Controller
↓
Torque / Motor Command
↓
Robot moves
```

机器人运动后，相机获得新的图像。

于是：

```text
New Image
+
New Robot State
↓
VLA
↓
New Action
```

形成 closed-loop。

完整流程：

```text
Observation
↓
VLA
↓
Cartesian Delta
↓
EEF Target
↓
IK / Controller
↓
Joint Motion
↓
Environment Changes
↓
New Observation
↓
VLA
```

这里可以看到：

> VLA 并没有取代机器人控制学，而是成为传统控制系统上方的 learned policy。

---

# 23. Action 表示对应不同控制层级

可以建立一个近似的层级直觉：

```text
Higher-level

EEF Pose
Cartesian Delta

↓

Joint Position

↓

Joint Velocity

↓

Joint Torque

Lower-level
```

这不是绝对严格的理论分类，但非常有助于理解系统接口。

总体而言：

越往上：

```text
更接近任务空间
更接近行为语义
对具体 motor dynamics 依赖较弱
```

越往下：

```text
更接近 robot embodiment
更接近 dynamics
控制频率要求更高
```

大型 VLA 通常更加适合工作在中高层 action representation 上，而把高频动力学控制留给传统机器人 controller。

---

# 24. 容易混淆的概念

## 24.1 Joint Position 与 EEF Position

Joint Position：

$$
q=
[q_1,q_2,\ldots,q_n]
$$

描述：

> 机器人各个 joint 的 configuration。

EEF Position：

$$
p=
[x,y,z]
$$

描述：

> 机械臂末端在 Cartesian space 中的位置。

两者之间由 Forward / Inverse Kinematics 联系。

---

## 24.2 EEF Position 与 EEF Pose

Position 只有：

```text
Where
```

Pose 包括：

```text
Where + Orientation
```

因此抓取任务通常更加关心 Pose。

---

## 24.3 Relative Action 与 Velocity Action

Relative Action：

$$
\Delta x
$$

描述：

> 相对于当前状态变化多少。

Velocity：

$$
\dot x
$$

描述：

> 单位时间变化多少。

这是本章真实出现过的一个典型误区，必须明确区分。

---

## 24.4 IK 与 Jacobian Velocity Mapping

Position / Pose Target：

$$
x^*
\rightarrow q^*
$$

通常使用：

```text
Inverse Kinematics
```

Velocity：

$$
\dot x^*
\rightarrow\dot q^*
$$

通常使用：

$$
\dot q^*
=
J^\dagger(q)\dot x^*
$$

不能因为二者都是 Cartesian → Joint，就完全混为同一个数学过程。

---

## 24.5 7 DoF Robot 与 7D Action

一台 7 DoF 机械臂可能具有：

```text
7 个 joint variables
```

但一个 7D Cartesian Action 也可能表示：

```text
3 translation
+
3 rotation
+
1 gripper
```

二者的“7”没有必然关系。

---

# 25. 阅读 VLA 论文时如何检查 Action Representation

以后拿到一篇新的 VLA 论文，建议优先检查以下问题。

第一，Action 在什么空间？

```text
Joint Space
还是
Cartesian Space
```

第二，它控制什么物理量？

```text
Position
Velocity
Torque
```

第三，是：

```text
Absolute
还是
Relative / Delta
```

第四，orientation 如何表示？

```text
Euler
Quaternion
Rotation Matrix
Axis-Angle
```

第五，gripper 如何表示？

```text
Binary
Continuous
Position
```

第六，策略输出以后如何执行？

```text
IK
Jacobian
Operational Space Controller
Joint Controller
```

第七，policy frequency 是多少？

第八，low-level control frequency 是多少？

第九，动作是否与特定 embodiment 强绑定？

这些设计往往直接影响模型能否迁移到新的机器人平台。

---

# 26. 本章内容在 VLA 学习路线中的位置

完整知识链可以写成：

```text
Robot Policy
↓
Transformer
↓
Vision-Language Model
↓
Behavioral Cloning
↓
Distribution Shift
↓
Robot Dataset
↓
Action Representation   ← 本章
↓
Action Chunking
↓
Diffusion / Flow Matching Policy
↓
VLA
↓
Fine-tuning / Agent / RL
```

前面的 Dataset 章节回答：

> 训练数据里 action 长什么样？

本章进一步回答：

> 这些 action 在机器人系统里到底意味着什么？

而下一阶段 Action Chunking 会开始回答：

> 为什么现代 robot policy 往往不是只预测一个 action，而是一次预测一段连续动作？

因此 Action Representation 是连接：

```text
Robot Learning
```

和：

```text
Robot Control
```

之间非常关键的一座桥梁。

---

# 27. 研究视角：为什么 Action Space 是一个真正的研究设计选择

对于工程实现，Action Space 决定机器人接口。

对于学习算法，它还进一步决定：

```text
模型拟合什么目标
数据分布是什么样
动作是否平滑
是否容易闭环修正
是否容易迁移
是否容易跨 embodiment
```

因此，Action Representation 并不是预处理阶段随意选一个格式。

尤其在 modern VLA 中，它与以下研究问题直接相关：

```text
cross-embodiment training
action normalization
robot adaptation
fine-tuning
generalization
real-time control
action chunking
```

例如，一个模型即使拥有很强的 vision-language representation，如果 action space 与目标机器人的控制接口不匹配，仍然无法直接部署。

这也是实际跨平台 VLA 复现中非常常见的问题：

> Vision 和 Language 可能已经对齐，但 Action Interface 并没有真正对齐。

---

# 28. 综合思考

## 综合问题 1

为什么不能只根据 action vector 的 dimension 判断它是什么控制方式？

### 解析

因为相同维度可能对应完全不同的 physical semantics。

例如：

```text
7D Joint Position
```

和：

```text
7D Cartesian Delta + Gripper
```

都可能是七维，但分别对应：

```text
Joint Space
```

和：

```text
Task Space
```

---

## 综合问题 2

如果一个 VLA 输出：

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

至少可以直接读出哪些信息？

### 解析

可以判断：

第一，它采用 Cartesian action。

第二，它采用 relative / delta representation。

第三，前六维控制 EEF motion。

第四，最后一维控制 gripper。

因此可以概括为：

```text
7D Cartesian Delta Action
```

---

## 综合问题 3

为什么 Cartesian Action 比 Joint Action 更有利于 generalization？

### 解析

因为它弱化了对具体 joint configuration 的依赖，使策略更加直接地表达：

```text
机器人末端应该如何运动
```

而不是：

```text
某个机器人上的第几个关节应该转多少
```

但这种 generalization 仍然受到 workspace、gripper、DoF、robot dynamics 等 embodiment difference 的限制。

---

## 综合问题 4

VLA inference frequency 很低，为什么机器人仍然可以稳定运动？

### 解析

因为实际系统采用分层控制。

```text
Slow Policy
↓
Reference
↓
Fast Controller
↓
Robot
```

VLA 只需要周期性更新 reference，而传统 low-level controller 可以高速执行 tracking。

---

## 综合问题 5

用一句话概括 VLA 与传统机器人控制器的关系。

### 解析

最准确的心智模型是：

> VLA 负责决定机器人接下来应该怎样运动，传统机器人控制器负责把这个目标稳定、实时地转化为物理运动。

---

# 29. 本章总结

本章最重要的不是记住若干 action 名称，而是建立从 learned policy 到 physical robot 的完整控制链。

第一，VLA 的输出并不是天然固定的。Action 可以定义在 joint space，也可以定义在 Cartesian space，还可以是 position、velocity、torque、absolute target 或 relative delta。

第二，Action Representation 决定了 policy 学习的物理目标。Joint action 更接近具体 robot embodiment，Cartesian action 更接近 task-space motion。

第三，VLA 通常不会直接控制 PWM。更加典型的系统结构是：

```text
VLA
↓
EEF / Joint Target
↓
IK / Jacobian / Controller
↓
Joint Command
↓
Motor
```

第四，必须区分不同数学映射。Position / Pose 到 joint configuration 主要对应 IK，而 Cartesian velocity 到 joint velocity 通常对应 Jacobian velocity mapping。

第五，Policy Frequency 与 Control Frequency 是两个层级。大型 VLA 可以低频推理，而底层 robot controller 仍然高速运行。

第六，Cartesian action 可以缓解 cross-embodiment 的 action alignment 问题，但不能彻底解决 embodiment gap。

最终应形成如下心智模型：

```text
Perception / Language
↓
VLA Policy
↓
Action Representation
↓
Robot Kinematics / Controller
↓
Actuator
↓
Physical World
↓
New Observation
```

VLA 是完整机器人闭环系统中的一个 learned policy，而不是整个控制系统本身。

---

# 30. 关键词

**Action Space — 动作空间 — 策略允许输出的机器人控制变量集合**

**Action Representation — 动作表示 — 如何用数值向量表达机器人动作**

**Joint Position — 关节位置 — 各关节的目标角度或位移**

**Joint Velocity — 关节速度 — 各关节期望运动速度**

**Joint Torque — 关节力矩 — 直接作用于关节动力学层的控制量**

**End-Effector, EEF — 末端执行器 — 机械臂真正与环境交互的末端部分**

**EEF Position — 末端位置 — 末端在 Cartesian space 中的位置**

**EEF Pose — 末端位姿 — Position 加 Orientation**

**Cartesian Space — 笛卡尔空间 — 使用末端位置和姿态描述机器人运动**

**Joint Space — 关节空间 — 使用各个关节变量描述机器人状态与动作**

**Absolute Action — 绝对动作 — 直接给出目标状态的绝对值**

**Relative Action — 相对动作 — 给出相对于当前状态的变化量**

**Delta Pose — 位姿增量 — 当前 EEF pose 应发生多少变化**

**Gripper Action — 夹爪动作 — 控制夹爪开合或宽度**

**Forward Kinematics, FK — 正运动学 — 从 joint configuration 计算 EEF pose**

**Inverse Kinematics, IK — 逆运动学 — 从 EEF target 求对应 joint configuration**

**Jacobian — 雅可比矩阵 — 描述 joint velocity 与 Cartesian velocity 之间的局部映射**

**Jacobian Pseudoinverse — 雅可比伪逆 — 常用于将 Cartesian velocity 映射为 joint velocity**

**Policy Frequency — 策略频率 — 学习策略重新产生 action 的频率**

**Control Frequency — 控制频率 — 底层 controller 更新控制量的频率**

**Embodiment — 机器人本体 — 机器人具体的机械结构、传感器和执行机构**

**Cross-Embodiment — 跨本体 — 在不同 robot morphology 和 control interface 之间学习或迁移**

**Hierarchical Control — 分层控制 — 高层策略产生 reference，低层 controller 高频执行**

---

## 本章最终心智模型

面对任何新的 VLA 系统，不应首先问：

> 模型用了多少参数？

而应先问：

```text
它看到了什么？
↓
它预测的 action 到底是什么？
↓
Action 位于 Joint Space 还是 Cartesian Space？
↓
是 Absolute 还是 Relative？
↓
是 Position、Velocity 还是 Torque？
↓
如何转换成机器人真正能够执行的 command？
↓
底层 controller 如何运行？
↓
换一个 robot 后，这个 action representation 还能否成立？
```

能够自然回答这些问题，才意味着真正理解了 VLA 的 Action Space，而不仅仅是知道几个机器人控制术语。
