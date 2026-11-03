# 22｜Agent + VLA：从 VLA Primitive 到具备规划、记忆与恢复能力的机器人系统

## 1. 本章导言

Vision-Language-Action（VLA）模型把视觉、语言与机器人状态映射为动作，是现代通用机器人策略的重要路线。然而，当机器人从“完成一个局部操作”走向“在真实环境中长期完成复杂任务”时，仅靠一个端到端 VLA 往往还不够。

一个 VLA 可以根据当前图像和语言指令不断重新预测动作，但真实机器人系统还必须回答一系列更高层的问题：

- 当前到底应该做哪一个子任务？
- 哪些步骤应该交给 VLA，哪些步骤应该交给确定性的运动控制器？
- 调用 VLA 之前，当前状态是否落在它熟悉的能力范围内？
- 一次 skill 执行结束后，系统如何判断它真的成功了？
- 失败是因为目标位置变了、视角不合适、抓取失误，还是任务本身需要重新规划？
- 失败以后应该 retry、re-ground、re-stage，还是换一个 skill？
- 过去的成功和失败经验是否应该被记住？
- Skill Library 应该固定不变，还是能够随着经验自动增长？
- 为什么单个 skill 已经很可靠，长程任务的整体成功率仍然很低？

这些问题共同构成了 Agent + VLA、Hierarchical Robotics 和 Agentic Robotics 的核心研究空间。

本章的目标不是简单把“大语言模型 Agent”接在 VLA 前面，而是建立一个更准确的系统心智模型：

```text
Task Planning
↓
World State / Grounding
↓
Skill Selection
↓
Analytic Skill / Learned Skill / VLA Skill
↓
Robot Execution
↓
Feedback
↓
Verification
↓
Failure Diagnosis
↓
Recovery / Replanning
↓
Memory
↓
下一轮决策
```

最终需要理解的并不是“Agent 会不会说出一个合理计划”，而是：

> 如何围绕一个并不完美的 VLA，组织规划、控制、反馈、记忆与恢复机制，使机器人能够在真实世界中持续完成任务。

---

## 2. 问题从哪里来：为什么端到端 VLA 还不够

### 2.1 普通 VLA 的基本形式

一个典型 VLA 可以抽象为：

$$
a_{t:t+H}
\sim
\pi_{\mathrm{VLA}}
\left(
a
\mid
o_t,\,
l,\,
s_t
\right)
$$

其中：

- o_t：当前视觉 observation；
- l：语言指令；
- s_t：机器人 proprioception 或状态；
- a_{t:t+H}：未来一段 action chunk；
- H：动作预测长度。

例如任务为：

```text
“把红色药盒放进右侧篮子。”
```

VLA 可能直接从当前 RGB 图像、机器人状态和语言指令开始，连续执行：

```text
找到药盒
→ 靠近
→ 抓取
→ 抬起
→ 移动
→ 找到目标篮子
→ 放置
```

这种方案结构简洁，但它隐含地要求同一个模型同时承担：

- 语义理解；
- 目标识别；
- 空间推理；
- 长程规划；
- 接触操作；
- 失败检测；
- 失败恢复；
- 动作生成。

对于真实机器人，这种“一个 policy 负责所有事情”的设计很容易遇到能力边界问题。

---

### 2.2 Agent + VLA 的基本变化

Agent + VLA 不再要求高层系统直接输出每一个机器人动作，而是先输出“下一步应该调用哪个能力”。

可以写成：

$$
k_t
\sim
\pi_H
\left(
k
\mid
o_t,\,
l,\,
M_t
\right)
$$

其中：

- π_H：高层 policy 或 planner；
- k_t：当前选择的 skill；
- M_t：当前 memory 或 task state。

随后，被选中的 skill 再产生真正的机器人动作：

$$
a_{t:t+H}
\sim
\pi_{k_t}
\left(
a
\mid
o_t,\,
g_t
\right)
$$

其中：

- g_t：当前 subgoal；
- π_{k_t}：被选中的 skill policy；
- a_{t:t+H}：机器人真正执行的动作序列。

因此两层的输出不是同一种东西：

```text
高层 Agent 输出：
“现在抓药盒。”

低层 VLA 输出：
未来若干步 Δx、Δy、Δz、ΔR、gripper action。
```

最核心的区别是：

> Agent 决定“做什么”，VLA 负责“把某一类复杂动作做出来”。

---

## 3. Hierarchical Robot System：把 Agent + VLA 放回传统机器人体系

Agent + VLA 并不是完全脱离经典机器人学的新体系。

传统机器人本来就存在多层控制结构：

```text
任务规划
↓
路径规划
↓
轨迹规划
↓
位置 / 速度控制
↓
电流控制
↓
电机
```

不同层解决不同时间尺度的问题。

例如：

```text
高层：
把药盒放进篮子 B

中层：
靠近药盒
抓取
移动到篮子
释放

低层：
Δx
Δy
Δz
ΔR
gripper command
```

Agent + VLA 只是把其中一些层替换成了现代学习系统：

```text
LLM / VLM Agent
↓
Skill / Subgoal
↓
VLA / Learned Policy / Analytic Controller
↓
Joint / Cartesian Command
↓
Robot Controller
↓
Actuator
```

因此，Agent + VLA 可以被理解为现代版本的 Hierarchical Control。

---

## 4. 阶段思考 A：高层与低层到底如何区分

### 思考 1

Agent + VLA 和普通端到端 VLA 最大的结构区别是什么？

### 解析

只回答“架构不一样”还不够。真正的区别在于决策变量的抽象层级不同。

普通 VLA 通常直接预测动作：

$$
(o_t,l,s_t)
\rightarrow
a_{t:t+H}
$$

而 Agent + VLA 先选择 skill：

$$
(o_t,l,M_t)
\rightarrow
k_t
$$

再由 skill 产生动作：

$$
k_t
\rightarrow
a_{t:t+H}
$$

因此，高层 policy 和低层 policy 的输出空间本质不同。

---

### 思考 2

为什么“Agent 输出 skill/subgoal，而 VLA 输出 action”不是简单的语言表述差异？

### 解析

因为它们控制的是不同抽象层。

例如：

```text
Agent：
GRASP(box)

VLA：
Δx = +2 mm
Δy = -1 mm
Δz = -3 mm
……
```

前者是语义任务决策，后者是机器人执行动作。

如果把二者混为一谈，就很难理解 Hierarchical Robotics 中 planner、skill policy 和 controller 的边界。

---

### 思考 3

为什么从桌面左侧移动到药盒附近，通常更适合 analytic controller，而不一定要使用 π0.5？

### 解析

因为这种 free-space motion 往往具有：

- 明确几何目标；
- 清晰约束；
- 可建模运动学；
- 可验证轨迹；
- 较低不确定性。

传统 motion planning、IK 和 position controller 往往能够更准确、更稳定、更低计算成本地完成任务。

这并不意味着 VLA 不能完成 free-space motion，而是：

> 当问题已经有可靠结构化解法时，没有必要让大型 learned model 重新猜一次。

---

## 5. Skill Library：Agent 真正调用的是什么

### 5.1 Skill 不是简单函数名

最简单的 Skill Library 可能写成：

```text
MOVE_TO(x, y, z)
ROTATE(yaw)
OPEN_GRIPPER()
CLOSE_GRIPPER()
NAVIGATE(target)
VLA_ACT(instruction)
```

但一个真正可用于 Agent Planning 的 skill，最好至少包含：

```text
name
precondition
input
execution
termination condition
failure condition
```

可以抽象为：

$$
k
=
\left(
\text{name},
\text{precondition},
\text{input},
\text{execution},
\text{termination},
\text{failure}
\right)
$$

例如一个 VLA 抓取 skill：

```text
name:
    grasp_with_vla

precondition:
    object visible
    object reachable
    robot approximately staged

input:
    object semantic description

execution:
    VLA_ACT("grasp the red medicine box")

termination:
    object lifted successfully

failure:
    object still on table
    object lost
    timeout
```

因此 skill 不是“可以调用的一段代码”这么简单，而是：

> 一个带适用条件、执行机制、终止条件和失败定义的能力单元。

---

### 5.2 为什么 skill 必须有 precondition

假设机器人离药盒很远，目标还处于图像边缘。

如果 Agent 直接调用：

```text
VLA_ACT("grasp the medicine box")
```

可能没有意义。

因为抓取 skill 实际上隐含：

```text
目标必须可见
目标必须可达
末端必须处于合理初始位置
相机视角不能严重偏离训练数据
```

可以将这些条件粗略写成：

$$
s_t
\in
\mathcal{S}_{\mathrm{skill}}
$$

其中：

- s_t：当前机器人和环境状态；
- 𝒮_skill：该 skill 的适用状态集合。

只有当当前状态进入这个集合时，调用 skill 才具有较高成功概率。

---

## 6. Tool Use：选什么 skill 与给它什么参数

Agent 的动作不只是选择 skill，还包括生成 skill 参数。

可以写成：

$$
(k_t,\phi_t)
=
\pi_H
\left(
o_t,l,M_t
\right)
$$

其中：

- k_t：Skill Selection；
- φ_t：Argument Generation。

例如：

```text
Skill Selection:
MOVE_TO

Argument Generation:
x = 0.42
y = -0.13
z = 0.25
```

或者：

```text
Skill Selection:
VLA_ACT

Argument Generation:
"grasp the red medicine box"
```

因此机器人 Agent 中的 Tool Use，本质上包含两个问题：

```text
用哪个工具？
+
工具参数是什么？
```

---

## 7. Semantic Grounding：从“药盒”到机器人可以操作的物理实体

### 7.1 语言理解不等于机器人定位

用户说：

```text
“抓红色药盒。”
```

语言模型可以理解“红色药盒”是什么，但机器人真正执行 MOVE_TO 或 GRASP 时还需要知道：

```text
它是哪一个物体？
它在哪里？
它与相机是什么关系？
它与机器人坐标系是什么关系？
```

因此 Semantic Grounding 的任务是把：

```text
语言语义
↓
现实实体
↓
空间位置
```

连接起来。

---

### 7.2 Grounding 的一个典型数据流

相机首先得到：

```text
red medicine box
bbox = [410, 230, 520, 390]
```

如果系统具有 3D 感知能力，可以得到相机坐标系下的位置：

$$
p_c
=
(x_c,y_c,z_c)
$$

再通过坐标变换：

$$
p_w
=
T^w_c
p_c
$$

其中：

- p_c：目标在 camera frame 下的位置；
- p_w：目标在 world frame 下的位置；
- T^w_c：camera frame 到 world frame 的变换。

最终系统得到：

```text
object:
medicine_box_1

world position:
(x, y, z)
```

此时 planner 才能够生成一个真正可执行的 MOVE_TO。

---

## 8. World State：Agent 不应该只看一句语言指令

一个实用 Agent 往往需要显式维护 World State。

例如：

```text
medicine_box:
    visible = True
    position = ...
    grasped = False

blue_basket:
    visible = True
    position = ...

robot:
    ee_pose = ...
    gripper = open

task:
    current_subgoal = grasp_box

history:
    grasp_attempt = 1
```

可以抽象为：

$$
W_t
=
\left\{
O_t,\,
R_t,\,
G_t,\,
H_t
\right\}
$$

其中：

- O_t：对象状态；
- R_t：机器人状态；
- G_t：当前 goal / subgoal；
- H_t：执行历史。

于是 Agent 的决策基础从：

```text
当前图像 + 用户指令
```

升级为：

```text
当前观察
+
显式世界状态
+
任务进度
+
历史经验
```

---

## 9. 阶段思考 B：Grounding 与 Re-grounding

### 思考 1

为什么 Grounding 不能只在任务开始时做一次？

### 解析

机器人一旦与环境发生交互，世界状态可能变化。

例如：

```text
第一次定位：
box = (x₁, y₁, z₁)

抓取失败后：
机器人碰到了 box

新的位置：
box = (x₂, y₂, z₂)
```

如果系统仍然使用旧位置，就会产生 stale world state。

因此失败或环境变化后应该：

```text
Observe
↓
Re-detect
↓
Re-ground
↓
Update World State
↓
Replan
```

---

### 思考 2

Re-grounding 和 Re-staging 有什么区别？

### 解析

两者非常容易混淆。

Re-grounding 解决：

```text
“目标现在在哪里？”
```

它改变的是系统对世界状态的估计。

Re-staging 解决：

```text
“机器人应该从什么状态重新开始调用 VLA？”
```

它改变的是机器人本身的起始状态或机器人—目标相对关系。

因此：

```text
Re-grounding：
更新目标位置

Re-staging：
主动改变机器人状态
```

---

## 10. Feedback：为什么机器人 Agent 必须显式验证结果

### 10.1 执行完不等于成功

假设 VLA 执行：

```text
approach
↓
close gripper
↓
lift
```

但实际上夹爪夹空。

如果没有结果验证，系统仍然可能继续：

```text
move to basket
↓
release
```

机器人会完整执行一套动作，却没有完成任务。

因此真实机器人系统必须区分：

```text
Skill execution completed
```

与：

```text
Task effect achieved
```

这就是 Success Verification 的必要性。

---

## 11. 两层 Closed Loop

VLA 本身可以是闭环的，但这不代表整个 Agent 系统已经具备 task-level closed loop。

### 11.1 Action-Level Closed Loop

```text
Camera
↓
VLA
↓
Action Chunk
↓
Robot
↓
New Observation
↓
VLA
```

它解决：

> 下一段动作应该如何根据新 observation 调整？

---

### 11.2 Task-Level Closed Loop

```text
Agent
↓
Select Skill
↓
Execute Skill
↓
Verify Result
↓
Update World State
↓
Select Next Skill
```

它解决：

> 当前 subgoal 是否真的完成？如果没有，下一步任务层面应该怎么变化？

这两种 closed loop 处于不同层级。

---

## 12. Failure Detection 与 Failure Diagnosis

一个简单 verifier 可以写成：

$$
V
\left(
o_{t+1},
g_t
\right)
\rightarrow
y_t
$$

其中：

$$
y_t
=
\begin{cases}
1, & \text{success}\\
0, & \text{failure}
\end{cases}
$$

Verification 回答：

```text
“成功了吗？”
```

而 Diagnosis 回答：

```text
“为什么失败？”
```

例如：

```text
Verifier:
药盒没有被抓起
→ failure

Diagnosis:
目标位于图像边缘
+
末端初始位置偏远
→ staging failure
```

只有 Verification 而没有 Diagnosis，系统很难选择正确 recovery。

---

## 13. Failure Signal 可以来自哪里

### 13.1 机器人状态

例如：

- gripper width；
- joint state；
- end-effector pose；
- force / torque；
- motion completion。

如果目标是抓取，夹爪闭合量与实际状态不一致，可能提示夹空或接触异常。

---

### 13.2 视觉验证

例如比较抓取前后：

```text
抓取前：
药盒在桌面

抓取后：
药盒是否跟随夹爪移动？
```

可以构造：

$$
P
\left(
\text{grasp success}
\mid
I_t,I_{t+1}
\right)
$$

对于 RGB-only 系统，视觉 verifier 尤其重要。

---

### 13.3 多模态验证

更完整的 verifier 可以同时结合：

$$
V
=
f
\left(
RGB,\,
proprioception,\,
gripper,\,
force,\,
history
\right)
$$

它不仅判断 success / failure，还可以进一步输出 failure type。

---

## 14. Recovery：为什么 Retry 不是完整答案

最低级恢复方式是：

```text
失败
↓
再执行一次相同 skill
```

这就是 Retry。

但如果失败是因为状态本身不满足 skill 的 precondition，那么：

```text
retry
retry
retry
```

只是重复同一个错误。

真正的 recovery 应该首先改变导致失败的条件。

---

### 14.1 Recovery 的层级

可以把常见恢复手段整理成：

```text
Level 0：Retry
↓
Level 1：Re-ground
↓
Level 2：Re-stage
↓
Level 3：Change Skill
↓
Level 4：Replan
↓
Level 5：Human Intervention
```

不同 failure type 对应不同恢复机制。

---

## 15. Re-staging：把状态拉回 VLA 的能力范围

### 15.1 直觉

假设训练数据中常见：

```text
camera height = 80 cm
box near image center
robot close to box
```

部署时却是：

```text
camera height = 95 cm
box near image edge
robot farther away
```

直接调用 VLA 抓取可能失败。

一种方案是重新 Fine-tune VLA。

另一种方案是：

```text
重新观察
↓
重新定位
↓
调整机器人位置
↓
调整相机—物体相对关系
↓
让状态回到训练分布附近
↓
再次调用 VLA
```

这就是 Re-staging。

---

### 15.2 用状态集合解释

假设 VLA 的高可靠区域记为 𝒮_VLA。

当前状态：

$$
s_t
\notin
\mathcal{S}_{\mathrm{VLA}}
$$

Agent 通过 analytic motion：

$$
s_t
\xrightarrow{k_{\mathrm{stage}}}
s'_t
$$

使得：

$$
s'_t
\in
\mathcal{S}_{\mathrm{VLA}}
$$

然后再调用：

$$
\pi_{\mathrm{VLA}}
\left(
a
\mid
s'_t
\right)
$$

因此 Re-staging 可以被理解为：

> State-space distribution correction。

---

## 16. Fine-tuning 与 Re-staging：两种完全不同的 Distribution Shift 处理方式

Fine-tuning 修改 policy：

$$
\pi_\theta
\rightarrow
\pi_{\theta'}
$$

目标是扩大模型能够可靠处理的状态范围。

Re-staging 不修改 VLA 参数，而是修改当前 state：

$$
s_t
\rightarrow
s'_t
$$

使其重新进入 VLA 熟悉区域。

因此可以概括为：

```text
Fine-tuning：
让 VLA 适应世界

Re-staging：
让系统把世界状态组织到 VLA 更会做的位置
```

这也是 VLA Fine-tuning 路线与 Agent + VLA 路线之间最重要的区别之一。

---

## 17. 阶段思考 C：Closed Loop、Failure 与 Recovery

### 思考 1

为什么“VLA 自己不断重新规划动作”仍然不等于完整 Agent 闭环？

### 解析

VLA 的闭环主要存在于 action level。

它可以根据新 observation 重新预测下一段动作，但它未必显式判断：

```text
当前 subgoal 是否完成？
是否连续失败？
是否应该换 skill？
是否应该重新规划整个任务？
```

因此：

```text
Action-Level Feedback
≠
Task-Level Feedback
```

---

### 思考 2

为什么简单 Retry 往往不是好的 recovery？

### 解析

如果当前状态已经落在 VLA 的低成功率区域，那么重复调用相同 policy 并不会改变问题本身。

Recovery 的目标应该是：

```text
改变失败条件
↓
恢复 skill 的适用条件
↓
重新调用
```

---

### 思考 3

一个 RGB-only π0.5 抓取系统最先应该检查哪些 failure signal？

### 解析

至少应关注三类信息：

1. 目标视觉状态  
   药盒是否仍在原位置？是否随夹爪移动？

2. 机器人 / 夹爪状态  
   夹爪闭合程度、末端是否到达预期区域。

3. 抓取前 staging quality  
   bbox 位置、bbox 尺寸、相机高度、视角、末端与目标的相对位置是否接近训练数据分布。

对于 RGB-only 系统，第三类尤其重要，因为系统缺少 depth 或 force 等直接物理信息。

---

## 18. Memory：为什么机器人要“记住经验”

### 18.1 没有 Memory 会发生什么

假设一次任务经历：

```text
状态 A
↓
直接抓取
↓
失败
↓
向左移动 8 cm
↓
再次抓取
↓
成功
```

如果没有 memory，下一次再遇到类似状态 A，机器人可能仍然重复：

```text
先失败一次
↓
再 recovery
```

如果有 memory，可以提前检索：

```text
“类似状态下直接抓取失败，
向左调整后成功。”
```

于是直接：

```text
Re-stage
↓
VLA grasp
```

Memory 的意义因此不是“保存日志”，而是：

> 让过去经验改变当前 planning、skill selection 和 recovery。

---

## 19. Memory 与 Fine-tuning 的区别

Fine-tuning 把经验写入模型参数：

$$
\theta
\rightarrow
\theta'
$$

Memory 则把经验保存在外部：

$$
M
=
\left\{
m_1,
m_2,
\dots,
m_N
\right\}
$$

推理时：

$$
\pi_H
\left(
k_t
\mid
o_t,l,M_t
\right)
$$

模型参数可以保持不变。

因此：

```text
Fine-tuning：
经验进入 weights

Memory：
经验进入 external context / database
```

---

## 20. Task-Specific Memory 与 Global Memory

### 20.1 Task-Specific Memory

记录当前任务的执行状态：

```text
box_A 已完成
box_B 抓取失败 1 次
current_subgoal = grasp box_B
```

它回答：

> 当前任务做到哪里了？

它更接近 Working Memory、Task State 和 Execution History。

---

### 20.2 Global Memory

保存跨 episode 可复用的经验：

```text
相机高度偏高时抓取容易失败

目标位于图像右边缘时，
先向右移动再调用 VLA 成功率更高
```

它回答：

> 过去做类似事情时，学到了什么？

因此可以粗略记成：

```text
Task-Specific Memory
=
当前任务笔记

Global Memory
=
长期机器人经验库
```

---

## 21. Memory 里应该存什么

一条结构化 experience 可以表示为：

$$
m_i
=
\left(
c_i,\,
k_i,\,
r_i,\,
o'_i
\right)
$$

其中：

- c_i：context；
- k_i：执行的 skill；
- r_i：结果；
- o'_i：执行后的 observation。

针对真实机器人，可以进一步保存：

```text
visual context
robot staging state
skill
outcome
recovery
```

例如：

```text
Context:
bbox center = (0.82, 0.54)
bbox scale = 0.08
camera height = high

Skill:
VLA_GRASP

Result:
FAIL

Recovery:
MOVE_LEFT + MOVE_FORWARD

Retry:
SUCCESS
```

这种结构化经验远比单纯记录“上次失败了”更有价值。

---

## 22. Memory Retrieval：为什么不是把所有历史都塞给 Agent

如果系统积累大量经验，不可能每次全部输入 planner。

因此需要：

```text
Current State
↓
Memory Retrieval
↓
Relevant Experiences
↓
Planner
```

可以用 embedding 表示当前状态：

$$
z_t
=
f
\left(
o_t,g_t
\right)
$$

每条 memory 也有：

$$
z_i
=
f
\left(
m_i
\right)
$$

然后寻找最相关经验：

$$
i^*
=
\arg\min_i
d
\left(
z_t,z_i
\right)
$$

Memory Retrieval 的研究问题包括：

- 什么值得存？
- 什么值得取？
- 如何判断两个状态真的“相似”？
- 什么时候更新或遗忘？

---

## 23. Memory 如何从 Failure Recovery 变成 Failure Prevention

假设历史统计显示：

```text
训练视角附近：
grasp success ≈ 92%

相机高度偏高：
grasp success ≈ 24%

降低高度并前移：
grasp success ≈ 81%
```

下一次遇到类似状态时，系统不需要：

```text
VLA
↓
失败
↓
Recovery
```

而可以：

```text
Retrieve Risk Pattern
↓
提前 Re-stage
↓
VLA
```

因此 memory 可以将：

```text
Failure Recovery
```

升级为：

```text
Failure Prevention
```

---

## 24. Memory 与 World Model 的区别

Memory 主要回答：

```text
过去发生了什么？
```

World Model 主要回答：

```text
如果现在执行动作 a，
未来可能发生什么？
```

可以写成：

$$
p
\left(
s_{t+1}
\mid
s_t,a_t
\right)
$$

因此：

```text
Memory
=
Past Experience

World Model
=
Future Prediction
```

两者可以互补，但概念上并不相同。

---

## 25. 阶段思考 D：Memory 的真正作用

### 思考 1

为什么 Memory 和 Fine-tuning 不是同一种“学习经验”？

### 解析

因为 Fine-tuning 改变模型参数，而 Memory 改变 inference context。

一个是 parameter adaptation，一个是 external experience retrieval。

---

### 思考 2

长程任务没有 Task-Specific Memory 会怎样？

### 解析

机器人可能：

- 忘记已完成步骤；
- 重复处理同一个对象；
- 丢失当前 subgoal；
- 无法记录 retry 次数；
- 无法判断是否应该升级 recovery。

因此长程任务需要显式 task state。

---

### 思考 3

针对傅里叶 + π0.5 抓取任务，一条 Global Memory 最值得保存哪些字段？

### 解析

至少包括：

```text
1. Visual Context
2. Robot Staging State
3. Skill
4. Outcome
5. Recovery
```

这五类信息已经足以支持：

```text
相似状态
↓
检索历史
↓
判断风险
↓
提前 Re-stage
```

---

## 26. Skill Discovery：当 Skill Library 不够用怎么办

前面的讨论默认系统已经拥有：

```text
MOVE_TO
GRASP
PLACE
VLA_ACT
```

但真实任务经常出现：

> Agent 知道应该做什么，但系统没有对应能力。

例如目标盒子贴在墙边，直接抓取失败。

真正需要：

```text
先把盒子推出墙角
↓
再抓取
```

如果 Skill Library 中没有 SIDE_PUSH，那么高层 planner 即使“想到了”这个方案，也无法执行。

因此：

```text
Planning Capability
≠
Execution Capability
```

---

## 27. Task Decomposition 与 Skill Discovery

这两个概念必须区分。

### Task Decomposition

把大任务拆成已有 skill：

```text
Put box in basket
↓
MOVE_TO
↓
GRASP
↓
TRANSPORT
↓
PLACE
```

Skill Library 没有变化。

---

### Skill Discovery

发现一个新的 reusable behavior：

```text
侧推物体
```

并将它加入：

```text
SIDE_PUSH
```

因此：

```text
Task Decomposition
=
如何组合已有能力

Skill Discovery
=
能力集合本身如何扩张
```

---

## 28. 新 Skill 可以从哪里来

### 28.1 Demonstration

从大量 demonstration 中发现反复出现的功能性片段。

---

### 28.2 Failure Recovery

如果某种 recovery 经常重复出现：

```text
move left
↓
lower EEF
↓
retry grasp
↓
success
```

系统可能把它抽象为：

```text
RE_STAGE_FOR_GRASP
```

于是原本只是恢复动作，逐渐成为独立 skill。

---

### 28.3 Trajectory Segmentation

一条长 trajectory：

$$
\tau
=
(a_1,\dots,a_T)
$$

可以被划分为：

$$
\tau
=
\tau^{(1)}
\oplus
\tau^{(2)}
\oplus
\tau^{(3)}
$$

例如：

```text
reach
|
grasp
|
transport
|
insert
```

这些 segment 都可能成为候选 primitive。

---

## 29. 什么样的片段才值得叫 Skill

不是任意 trajectory segment 都是 skill。

一个理想 skill 应该具备：

- 可重复性；
- 语义一致性；
- 可调用接口；
- 明确 precondition；
- 明确 termination；
- 明确 failure；
- 在多个任务中有复用价值。

因此，Skill Discovery 不是“把动作切段”，而是：

> 从行为中发现稳定、可复用、有功能意义的控制抽象。

---

## 30. Long-Horizon Manipulation：为什么单技能强不代表长程任务强

假设每个 skill 的成功率为 p，一个任务需要 N 个 skill。

在极简独立假设下：

$$
P_{\mathrm{task}}
\approx
p^N
$$

例如：

$$
0.95^{10}
\approx
0.60
$$

而：

$$
0.9^{10}
\approx
0.35
$$

这说明即使单个 skill 看起来已经非常可靠，长程任务的整体成功率仍然会快速下降。

---

## 31. Long-Horizon 的困难不只是“步骤更多”

长程任务真正困难在于：

- perception 会变化；
- 物体位置会变化；
- 某一步失败会污染后续状态；
- 前一 skill 的小误差可能在后面才暴露；
- Agent 需要记住已完成步骤；
- Recovery 可能改变原计划；
- Skill 之间存在依赖关系。

因此 Long-Horizon Manipulation 更接近：

```text
Planning
+
Memory
+
Feedback
+
Recovery
+
Reliable Skills
```

而不是简单：

```text
生成更长的 plan
```

---

## 32. Hierarchy 为什么适合 Long-Horizon

一个超长动作任务：

$$
\pi
\left(
a_{1:T}
\mid
o_0,l
\right)
$$

可以被分解成：

$$
g_1,g_2,\dots,g_K
$$

每个 subgoal 再交给一个短程 skill。

于是：

```text
High Level:
What next?

Low Level:
How to execute it?
```

并且每个 skill 之间可以插入：

```text
Verify
↓
Recover
↓
Replan
```

这就是 hierarchy 对长程任务最重要的价值。

---

## 33. Skill Granularity：Skill 应该多大

如果 skill 太细：

```text
move 1 cm
rotate 2°
close gripper 3 mm
```

高层 planner 会被迫处理大量低级动作。

如果 skill 太粗：

```text
prepare breakfast
```

这个 skill 自己又变成一个新的 long-horizon black box。

因此存在：

```text
Skill Abstraction Trade-off
```

比较合理的 skill 往往是中等粒度：

```text
grasp_object
open_drawer
insert_object
place_object
navigate_to_table
```

---

## 34. Fixed Skill Library 与 Learned Skill Library

### 34.1 Fixed Skill Library

优势：

- 稳定；
- 可解释；
- 容易验证；
- 容易 debug。

缺点：

- 能力边界固定；
- 新任务可能缺少必要 skill。

---

### 34.2 Learned / Growing Skill Library

优势：

- 可以适应新任务；
- 可以从经验中扩充能力；
- 具有 continual learning 潜力。

困难：

- 新 skill 是否可靠？
- 是否与已有 skill 重复？
- 应该如何命名？
- 什么情况下应该新建 skill？
- 什么情况下应该合并 skill？
- 新 skill 会不会破坏旧 skill？

因此 Skill Discovery 最终会自然连接到 Continual Learning。

---

## 35. 阶段思考 E：Skill Discovery 与 Long-Horizon

### 思考 1

Task Decomposition 和 Skill Discovery 的本质区别是什么？

### 解析

前者改变 plan，后者改变 skill set。

---

### 思考 2

为什么单个 skill 成功率已经很高，Long-Horizon Task 仍然可能失败？

### 解析

因为任务需要连续完成多个 dependent stages，单步成功率会累乘，而且真实系统还存在状态误差传播、perception drift 和错误 recovery。

---

### 思考 3

Skill 太细和太粗分别有什么问题？

### 解析

太细使 planner 退化成低级 motion planner。

太粗则把困难重新塞回一个黑盒 policy。

因此 Skill Granularity 本身就是系统设计变量。

---

## 36. Learned Re-staging：从人工规则变成可学习 Skill

针对傅里叶 + π0.5 抓取任务，可以把 RE_STAGE_FOR_GRASP 设计成 learned skill：

$$
\pi_{\mathrm{stage}}
\left(
o_t,s_t,g
\right)
\rightarrow
\Delta T_{\mathrm{EEF}}
$$

输入：

- 当前 RGB；
- robot state；
- target object / grasp goal。

输出：

$$
\Delta T
=
\left(
\Delta x,
\Delta y,
\Delta z,
\Delta R
\right)
$$

目标不是直接完成抓取，而是：

> 把机器人送到一个更适合 VLA 开始抓取的状态。

---

### 36.1 最简单训练目标：Behavioral Cloning

如果已有专家 re-staging 数据：

$$
\mathcal{L}
=
\left\|
\Delta T_\theta
-
\Delta T^*
\right\|^2
$$

模型学习专家如何重新摆位。

---

### 36.2 更研究化的目标：最大化下游 VLA 成功率

更有意义的目标不是“动作像不像专家”，而是：

$$
J
\left(
\pi_{\mathrm{stage}}
\right)
=
\mathbb{E}
\left[
P
\left(
\text{VLA success}
\mid
s'
\right)
-
\lambda
C_{\mathrm{motion}}
\right]
$$

其中：

- s'：re-stage 后的状态；
- P(VLA success | s')：下游 VLA 成功概率；
- C_motion：re-staging 运动成本；
- λ：成本权重。

这个目标直接对应：

> 怎样把系统送进 VLA 的高成功率区域？

---

# 37. 完整机器人实例：傅里叶机器人 + π0.5 Pick-and-Place

下面把整章机制放进一个完整任务。

任务：

```text
将药盒从篮子 A 搬运到篮子 B。
```

视觉：

```text
单路 RGB
```

VLA：

```text
π0.5
```

---

## 37.1 第一步：Observe 与 Ground

系统得到当前 RGB。

Grounding 模块识别：

```text
medicine_box
basket_B
```

并估计：

```text
box visual position
box scale
basket position
```

如果有足够 3D 信息，则进一步转换到 robot / world frame。

---

## 37.2 第二步：判断当前状态是否适合调用 VLA

系统根据：

```text
bbox center
bbox scale
camera height
camera pitch
EEF relative position
robot configuration
```

判断：

$$
P
\left(
\text{VLA grasp success}
\mid
s_t
\right)
$$

如果低于阈值：

```text
不要直接调用 π0.5
```

而是：

```text
MOVE_TO
或
RE_STAGE_FOR_GRASP
```

---

## 37.3 第三步：调用 VLA 抓取

状态合适后：

```text
VLA_ACT("grasp the medicine box")
```

π0.5 输出 continuous action chunk。

机器人执行。

---

## 37.4 第四步：Verify

执行结束后检查：

```text
药盒是否离开原位置？
药盒是否跟随夹爪？
gripper state 是否合理？
EEF 是否到达预期区域？
```

如果成功：

```text
进入 transport
```

如果失败：

```text
进入 diagnosis
```

---

## 37.5 第五步：Diagnosis 与 Recovery

例如：

```text
目标位置变化
→ Re-ground

目标仍在原位但视角异常
→ Re-stage

随机夹空
→ Retry

连续失败
→ Change Skill / Replan
```

---

## 37.6 第六步：Transport

对于长距离 free-space motion，可以使用 analytic controller：

```text
MOVE_TO(target basket vicinity)
```

没有必要让 π0.5 完整承担整个搬运过程。

---

## 37.7 第七步：Placement

进入目标区域后：

```text
VLA_ACT("place the medicine box into basket B")
```

执行后再次 Verification。

---

## 37.8 第八步：Store Experience

如果此次经历：

```text
高视角
→ direct grasp failed
→ lower camera / move forward
→ retry succeeded
```

则将 experience 写入 Global Memory。

下次遇到类似情况时提前 Re-stage。

---

## 37.9 整体流程

```text
RGB
↓
Ground Object
↓
Update World State
↓
Retrieve Relevant Memory
↓
Check VLA Applicability
├── 不适合
│   ↓
│ Re-stage / MOVE_TO
│   ↓
└── 适合
    ↓
π0.5 Grasp
↓
Verify
├── Success
│   ↓
│ Analytic Transport
│   ↓
│ π0.5 Place
│   ↓
│ Verify
│
└── Failure
    ↓
Diagnosis
    ↓
Re-ground / Re-stage / Retry / Replan
    ↓
再次执行
↓
Store Experience
```

这就是一个最小但完整的 Agent + VLA 系统。

---

# 38. 与传统控制和 Robot Learning 的联系

整章内容可以放回以下层级：

```text
High-Level Task Planning
↓
Agent / Planner
↓
Skill Selection
↓
Analytic Skill / Learned Skill / VLA
↓
Action / Trajectory Generation
↓
IK / Controller
↓
Actuator
```

其中：

- Agent 解决任务层决策；
- VLA 解决复杂 learned skill；
- analytic controller 解决可建模、可验证的结构化运动；
- low-level controller 负责真实执行。

因此，Agent + VLA 不是“抛弃传统控制”，反而是：

> 把传统控制、学习策略和高层 reasoning 放在各自最合适的层级中。

---

# 39. 容易混淆的概念

## 39.1 VLA Closed Loop 与 Agent Closed Loop

VLA Closed Loop：

```text
Observation
→ Action
→ New Observation
→ Action
```

Agent Closed Loop：

```text
Task State
→ Skill
→ Verification
→ Update Plan
```

核心区别是控制层级。

---

## 39.2 Verification 与 Diagnosis

Verification：

```text
成功了吗？
```

Diagnosis：

```text
为什么失败？
```

前者决定是否继续，后者决定如何 recovery。

---

## 39.3 Re-grounding 与 Re-staging

Re-grounding：

```text
重新估计目标在哪里
```

Re-staging：

```text
主动改变机器人初始状态
```

一个改变 world estimate，一个改变 robot/world configuration。

---

## 39.4 Retry 与 Recovery

Retry：

```text
重复原 skill
```

Recovery：

```text
改变失败条件后再执行
```

Retry 只是 Recovery 的最低级形式。

---

## 39.5 Fine-tuning 与 Re-staging

Fine-tuning：

```text
改变 policy
```

Re-staging：

```text
改变 state
```

两者都可能缓解 distribution shift，但作用对象不同。

---

## 39.6 Memory 与 World Model

Memory：

```text
过去发生过什么？
```

World Model：

```text
未来可能发生什么？
```

---

## 39.7 Task Decomposition 与 Skill Discovery

Task Decomposition：

```text
怎么组合已有 skill
```

Skill Discovery：

```text
如何产生新 skill
```

---

# 40. Agent + VLA 的研究版图

整个方向可以压缩成六类核心研究问题。

## 40.1 Planning

问题：

```text
下一步做什么？
```

核心变量：

$$
W_t,g,M_t
\rightarrow
k_t
$$

研究重点包括：

- task decomposition；
- skill sequencing；
- replanning；
- long-horizon planning。

---

## 40.2 Grounding

问题：

```text
语言里的对象和关系，
到底对应现实中的什么？
```

核心是：

```text
Semantics
→ Entity
→ Geometry
```

---

## 40.3 Skill Selection

问题：

```text
什么时候应该调用 VLA？
什么时候应该调用 analytic controller？
```

可以抽象为：

$$
k^*
=
\arg\max_k
\left[
P
\left(
\text{success}
\mid
s,k
\right)
-
\lambda C(k)
\right]
$$

其中：

- success：执行成功概率；
- C(k)：计算、时间、安全或动作成本；
- λ：成本权重。

这比“contact-rich 一律给 VLA”更加一般。

---

## 40.4 Verification

问题：

```text
系统如何知道 skill 真正完成了目标？
```

这是机器人 Agent 与纯软件 Agent 的关键差别之一。

---

## 40.5 Recovery

问题：

```text
失败以后到底应该改变什么？
```

对应：

```text
retry
re-ground
re-stage
change skill
replan
human intervention
```

---

## 40.6 Capability Growth

问题：

```text
Skill Library 能不能自己增长？
```

从：

```text
Agent uses VLA
```

进一步发展到：

```text
Agent improves VLA
```

甚至：

```text
Agent expands its own skill set
```

---

# 41. 代表性研究脉络

本章涉及的代表性方向可以按思想演化理解。

## 41.1 SayCan

关键思想：

```text
Language Planning
+
Robot Affordance
```

高层语言模型不仅要认为某个 skill “有用”，还要考虑机器人实际上能不能完成。

---

## 41.2 Inner Monologue

关键思想：

```text
Plan
→ Execute
→ Environment Feedback
→ Replan
```

开始显式强调 task-level closed loop。

---

## 41.3 Code as Policies

关键思想：

> LLM 不只选择固定 skill，也可以组合控制 API 和程序逻辑。

---

## 41.4 VoxPoser

关键思想：

```text
Language / VLM Reasoning
→ 3D Spatial Grounding
→ Motion Planning
```

强调语义推理如何落到机器人几何空间。

---

## 41.5 RT-H

关键思想：

```text
Task Language
→ Intermediate Motion Language
→ Robot Action
```

体现 hierarchical representation。

---

## 41.6 Harness VLA

关键思想：

```text
Frozen VLA
+
Analytic Primitives
+
Memory
+
Re-grounding
+
Re-staging
+
Failure Recovery
```

它的核心不是继续修改 VLA 本身，而是：

> 通过外围系统管理 VLA 的能力边界，使 VLA 成为一个可调用、可验证、可恢复的 primitive。

---

## 41.7 Primitive / Skill Discovery 方向

进一步研究：

```text
如何从长轨迹中发现 reusable primitives？
```

以及：

```text
如果现有 primitive 不够，
系统能不能获得新能力？
```

这类工作把 Agent + VLA 从“调用固定技能”推进到“能力集合可增长”。

---

# 42. 常见 Benchmark 应该怎么看

不同 benchmark 测试的能力并不相同，不能简单按一个分数比较。

## 42.1 CALVIN

主要关注：

```text
Language-Conditioned Manipulation
+
Long-Horizon Task Composition
```

适合观察连续任务完成能力。

---

## 42.2 LIBERO

主要关注：

```text
Lifelong Learning
+
Knowledge Transfer
+
Forgetting
```

适合研究不同任务之间的迁移。

---

## 42.3 VLABench

更关注 VLA 在：

```text
World Knowledge
Natural Language Intent
Multi-Step Reasoning
```

方面的能力。

---

## 42.4 RoboCasa365

厨房 manipulation 中同时包含：

```text
Atomic Skill
+
Composite Task
```

适合观察：

> 单技能能力能否组合成长程任务能力。

---

## 42.5 BEHAVIOR-1K

更接近复杂家庭环境：

```text
Navigation
+
Manipulation
+
State Change
+
Long Horizon
```

说明 Agentic Robotics 的最终目标并不是“抓取一次成功”，而是在复杂环境中持续完成任务。

---

# 43. 核心 Open Problems

## 43.1 什么时候应该调用 VLA

关键问题：

$$
P
\left(
\text{success}
\mid
s,k
\right)
$$

如何估计？

这涉及：

- capability boundary；
- OOD detection；
- learned routing；
- uncertainty estimation。

---

## 43.2 Analytic Skill 与 Learned Skill 如何动态分工

不是简单固定：

```text
Free Space → Analytic
Contact → VLA
```

而是需要根据：

- 成功率；
- 计算成本；
- 时间成本；
- 精度；
- 安全性；

动态选择。

---

## 43.3 Verifier 自己是否可靠

如果 verifier 把失败判断成 success：

```text
False Success
```

整个后续 plan 都可能基于错误状态继续执行。

因此 Verification 本身就是研究对象。

---

## 43.4 如何定义 VLA 的能力边界

特别是在 RGB-only 系统中，需要判断：

```text
当前状态是否属于 VLA 高成功率区域？
```

这是 Re-staging 和 Skill Routing 的基础。

---

## 43.5 Re-staging 如何从人工规则变成 Learned Policy

从：

```text
if bbox too far right:
    move right
```

变成：

$$
\pi_{\mathrm{stage}}
\left(
o_t
\right)
$$

并直接优化：

$$
P
\left(
\text{downstream VLA success}
\right)
$$

是一个自然且重要的研究方向。

---

## 43.6 Memory 应该存什么、取什么、忘什么

Memory 过少：

```text
无法复用经验
```

Memory 过多：

```text
检索成本高
噪声大
相似经验可能误导
```

因此需要研究：

```text
What to store?
What to retrieve?
When to forget?
How to update?
```

---

## 43.7 Skill Granularity

Skill 太小，高层 planning 过重。

Skill 太大，skill 自己变成 long-horizon black box。

目前不存在一个普适的最优粒度。

---

## 43.8 Skill Discovery 与 Continual Learning

完整问题不是：

```text
发现一个新 skill
```

而是：

```text
Discovery
↓
Training
↓
Verification
↓
Storage
↓
Reuse
↓
Avoid Forgetting
```

---

## 43.9 Long-Horizon Credit Assignment

假设：

```text
skill 1
skill 2
skill 3
skill 4
skill 5
↓
最终失败
```

失败可能并不是 skill 5 本身造成的。

例如 skill 2 把物体放偏了少量距离，但问题直到 skill 5 才暴露。

因此需要回答：

```text
到底哪一步应该为最终失败负责？
```

这会自然连接到后续 VLA + RL。

---

## 43.10 Planner 与 VLA 最终应该模块化还是联合训练

一种路线：

```text
Agent
+
Frozen / Fine-tuned VLA
```

另一种可能路线：

```text
Planner
+
Subgoal
+
Skill Policy
+
Action
```

被统一学习。

模块化系统可解释、可替换。

统一系统可能获得更强协同能力。

目前两者都有研究价值。

---

# 44. 对 Harness VLA 的最终定位

经过本章后，Harness VLA 不应再被理解为：

```text
“Agent 调用 π0.5 的一个特殊系统。”
```

更准确的定位是：

```text
Fixed Skill Library
+
Frozen VLA Skill
+
Analytic Primitives
+
Memory
+
Re-grounding
+
Re-staging
+
Failure Recovery
```

它代表一种重要研究思想：

> 不急于让 VLA 本体承担所有问题，而是围绕 VLA 建立一个显式的系统层，用规划、控制和恢复机制管理 VLA 的能力边界。

---

# 45. 对傅里叶 + π0.5 项目的研究启发

当前真实问题可以概括为：

```text
训练数据内表现尚可
↓
部署时高度 / 视角变化
↓
出现 OOD
↓
抓取成功率下降
```

在 Fine-tuning 路线中，会问：

```text
如何增加数据？
如何增强？
如何重新训练？
如何提高泛化？
```

而在 Agent + VLA 路线中，会额外问：

```text
当前状态是否应该调用 π0.5？
↓
如果不应该，
如何 Re-stage？
↓
调用后，
怎样 Verify？
↓
失败后，
应该 Retry、Re-ground、Re-stage 还是 Replan？
↓
过去经验能否减少下一次失败？
↓
Re-staging 能否进一步学成一个 Skill？
```

这组问题体现了一个关键视角变化：

> 研究对象从“单独一个 VLA 模型”扩大为“围绕 VLA 构建的完整闭环机器人系统”。

---

# 46. 本章在完整 VLA 知识链中的位置

```text
Robot Policy
↓
Transformer
↓
VLM
↓
Robot Learning
↓
Action Representation
↓
Action Chunking
↓
Diffusion / Flow Matching
↓
VLA
↓
VLA Fine-tuning
↓
Agent + VLA
↓
VLA + RL
```

本章处于：

```text
“模型已经能输出动作”
```

之后。

它研究的不是如何重新发明 VLA，而是：

> 如何让一个已有 VLA 真正成为可组合、可验证、可恢复的机器人能力。

---

# 47. 综合思考

## 综合题 1

为什么 Agent + VLA 的核心不能被概括成“LLM 负责规划，VLA 负责执行”？

### 解析

因为真正的系统还需要：

```text
Grounding
Skill Preconditions
World State
Verification
Diagnosis
Recovery
Memory
Replanning
```

如果缺少这些模块，系统仍然只是“生成计划 + 执行动作”，并不具备可靠的 Agentic Closed Loop。

---

## 综合题 2

如果一个 VLA 在训练分布内抓取成功率很高，但 OOD 状态下失败，Agent + VLA 有哪些不修改 VLA 参数的处理方式？

### 解析

包括：

```text
OOD / applicability detection
↓
Re-ground
↓
Re-stage
↓
换 analytic skill
↓
换 learned skill
↓
Retry
↓
Replan
↓
Retrieve Memory
```

其中最核心的是：

> 主动管理输入 VLA 的 state distribution。

---

## 综合题 3

为什么 Memory、Recovery 和 Skill Discovery 最终会连成一条链？

### 解析

一次失败首先触发 Recovery。

如果某种 recovery 多次成功，系统可以把经验保存到 Memory。

如果这类 recovery 具有稳定、可复用的功能，就可能进一步抽象成新的 skill。

因此：

```text
Failure
↓
Recovery
↓
Memory
↓
Repeated Pattern
↓
Skill Discovery
↓
Skill Library Expansion
```

这是 Agentic Robotics 从“会恢复”进一步走向“会成长”的关键逻辑。

---

## 综合题 4

为什么 Long-Horizon Manipulation 最终不能只靠更强的 Planner？

### 解析

因为长程任务失败并不只来自规划错误，还来自：

- perception error；
- grounding error；
- skill execution error；
- state drift；
- verifier error；
- recovery failure；
- memory error。

所以 Long-Horizon Success 是整个闭环系统的结果，而不是单个 Planner 的结果。

---

# 48. 本章总结

本章最核心的结论可以压缩为六点。

第一，Agent + VLA 的本质是 Hierarchical Robot System。高层 Agent 选择 skill 或 subgoal，低层 VLA、learned policy 或 analytic controller 负责具体执行。

第二，一个真正可用于机器人规划的 skill 不只是函数名，而应该包含 precondition、输入、执行机制、termination 和 failure condition。Skill Selection 和 Argument Generation 共同构成机器人 Tool Use。

第三，真实机器人必须建立 task-level closed loop。VLA 的 action-level feedback 不能替代高层 Verification、Failure Diagnosis、Recovery 和 Replanning。

第四，Re-staging 是理解 Agent + VLA 的关键机制之一。Fine-tuning 通过修改 policy 扩大能力范围，而 Re-staging 通过改变 state，把机器人重新送回 VLA 的高成功率区域。

第五，Memory 使系统从“失败后恢复”进一步发展成“失败前预防”；Skill Discovery 则使系统从“调用固定能力”进一步发展成“扩张自身能力”。

第六，Long-Horizon Manipulation 的难点不是简单增加任务步骤，而是如何让 Planning、Grounding、Skill Execution、Verification、Recovery、Memory 和 Skill Discovery 形成一个可靠闭环。

最终应该建立如下心智模型：

```text
Observe
↓
Ground
↓
Build / Update World State
↓
Retrieve Memory
↓
Plan
↓
Select Skill
↓
Check Preconditions
↓
Execute
↓
Verify
↓
Diagnose
↓
Recover / Replan
↓
Store Experience
↓
Repeat
```

这才是 Agent + VLA 的完整系统观。

---

# 49. 关键词

**Agentic Robotics — 智能体机器人系统 — 通过规划、技能调用、反馈和恢复形成任务级闭环。**

**Hierarchical Control — 分层控制 — 不同抽象层分别处理任务规划、技能执行与低层控制。**

**Skill Library — 技能库 — Agent 可调用的机器人能力集合。**

**Primitive — 原语 — 可独立调用、具有明确功能的基础行为单元。**

**Task Planner — 任务规划器 — 根据目标和当前世界状态决定下一步做什么。**

**Skill Selection — 技能选择 — 从 Skill Library 中选择当前最合适的 skill。**

**Argument Generation — 参数生成 — 为 skill 生成位置、对象、语言指令等具体参数。**

**Semantic Grounding — 语义落地 — 将语言概念映射到真实环境中的对象和空间状态。**

**World State — 世界状态 — 对物体、机器人、任务进度和历史的显式表示。**

**Success Verification — 成功验证 — 判断 skill 执行后目标效果是否真正发生。**

**Failure Diagnosis — 失败诊断 — 判断失败的具体原因。**

**Retry — 重试 — 在基本不改变状态的情况下再次执行同一 skill。**

**Re-grounding — 重新落地 / 重新定位 — 更新目标对象和环境的当前状态估计。**

**Re-staging — 重新摆位 — 主动改变机器人与环境的相对状态，使其重新进入 skill 的高成功率区域。**

**Task-Specific Memory — 任务特定记忆 — 保存当前任务进度、subgoal 和执行历史。**

**Global Memory — 全局记忆 — 保存跨任务、跨 episode 可复用的长期经验。**

**Memory Retrieval — 记忆检索 — 从大量历史经验中找出与当前状态最相关的内容。**

**Skill Discovery — 技能发现 — 从轨迹、示教或 recovery 中发现新的 reusable skill。**

**Skill Granularity — 技能粒度 — Skill 的抽象尺度，过细和过粗都会增加系统困难。**

**Long-Horizon Manipulation — 长程操作 — 需要多个技能连续协作完成的复杂机器人任务。**

**Capability Boundary — 能力边界 — 一个 skill 或 VLA 能够可靠工作的状态范围。**

**OOD Detection — 分布外检测 — 判断当前输入是否偏离模型训练分布。**

**Failure Prevention — 失败预防 — 根据 memory 和 applicability 判断，在调用 skill 前主动降低失败概率。**

**Continual Learning — 持续学习 — 系统不断获得新技能，同时尽量保留已有能力。**

**World Model — 世界模型 — 建模动作如何影响未来状态，而不是仅记录过去经验。**

**VLA as Skill — VLA 作为技能 — 把 VLA 从 whole-task policy 封装为 Agent 可调用的机器人 primitive。**
