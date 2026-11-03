# 21｜Harness VLA：以 Memory-Guided Agent 驾驭 Frozen VLA

## 1. 本章导言：从“训练一个更强的 VLA”转向“设计一个更可靠的系统”

在此前的学习中，VLA（Vision-Language-Action）通常被理解为一种端到端机器人策略：模型接收视觉观测、语言指令和机器人状态，直接预测机器人动作或动作序列。以 π0、π0.5、OpenVLA 等为代表的路线，核心问题通常是如何提升模型本身的表示能力、动作建模能力、数据规模、跨 embodiment 泛化能力以及新任务适配能力。

Harness VLA 提出了一个不同的研究视角。

它并不首先问：

> 如何继续训练一个更强、更泛化的 VLA？

而是问：

> 如果已经有一个能力较强但存在明显分布外失败的 Frozen VLA，能不能不修改它的参数，而是通过 Agent、解析控制器、Memory 和失败恢复机制，让它在更复杂、更长时序、更容易发生分布偏移的任务中可靠工作？

这意味着研究对象从“单一 policy”扩展到了“围绕 policy 构建的完整系统”。

Harness VLA 的核心思想可以先概括为：

```text
Agent 负责 reasoning / grounding / recovery
        ↓
Analytic Primitive 负责非接触、几何明确的运动
        ↓
Frozen VLA 负责局部 contact-rich 操作
```

因此，VLA 不再承担从语言理解、目标绑定、长时序规划、自由空间运输到接触操作的全部责任，而被重新定位为：

> 一个可以被上层 Agent 调用、重试和重新摆位的 contact-rich manipulation primitive。

本章将围绕这一系统思想展开，并重点回答以下问题：

- 为什么端到端 VLA 在部署扰动下容易失败？
- 为什么纯 analytic controller 也无法替代 VLA？
- Agent、Primitive Library 和 VLA_ACT 如何组成一个统一接口？
- Task Specific Memory 与 Global Memory 分别保存什么？
- Retry 与 Re-staging 有什么区别？
- Harness VLA 的实验究竟证明了什么？
- 这种方法提高的是 VLA 自身能力，还是 VLA 的系统可用范围？
- 论文没有解决哪些问题？
- 从研究视角看，哪些方向可以进一步发展？

---

## 2. 问题从哪里来：端到端 VLA 为什么会在真实部署中失效？

### 2.1 端到端 VLA 的标准形式

一个典型 VLA 策略可以写成：

$$
a_{t:t+H}
=
\pi_\theta
\left(
I_t,\,
l,\,
q_t
\right)
$$

其中：

- I_t：当前视觉观测；
- l：语言任务描述；
- q_t：机器人 proprioception，例如末端位姿和夹爪状态；
- a_{t:t+H}：从当前时刻开始预测的一段 action chunk；
- θ：VLA 参数。

这种范式的优势是：模型可以直接从大量机器人轨迹中学习复杂的视觉—动作映射，特别适合传统解析控制器难以写清楚的接触操作。

例如：

- 抓取不规则物体；
- 受约束放置；
- 开门、开抽屉；
- 按按钮；
- 操作水龙头；
- 插入；
- 双臂协作中的复杂接触。

这些行为往往依赖视觉、接触、局部几何和动作连续性，很难被简单地拆成若干固定的几何命令。

### 2.2 VLA 的弱点：部署状态不再属于训练轨迹分布

端到端 VLA 的主要问题并不是“它完全不会做机器人操作”，而是：

> 它往往只在自己熟悉的局部轨迹分布中表现稳定。

部署时可能出现多种偏移：

```text
训练时：
固定物体位置
固定目标绑定
固定视角
固定任务组合
固定前接触状态

部署时：
目标被重新指定
物体位置变化
场景布局变化
长任务需要重新组合
局部接触失败
```

可以抽象为：

$$
p_{\text{deploy}}
\left(
o,l,g
\right)
\neq
p_{\text{train}}
\left(
o,l,g
\right)
$$

其中：

- o：当前 observation；
- l：语言任务；
- g：目标或任务 predicate。

这类问题不只是一种视觉 domain shift，还可能包括：

- semantic retargeting；
- goal re-binding；
- spatial-layout shift；
- long-horizon composition；
- unstable local contact。

端到端 VLA 一旦在长轨迹中某个位置偏离原轨迹，后续又继续根据偏离后的状态做预测，就可能产生典型的 compounding error。

因此，单次局部失败很容易演变为：

```text
局部接触失败
↓
机器人进入陌生状态
↓
VLA 继续输出动作
↓
状态进一步偏离训练分布
↓
整个 rollout 失败
```

---

## 3. 为什么不能只使用 Analytic Primitive？

如果 VLA 在分布外状态下不可靠，一种直接想法是：是否可以完全绕开 VLA，让上层大模型只调用 IK、Cartesian controller、gripper controller 等传统机器人模块？

答案是否定的。

### 3.1 Analytic Primitive 擅长什么？

对于自由空间运动，如果目标位置明确，例如：

```text
当前 EEF 位姿
↓
目标世界坐标
↓
IK / Jacobian / Operational Space Controller
↓
移动到目标位置
```

这类问题有清晰几何结构，因此解析控制器通常：

- 稳定；
- 可预测；
- 易验证；
- 计算成本低；
- 不依赖示教数据。

典型操作包括：

- MOVE_TO；
- MOVE_POSE；
- ROTATE_WRIST；
- ROTATE_PITCH；
- SET_GRIPPER；
- RELEASE；
- NAVIGATE_TO；
- MOVE_BASE。

### 3.2 Analytic Primitive 不擅长什么？

一旦进入 contact-rich manipulation，问题会显著复杂。

例如：

- 一个不规则盒子如何抓住？
- 抓取姿态稍有误差时如何自然纠正？
- 一个物体如何插入狭窄容器？
- 水龙头如何旋转到合适角度？
- 双臂如何协同完成接触过程？

此时，解析控制器需要精确建模：

- 接触位置；
- 摩擦；
- 物体几何；
- 关节约束；
- 操作容差；
- 视觉误差；
- 运动过程中的局部修正。

这正是 learned visuomotor policy 更擅长的区域。

因此，Harness VLA 不选择“VLA 或 analytic control”二选一，而是重新划分职责。

---

## 4. Harness VLA 的核心分工

Harness VLA 最重要的思想可以写成：

$$
\text{System}
=
\text{Agentic Planner}
+
\text{Analytic Primitives}
+
\text{Frozen VLA}
+
\text{Memory}
+
\text{Closed-loop Feedback}
$$

其中三种能力的分工为：

```text
Agent：
理解任务
语义绑定
场景重新 grounding
primitive 选择
失败诊断
retry / recovery

Analytic Primitive：
自由空间运输
姿态调整
导航
释放
重摆位
非接触阶段

Frozen VLA：
抓取
受约束放置
插入
夹具操作
按钮操作
其他 contact-rich phase
```

这种分工并不是简单的“高层规划 + 低层控制”，因为论文刻意把 VLA 和 analytic controller 统一成了一个 Primitive Library。

---

## 5. 阶段思考 A：Harness VLA 与 Fine-tuning 的本质区别是什么？

### 问题

如果一个 Frozen VLA 在新场景中失效，Fine-tuning 与 Harness VLA 分别在改变什么？

### 常见部分正确回答

一种常见表述是：

> Fine-tuning 改变模型分布，而 Harness 把状态分布拉回 VLA 分布内。

这个方向基本正确，但“模型分布”并不是最准确的术语。

### 解析与纠偏

Fine-tuning 的核心是修改模型参数：

$$
\theta
\rightarrow
\theta'
$$

目标是让 VLA 本身对新的任务、机器人或状态分布产生更好的策略输出。

因此：

```text
Fine-tuning：
让模型适应新环境
```

Harness VLA 则保持：

$$
\theta
=
\text{constant}
$$

它通过 Agent 和 analytic primitive 改变：

- 机器人姿态；
- 末端位置；
- 观察视角；
- 局部目标绑定；
- VLA 被调用时的初始状态。

因此：

```text
Harness：
不直接改变模型能力，
而是改变模型被调用时所处的条件。
```

更准确的对比是：

```text
Fine-tuning：
model adaptation

Harness：
system-level orchestration
```

---

## 6. VLA-compatible Local State：重新理解“把状态拉回分布内”

Harness VLA 中非常重要的一个心智模型是：

> VLA 不需要在整个任务状态空间中都可靠，只需要在一个局部、熟悉、适合接触操作的区域中可靠。

可以把部署状态记为 s_OOD。

如果直接调用 VLA：

$$
s_{\text{OOD}}
\xrightarrow{\text{VLA}}
?
$$

可能失败。

Harness 的做法是先通过 analytic primitive 改变状态：

$$
s_{\text{OOD}}
\xrightarrow{\text{Agent + Analytic}}
s_{\text{compatible}}
$$

再调用：

$$
s_{\text{compatible}}
\xrightarrow{\text{VLA}}
s'
$$

这里的 s_compatible 不是论文严格定义的一个概率集合，而是一种工程概念：

> 当前视觉、机器人姿态、目标位置和局部任务状态已经进入 Frozen VLA 更熟悉、更容易成功的 operating region。

例如，某个 VLA 的训练数据中，大部分抓取轨迹都满足：

```text
目标物体位于画面中心附近
EEF 已经靠近目标
视角与训练相似
目标语义清晰
```

部署时如果变成：

```text
目标在画面边缘
EEF 距离目标较远
机器人姿态偏离
相机高度发生变化
```

Harness 不强迫 VLA 自己完成全部纠正，而是：

```text
定位目标
↓
MOVE_TO / ROTATE
↓
形成更熟悉的 pre-contact state
↓
VLA_ACT
```

这就是 re-staging 的核心意义。

---

## 7. Agentic Planner：输出的不是 Action，而是 Primitive Invocation

传统 VLA 输出：

$$
a_{t:t+H}
$$

而 Harness VLA 的 Planner 输出：

$$
c_t
\in
P
$$

其中：

- P：固定 Primitive Library；
- c_t：当前选择的 primitive invocation。

Planner 可以抽象为：

$$
c_t
=
\Pi
\left(
o_t,\,
l,\,
M
\right)
$$

其中：

- Π：Agentic Planner；
- o_t：当前 RGB-D + robot state；
- l：任务语言；
- M：Task Specific Memory 与 Global Memory。

Planner 输出的不是：

- torque；
- joint position；
- joint target；
- action chunk；

而是结构化调用，例如：

```json
{
  "action": "move_to",
  "xyz": [0.1, -0.2, 0.9]
}
```

或者：

```json
{
  "action": "vla_act",
  "prompt": "grasp the black bowl",
  "max_chunks": 2
}
```

因此，Agent 的决策空间已经从连续控制动作变成了一个离散—参数化 primitive space。

---

## 8. 阶段思考 B：MOVE_TO 与 VLA_ACT 为什么可以放在同一级？

### 问题

MOVE_TO 底层可能是 IK 或 Operational Space Controller，而 VLA_ACT 底层是神经网络 VLA。二者实现方式完全不同，为什么 Agent 却可以把它们当作同一层 primitive？

### 容易出现的误区

容易把它们理解成：

> 一个是高层指令，一个是低层模型。

但论文的关键设计恰恰是：从 Agent 的视角，不再暴露这些内部差异。

### 正确心智模型

对 Planner 来说：

```text
MOVE_TO(...)
ROTATE_WRIST(...)
RELEASE(...)
VLA_ACT(...)
```

都是：

> 有固定名称、固定参数、固定执行语义和返回条件的 Tool。

因此：

```text
Planner
   │
   ├── MOVE_TO
   ├── ROTATE
   ├── RELEASE
   └── VLA_ACT
```

VLA 被从“完整机器人 policy”重新包装成了“可调用的工具”。

这也是 Harness 一词的真正含义之一：系统通过标准化接口把异构执行模块封装到统一 runtime 中。

---

## 9. Primitive Library：小而固定的技能词汇

Harness VLA 使用一个固定 Primitive Library，而不是让 Agent 在部署时不断发明新技能。

### 9.1 Analytic Primitive

典型包括：

- MOVE_TO：移动 EEF 到世界坐标目标；
- MOVE_POSE：同时调整位置和姿态；
- ROTATE_WRIST：调整 wrist yaw；
- ROTATE_PITCH：调整 pitch；
- SET_GRIPPER：开合夹爪；
- RELEASE：满足释放条件后打开夹爪。

在 RoboCasa365 这种移动操作环境中，还增加：

- NAVIGATE_TO；
- MOVE_BASE。

### 9.2 Learned Primitive

整个共享 manipulation vocabulary 中最特殊的 learned primitive 是：

```text
VLA_ACT
```

因此可以将 Library 理解为：

$$
P
=
P_{\text{analytic}}
\cup
\left\{
\text{VLA\_ACT}
\right\}
$$

### 9.3 为什么 Library 故意保持小而固定？

很多 Agentic Robotics 系统会不断扩展：

```text
GRASP
INSERT
OPEN_DRAWER
CLOSE_DOOR
TURN_FAUCET
PRESS_BUTTON
...
```

Harness VLA 则故意研究另一个问题：

> 在不扩展 primitive vocabulary 的情况下，仅靠更合理的 primitive composition，能否扩展 Frozen VLA 的可用范围？

因此系统复杂度主要来自：

```text
如何组合 primitive
```

而不是：

```text
拥有多少 primitive
```

这为后续 Agent + VLA 研究提供了一个重要对照维度：

```text
fixed skill library
vs
expandable / learned skill library
```

---

## 10. VLA_ACT：不是一次 Forward，而是局部 VLA Rollout

VLA_ACT 是论文中唯一的 learned primitive。

一个典型调用可表示为：

```json
{
  "action": "vla_act",
  "prompt": "grasp the black bowl",
  "max_chunks": 30,
  "stop": "object_lifted"
}
```

它通常包含三个关键参数：

### 10.1 Prompt

Prompt 定义当前局部 contact-rich 子任务，例如：

```text
grasp the black bowl
```

而不是把整个长任务全部交给 VLA。

### 10.2 max_chunks

VLA 不是只运行一个 action chunk，而是允许连续运行若干 chunk。

如果单次动作预测为：

$$
a_{t:t+H}
$$

那么 VLA_ACT 可以在 primitive 内部连续生成多个 chunk。

max_chunks 的作用是限制：

```text
最多连续调用多少个 action chunk
```

### 10.3 Stop Predicate τ

定义 VLA_ACT 的提前返回条件。

例如：

```text
object_lifted
contact_established
benchmark_predicate
chunk_budget_reached
```

因此，VLA_ACT 更准确地表示为：

$$
\text{VLA\_ACT}
=
\text{local VLA rollout with termination condition}
$$

而不是：

```text
一次 VLA forward
```

---

## 11. 数学机制：Primitive-level Closed Loop

Harness VLA 的执行过程可以写成：

$$
c_t
=
\Pi
\left(
o_t,\,
l,\,
M
\right)
$$

环境执行 primitive：

$$
o_t
\xrightarrow{c_t}
o_{t+1}
$$

随后 Planner 重新读取 observation：

$$
c_{t+1}
=
\Pi
\left(
o_{t+1},\,
l,\,
M
\right)
$$

这一过程持续到：

$$
G=1
$$

或者 primitive / episode budget 耗尽。

其中：

- G：任务完成 predicate；
- c_t：当前 primitive；
- o_t：当前多模态 observation；
- M：memory context。

需要特别强调：

> Harness 的 closed loop 主要发生在 primitive boundary 上。

即：

```text
primitive 执行
↓
返回控制权
↓
Agent 重新观测
↓
下一 primitive
```

它并不意味着 Agent 在 VLA action chunk 的每一个控制周期都参与实时修正。

---

## 12. Task Specific Memory：保存“这个任务怎么做”

Harness VLA 的第一个 Memory 模块是 Task Specific Memory，简称 TSM。

其核心作用是：

> 保存某一个任务在 reference seed 上成功解决后形成的 task-level solution skeleton。

### 12.1 Exploratory Bootstrapping

在参考场景中，Agent 被允许：

- 多次尝试；
- 调整 primitive 顺序；
- 调整 staging；
- 调整 VLA 调用位置；
- 使用 RESET；
- 在更宽松预算下探索。

例如可能出现：

```text
VLA_ACT
↓
失败
↓
MOVE_TO
↓
VLA_ACT
↓
成功抓取
↓
MOVE_TO
↓
RELEASE
↓
任务成功
```

成功后，这套结构被保存。

### 12.2 TSM 保存两类内容

Task Specific Memory 不是一个简单文本，而是由两部分组成。

#### A. JSONL Procedural Trace

保存：

```text
实际执行了哪些 primitive
primitive 顺序是什么
VLA 在什么位置被调用
contact-rich 与 analytic phase 如何切换
```

示意：

```json
{"action":"vla_act","prompt":"grasp the black bowl"}
{"action":"move_to","xyz":[0.12,-0.08,0.92]}
{"action":"release"}
```

这回答：

> What was executed?

#### B. JSON Semantic Summary

保存：

```text
为什么这个策略有效
哪些尝试失败
哪些 primitive 更适合当前任务
出现过哪些恢复行为
哪些错误以后需要避免
```

例如：

```text
strategy:
use VLA for grasping,
then analytic transport and release

avoid:
do not reuse reference xyz values
```

这回答：

> Why did it work?

---

## 13. 阶段思考 C：为什么 TSM 不是轨迹 Replay？

### 问题

既然 reference seed 已经有成功 primitive trace，部署到新 seed 时能否直接照抄原命令？

### 错误心智模型

```text
seed 0：
MOVE_TO(0.12, -0.08, 0.92)

seed 1：
MOVE_TO(0.12, -0.08, 0.92)
```

这种做法相当于 replay 坐标。

一旦物体位置改变，就会失败。

### 正确心智模型

TSM 保存的是：

```text
VLA grasp
↓
analytic transport
↓
release
```

而不是旧场景中的具体空间参数。

因此：

```text
reuse structure
≠
reuse coordinates
```

部署时必须：

```text
读取当前 RGB-D
↓
重新定位 object
↓
重新定位 destination
↓
重新计算 xyz
↓
复用 primitive structure
```

所以 TSM 更接近：

> task-level program skeleton

而不是：

> open-loop motion trajectory

---

## 14. Global Memory：保存“通常什么有效、什么会失败”

Global Memory 与 Task Specific Memory 的作用不同。

### 14.1 TSM 回答

```text
这个 task 之前是怎么做成功的？
```

### 14.2 Global Memory 回答

```text
使用这些 primitive 时，一般什么情况有效？
什么情况容易失败？
失败后应该如何处理？
```

因此 Global Memory 是 task-independent、cross-task 的经验库。

典型成功规则：

```text
contact-rich phase 使用 VLA
稳定抓住后，长距离运输优先用 analytic motion
```

典型 failure model：

```text
如果夹爪闭合但物体没有跟随 EEF 移动，
判断为 empty grasp。
重新定位目标并 re-stage。
```

另一个典型 failure model：

```text
不能只根据视觉接近就宣布任务成功。
必须结合正式 benchmark success signal。
```

---

## 15. Task Specific Memory 与 Global Memory 的本质区别

| 维度 | Task Specific Memory | Global Memory |
|---|---|---|
| 作用范围 | 单个 task | 跨 task |
| 核心内容 | task-level solution skeleton | reusable success rules / failure models |
| 主要形式 | JSONL trace + JSON summary | 规则与经验 |
| 是否包含 primitive 顺序 | 是 | 通常不是主要内容 |
| 是否直接复用坐标 | 否 | 不涉及 |
| 核心问题 | How this task was solved | What usually works or fails |

最简记忆：

```text
TSM：
How this task was solved

GM：
What usually works / fails
```

---

## 16. Retry 与 Re-staging：失败恢复的真正核心

### 16.1 Retry

如果一次 VLA_ACT 失败后，直接再调用一次：

```text
VLA_ACT
↓
fail
↓
VLA_ACT
```

这只是 retry。

### 16.2 Re-staging

Re-staging 会先改变下一次 VLA 的初始条件。

例如：

```text
VLA_ACT
↓
抓空
↓
重新定位 object
↓
MOVE_TO
↓
调整 EEF pose
↓
重新形成 pre-contact state
↓
VLA_ACT
```

数学上可以理解为：

$$
s_1
\neq
s_2
$$

第一次调用：

$$
\text{VLA}(s_1)
\rightarrow
\text{fail}
$$

经过 staging：

$$
s_1
\xrightarrow{\text{analytic}}
s_2
$$

第二次调用：

$$
\text{VLA}(s_2)
\rightarrow
\text{success}
$$

因此：

> Re-staging 的关键不是“多试一次”，而是“主动改变下一次 VLA 调用的状态”。

---

## 17. 失败恢复可以如何理解为 FSM？

从控制背景来看，Harness VLA 很容易映射成 supervisory FSM：

```text
STAGING
   ↓
VLA_ACT
   ↓
VERIFY
 /     \
成功   失败
 ↓      ↓
NEXT   DIAGNOSE
        ↓
     RE-STAGE
        ↓
      VLA_ACT
```

这意味着 Harness VLA 与传统机器人系统中的 supervisory control 思想具有明显联系。

它并不是让神经网络承担所有职责，而是：

```text
高层监督逻辑
+
局部 learned skill
+
确定性 controller
```

共同组成完整闭环。

---

## 18. REPL：Agentic Robotics 的执行接口

Harness VLA 还采用了一个非常工程化的 file-mediated REPL。

REPL 可理解为：

```text
Read
↓
Eval
↓
Print
↓
Loop
```

机器人场景中变成：

```text
读取 observation
↓
Planner 输出 command.json
↓
Environment 执行 primitive
↓
写入新的 state / log / RGB-D
↓
Planner 再次读取
↓
下一 primitive
```

典型文件包括：

```text
command.json
state_01.json
log_01.json
done_01.flag
RGB-D files
Task Specific Memory
Global Memory
```

这种结构与 Coding Agent 很相似：

```text
Coding Agent：
read code
→ execute tool
→ read terminal result
→ choose next tool

Harness VLA：
read observation
→ execute primitive
→ read physical result
→ choose next primitive
```

因此，Harness VLA 本质上把机器人执行问题重构为：

> 一个拥有工具接口、状态、memory、反馈和错误恢复的 agentic runtime。

---

## 19. 完整机器人实例：Pick-and-Place

考虑任务：

```text
把药盒从 Basket A 放入 Basket B
```

### 19.1 端到端 VLA 方案

```text
RGB
+
Instruction
↓
π0.5
↓
完整 action rollout
```

如果部署时：

- 相机高度变化；
- 药盒位置变化；
- EEF 初始位置变化；

则 observation 很可能偏离训练时的状态分布。

### 19.2 Harness 风格方案

可以拆成：

```text
Task：
put the medicine box into Basket B

↓
Agent 识别 medicine box
↓
定位药盒当前空间位置
↓
MOVE_TO 进入 pre-contact state
↓
VLA_ACT("grasp the medicine box")
↓
Verify grasp
↓
MOVE_TO Basket B
↓
VLA_ACT("place the medicine box")
或 RELEASE
↓
Verify task success
```

如果抓取失败：

```text
gripper closed
but object stayed on table
```

则根据 Global Memory：

```text
判断 empty grasp
↓
重新定位 object
↓
重新 staging
↓
VLA_ACT
```

### 19.3 这个例子揭示的核心思想

Harness 并没有让 π0.5 变得更强。

它做的是：

```text
把 π0.5 从完整长时序 policy
缩小为局部 contact-rich specialist
```

从而减少它需要承担的 OOD 责任。

---

## 20. 与传统控制 / Robot Learning 的位置关系

Harness VLA 可以放在如下系统层次中理解：

```text
Natural Language Task
        ↓
Agentic Planner
        ↓
Primitive Selection / Memory / Recovery
        ↓
┌───────────────────────────┐
│ Analytic Primitive        │
│ Frozen VLA Primitive      │
└───────────────────────────┘
        ↓
Action / Action Chunk
        ↓
Low-level Controller
        ↓
Robot
```

与纯端到端 VLA 相比，最大的变化是：

> Task Planning、Primitive Routing、Failure Recovery 不再全部隐含在一个单一神经网络 policy 中。

---

# 21. 实验设计：论文到底想证明什么？

Harness VLA 的实验不是只想证明“成功率更高”，而是围绕三个机制性 claim 展开：

```text
Key Finding 1：
Planner-level semantic re-grounding

Key Finding 2：
Planner-staged VLA invocation and retry

Key Finding 3：
Analytic / VLA division of labor
```

---

## 22. Table 2：标准 LIBERO——Harness 会不会帮倒忙？

标准 LIBERO 中，关键结果是：

| 方法 | Overall Success Rate |
|---|---:|
| OpenVLA | 76.5% |
| π0 | 94.2% |
| πRLinf | 95.3% |
| Harness VLA (CC) | 96.0% |

这个实验最重要的意义不是 95.3% → 96.0% 的提升本身，而是：

> 在加入 Agent、Memory 和 analytic primitive 之后，原本较强的 in-distribution VLA 能力没有被明显破坏。

如果 Harness 在 OOD 上提升很多，却让标准场景性能大幅下降，那么系统就会存在明显代价。

因此 Table 2 更像一个“保持基本能力”的验证。

---

## 23. Table 3：LIBERO-Pro——真正关键的 Distribution Shift 实验

LIBERO-Pro 主要包含两类扰动。

### 23.1 T：Instruction Redirection

任务语言重新指向不同目标。

测试的是：

```text
semantic re-binding
```

### 23.2 S：Position Swap

场景中的物体位置或布局发生变化。

测试的是：

```text
spatial re-grounding
```

主要结果：

| 方法 | Overall |
|---|---:|
| OpenVLA | 0.0% |
| π0 | 0.3% |
| π0.5 | 11.0% |
| AtomVLA | 6.3% |
| RATS | 43.8%* |
| πRLinf | 50.0% |
| Harness VLA (Codex) | 72.1% |
| Harness VLA (CC) | 82.4% |

其中 RATS 的 overall 只基于其报告的六个非 LIBERO-10 cell，因此不能把所有 headline 数字完全等价比较。

真正最干净的比较是：

```text
同一个 frozen πRLinf：
50.0%
↓
加 Harness
82.4%
```

因为底层 VLA 没有变化。

因此，这组结果更直接支持：

> system-level orchestration 本身显著提升了 OOD robustness。

---

## 24. Figure 3：VLA 为什么会“按旧习惯做事”？

Figure 3 展示了两类典型失败。

### 24.1 Semantic Retargeting

视觉场景与训练时相似，但语言目标发生改变。

Frozen VLA 仍然倾向于执行训练时熟悉的目标。

说明：

```text
训练轨迹先验
可能压过
当前语言重新绑定
```

### 24.2 Position Swap

目标位置变化后，Frozen VLA 仍然可能向训练时熟悉区域执行。

Harness 则：

```text
读取新任务
↓
从当前 RGB-D 重新 grounding
↓
analytic staging
↓
局部 VLA_ACT
```

这支持：

> Planner-level semantic re-grounding 可以恢复 task-conditioned behavior。

---

## 25. 阶段思考 D：Open-world Generalization 是否等于任意 OOD Robustness？

### 问题

π0.5 强调 open-world generalization，为什么仍会在 LIBERO-Pro 这类 perturbation 中失败？

### 解析

“模型拥有更丰富的语义和视觉知识”并不等于：

```text
在任意新机器人状态和任意新任务绑定下
都能稳定输出正确动作
```

VLA 的语言和视觉能力可能很强，但 action policy 仍然受 robot trajectory distribution 影响。

因此：

```text
semantic knowledge
≠
task-conditioned motor reliability
```

这也是 Harness VLA 的重要出发点。

---

## 26. Table 4：RoboCasa365——长时序和移动操作

RoboCasa365 进一步引入：

- mobile base；
- kitchen fixture；
- articulated object；
- longer composite routine。

主要结果：

```text
RLDX-1：
30.0%

Harness VLA (Codex)：
57.1%

Harness VLA (CC)：
48.6%
```

Codex 版本相对 direct frozen VLA 提高 27.1 个百分点。

这表明：

> Harness 的价值不仅存在于桌面抓取，还体现在导航、staging、re-localization 与多阶段 contact-rich 操作的组合中。

---

## 27. Table 5：Memory 到底贡献了什么？

一个非常重要的实验是移除 target setting 对应的 Task Specific Memory 和 Global Memory，进行 strict zero-shot evaluation。

在 LIBERO-Pro GOAL 中：

```text
Instruction Redirection：
few-shot 87%
zero-shot 79%

Position Swap：
few-shot 87%
zero-shot 31%
```

### 27.1 如何解释？

语言重定向时，Planner 自身已有较强 semantic reasoning，因此从 87% 降到 79%，没有完全崩溃。

而 position swap 更依赖：

- primitive order；
- staging；
- transport；
- VLA invocation point；
- release；
- recovery。

这些内容正是 Task Specific Memory 的强项。

因此这个实验说明：

> TSM 的主要价值不是保存“答案”，而是保存 task-level primitive organization。

---

## 28. Table 6：RoboTwin C2R——Harness 不是万能外挂

RoboTwin C2R 使用 Frozen LingBot-VLA 作为 backend。

结果：

```text
LingBot-VLA：
50.4%

Harness VLA (Codex)：
58.0%

Harness VLA (CC)：
58.4%
```

提升存在，但不像 LIBERO-Pro 那么巨大。

这反而揭示了一个重要事实：

> 如果任务难点主要集中在复杂的双臂 contact-rich manipulation 本身，那么 Agent 能解决的 semantic / staging / composition 问题有限。

因此：

```text
VLA backend 的局部 contact 能力
仍然决定了系统上限
```

---

## 29. Figure 4：Repeated VLA Invocation 是否真的有用？

Figure 4 将最大允许 VLA primitive invocation 次数记为 k。

横轴：

```text
最大 VLA 调用次数 k
```

纵轴：

```text
累计 task success
```

多个 benchmark 中都观察到：

$$
k\uparrow
\Rightarrow
\text{success rate}\uparrow
$$

随后逐渐饱和。

这支持：

> repeated planner-staged invocation 是可靠性来源之一。

关键点不是“多采样几次相同的 VLA”，而是：

```text
VLA
↓
观察结果
↓
诊断
↓
re-stage
↓
VLA
```

因此每一次调用的初始状态都可能已经变化。

---

## 30. Figure 5：局部失败如何变成可恢复失败？

Figure 5 展示了典型的 adaptive VLA invocation。

例如：

```text
VLA_ACT
↓
奶盒没有稳定进入篮子
↓
Agent 发现不稳定状态
↓
重新调整 EEF
↓
再次 VLA_ACT
↓
最终稳定放置
```

或者：

```text
移动底盘
↓
VLA 抓取 pan
↓
失败
↓
调整 base / arm
↓
VLA 重试
↓
稳定抓取
```

因此：

```text
repeated VLA calls
≠
continuous VLA control
```

而是：

> sparse, planner-selected local contact attempts。

---

## 31. Figure 6：Division of Labor 是否真的存在？

Figure 6 统计：

> 成功 rollout 的最终 benchmark completion predicate 是在 analytic primitive 后触发，还是在 VLA primitive 后触发。

标准桌面任务中，很多任务的最终 success 发生在：

```text
MOVE_TO
或
RELEASE
```

之后。

例如：

```text
VLA grasp
↓
MOVE_TO
↓
RELEASE
↓
Success
```

而 RoboCasa365、RoboTwin 中，一些 terminal operation 本身仍然是 contact-rich，因此更多成功发生在 VLA primitive 内部。

这个结果说明：

> VLA 与 analytic primitive 确实承担不同任务阶段，而不是形式上的模块拆分。

---

## 32. Figure 7：闭环验证比“相信 VLA 自己成功了”更重要

一个典型 rollout：

```text
VLA_ACT
↓
开始向篮子移动
↓
Agent 发现其实没有抓住物体
↓
MOVE_TO 返回
↓
VLA_ACT 重抓
↓
MOVE_TO
↓
RELEASE
```

这说明 Harness 不把 VLA 的“动作意图”当成事实，而是：

```text
执行
↓
观察 actual state
↓
重新判断
```

这使得系统从：

```text
open-loop trust
```

变成：

```text
closed-loop verification
```

---

## 33. 三个实验发现如何串成一个完整机制？

整篇实验可以压缩成：

```text
Key Finding 1
Agent 重新理解当前任务和场景
↓
semantic / spatial re-grounding

Key Finding 2
VLA 可以局部失败、重新摆位、再调用
↓
planner-staged retry

Key Finding 3
非接触阶段不需要让 VLA 负责
↓
analytic / VLA division of labor
```

三者组合后得到：

> Frozen VLA 的系统有效工作范围被扩大。

必须特别区分：

```text
Frozen VLA 本身变强
```

与：

```text
系统更会使用 Frozen VLA
```

Harness VLA 做的是后者。

---

## 34. 阶段思考 E：提高 VLA 能力 vs 扩大 VLA 可用范围

### 问题

为什么不能简单说 Harness VLA “提高了 π0.5 的泛化能力”？

### 解析

因为 VLA 参数没有变化。

因此：

```text
πθ 本身：
没有学习新的 policy

系统：
学会了在更合适的位置调用 πθ
```

可以写成：

```text
提高 VLA 能力：
改变模型本身

扩大 VLA 可用范围：
改变系统组织和调用条件
```

所以更准确的说法是：

> Harness VLA 提高了 system-level robustness，而不等价于提高 Frozen VLA 自身的 intrinsic OOD generalization。

---

# 35. 论文的真正创新点

## 35.1 创新一：重新定义 VLA 的系统角色

传统：

```text
Task
↓
VLA
↓
完整动作序列
```

Harness：

```text
Task
↓
Agent
↓
Primitive Library
├── Analytic Primitive
└── VLA_ACT
```

VLA 被重新定位为：

```text
contact-rich specialist
```

而不是：

```text
monolithic robot policy
```

这是一项重要的系统级 conceptual contribution。

---

## 35.2 创新二：Re-staging，而不是要求模型吞掉全部 OOD

传统思路：

```text
遇到 OOD
↓
训练更强模型
```

Harness：

```text
遇到 OOD
↓
重新定位 / 搬运 / 调整姿态
↓
进入 VLA-compatible local state
↓
再调用 Frozen VLA
```

这与控制中的有效工作域思想非常相似：

> 不要求单一 controller 在全状态空间内都同样可靠，而是先把系统送入其稳定或有效区域。

---

## 35.3 创新三：把 Failure Recovery 从 trajectory-level 下沉到 primitive-level

普通 monolithic VLA：

```text
局部失败
↓
整个 rollout 偏离
```

Harness：

```text
局部失败
↓
当前 primitive 失败
↓
diagnose
↓
re-stage
↓
retry
```

因此：

```text
trajectory-level failure
→
recoverable primitive-level failure
```

这是 Harness 提升可靠性的核心机制之一。

---

## 36. Memory 的创新应该如何评价？

Memory 并不是 Harness VLA 独创的概念。

其价值更多体现在：

```text
Task Specific Memory
+
Global Memory
```

的职责分离。

特别是：

```text
parameterized primitive trace
```

即保存：

```text
如何做
```

而不是：

```text
上一次具体在什么坐标做
```

这种 memory 更适合：

- layout shift；
- seed variation；
- task structure transfer。

因此应把它理解为：

> memory-augmented primitive composition

而不是简单的“Agent 有记忆”。

---

# 37. 论文没有解决什么？

理解 limitation 是研究阅读中最重要的一步之一。

---

## 37.1 High-level Planner 与 Low-level VLA 之间仍然是粗粒度闭环

Harness 的 Agent 通常在 primitive 结束后重新获得控制权。

所以：

```text
Agent frequency
低

VLA / controller frequency
高
```

可以画成：

```text
Agent：
|------ primitive ------|------ primitive ------|

Low-level：
↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
```

因此它解决的是：

```text
primitive-level closed loop
```

而不是：

```text
control-timestep-level reactive correction
```

如果 VLA_ACT 内部 action chunk 在第 10 个控制周期已经明显偏离，Agent 未必能立即介入。

---

## 37.2 没有 Joint Fine-tuning

当前系统主要是：

```text
Frozen VLA
+
LLM Planner
+
hand-defined primitive interface
+
memory
```

并没有通过 environment reward 对整个系统联合优化。

因此：

```text
什么时候调用 VLA
什么时候 retry
什么时候 re-stage
```

仍主要依赖：

- LLM reasoning；
- prompt；
- memory；
- hand-designed interface。

这更接近：

```text
reasoned orchestration
```

而不是：

```text
learned orchestration policy
```

---

## 37.3 复杂视觉结构推理仍有限

Harness 强依赖：

```text
RGB
+
Depth / World Map
```

Planner 需要：

```text
从 RGB 识别目标
↓
选取 pixel
↓
从 depth / world map 获得 metric xyz
```

因此在：

- clutter；
- 严重遮挡；
- 透明物体；
- 反射；
- 多层空间关系；

等条件下，grounding 本身仍可能失败。

---

## 37.4 Contact-rich OOD 仍然没有被解决

Harness 主要擅长绕开：

- semantic shift；
- spatial shift；
- composition shift；
- recoverable staging shift。

但如果核心接触技能本身超出了 VLA 能力，例如：

```text
训练时只抓规则盒子
部署时要求抓柔性透明袋
```

那么：

```text
MOVE_TO
ROTATE
re-stage
```

再多也可能无效。

因此：

> Harness 不能替代底层 VLA 自身的 skill learning。

---

## 37.5 Planner 自身成为新的 Failure Source

引入 Agent 后，系统也引入新的错误来源：

- object grounding 错；
- primitive 选错；
- coordinate estimation 错；
- failure diagnosis 错；
- memory retrieval 错；
- premature success judgment。

因此：

```text
VLA failure
```

被部分降低的同时，

```text
Agent failure
```

成为新的系统问题。

---

## 37.6 对 RGB-D / Metric Localization 的依赖

Harness 中很多 analytic primitive 要求：

```text
MOVE_TO(x,y,z)
```

因此必须获得可靠 metric localization。

对于只有单路 RGB 的系统，如果没有：

- depth；
- multi-view geometry；
- calibration；
- 3D detector；
- learned metric grounding；

就不能直接复制 Harness 的完整流程。

这说明论文提出的是一种系统范式，而不是可以不加修改地迁移到所有硬件平台的现成方案。

---

# 38. 一个重要批判：Harness 到底“学习”了多少？

论文会描述：

> Planner 通过探索和 Memory 学习 primitive 的 operating range。

但这里的“learn”并不是典型神经网络训练意义上的：

$$
\min_\phi
\mathcal{L}
$$

而更接近：

```text
LLM
+
execution traces
+
semantic summary
+
success / failure rules
```

形成经验归纳。

因此：

```text
Harness 当前：
implicit / language-based operating-range reasoning

尚未做到：
explicit learned routing policy
```

这个差距本身就是很重要的研究空间。

---

# 39. 可以进一步研究的问题

## 39.1 Learned Primitive Routing

现在：

$$
c_t
=
\Pi_{\text{LLM}}
\left(
o_t,l,M
\right)
$$

可以进一步显式学习：

$$
c_t
\sim
\pi_\phi
\left(
c
\mid
o,l,M
\right)
$$

让一个 learned router 学习：

- 当前状态应该用 analytic 还是 VLA；
- 什么时候 retry；
- 什么时候 re-stage；
- 什么时候停止。

---

## 39.2 学习 VLA 的 Operating Region

可以定义一个成功 critic：

$$
C(o)
=
P
\left(
\text{VLA succeeds}
\mid
o
\right)
$$

如果：

$$
C(o_t)
<
\delta
$$

则先 re-stage。

直到：

$$
C(o_t)
\ge
\delta
$$

再调用 VLA。

这样可以把：

```text
VLA-compatible state
```

从模糊概念变成：

```text
可学习、可估计的成功概率
```

---

## 39.3 学习 Re-staging Policy

可以专门学习：

$$
\pi_{\text{stage}}
\left(
u
\mid
s
\right)
$$

其目标不是直接完成任务，而是：

> 让下一次 VLA 调用更容易成功。

可写成：

$$
\max_u
P
\left(
\text{VLA success}
\mid
T(s,u)
\right)
$$

这是一种典型的：

```text
meta-control around VLA
```

---

## 39.4 Harness + Fine-tuning

Harness 与 Fine-tuning 并不冲突。

可以构建：

```text
Harness rollout
↓
发现反复失败状态
↓
收集 failure-centric data
↓
LoRA / SFT
↓
更新 VLA
↓
重新部署 Harness
```

这样形成：

```text
system-guided data collection
```

相比盲目增加 demo，更容易针对真正的 failure distribution。

---

## 39.5 Harness + RL

如果引入 reward：

$$
r_t
=
\begin{cases}
1,&\text{success}\\
0,&\text{otherwise}
\end{cases}
$$

则可以优化：

- primitive selection；
- retry timing；
- staging；
- stop predicate；
- VLA invocation timing。

从而形成：

```text
Agentic VLA orchestration as RL
```

这正好连接后续 VLA + RL 研究方向。

---

## 39.6 Skill Discovery

当一些 primitive sequence 高频重复：

```text
MOVE_TO
→ ROTATE
→ VLA_ACT
→ VERIFY
```

可以自动抽象为：

```text
STAGED_GRASP
```

于是系统从：

```text
fixed primitive library
```

走向：

```text
growing / learned skill library
```

这会成为后续 Agent + VLA 研究中的重要分支。

---

# 40. 与三条研究路线的关系

Harness VLA 可以帮助建立一个更大的研究框架。

### 40.1 VLA Fine-tuning

核心是：

```text
改模型
```

即：

$$
\theta
\rightarrow
\theta'
$$

让 VLA 本身适应新任务。

### 40.2 Agent + VLA

核心是：

```text
改系统
```

保持 VLA 参数不变，研究：

```text
when
where
how
VLA is used
```

### 40.3 VLA + RL

核心是：

```text
利用环境反馈继续优化行为
```

三者可以理解为三个不同层级：

```text
Model Level
Fine-tuning
↓
System Level
Agent + VLA
↓
Interaction / Reward Level
VLA + RL
```

它们并不互斥，可以组合。

---

# 41. 容易混淆的概念

## 41.1 Harness VLA vs VLA Fine-tuning

Harness：

```text
θ 不变
改变调用条件与系统组织
```

Fine-tuning：

```text
θ 改变
让模型适应新数据
```

---

## 41.2 Retry vs Re-staging

Retry：

```text
同一条件下再试一次
```

Re-staging：

```text
先改变机器人 / 视觉 / 局部状态
再调用 VLA
```

---

## 41.3 Task Specific Memory vs Trajectory Replay

TSM：

```text
复用 primitive structure
```

Trajectory Replay：

```text
复用具体动作或坐标
```

二者不是一回事。

---

## 41.4 VLA-compatible State vs “严格落回训练分布”

VLA-compatible local state 是一种工程 operating-region 概念，并不意味着能精确证明：

```text
当前状态属于训练数据概率分布内部
```

它更强调：

> 通过 staging，使当前局部条件更加接近 VLA 熟悉和容易成功的操作状态。

---

## 41.5 System Robustness vs VLA Generalization

Harness 提升：

```text
system-level robustness
```

不等于：

```text
Frozen VLA intrinsic generalization
```

这是论文评价中最重要的区分之一。

---

# 42. 本章内容在 VLA 知识链中的位置

到这里，VLA 知识链可以继续延伸为：

```text
Robot Policy
↓
Transformer / VLM
↓
Behavioral Cloning
↓
Action Representation
↓
Action Chunking
↓
Diffusion / Flow Matching
↓
RT-2 / OpenVLA / π0 / π0.5
↓
VLA Fine-tuning
↓
VLA Inference / Latency
↓
Harness VLA
↓
Agent + VLA
↓
VLA + RL
```

Harness VLA 的特殊意义是：

> 它第一次把学习重点从“VLA 模型内部怎么工作”明显转向“VLA 在完整机器人系统中应该承担什么职责”。

---

# 43. 综合思考

## 思考 1

为什么 Table 2 中 Harness 96.0% 与 πRLinf 95.3% 的差异很小，但这个实验仍然重要？

### 解析

因为它证明：

> Harness 获得 OOD robustness 的同时，并没有明显牺牲原本的 in-distribution 操作能力。

---

## 思考 2

为什么“πRLinf 50.0% → Harness 82.4%”比“Harness 超过 OpenVLA”更能证明 Harness 的贡献？

### 解析

因为前者底层 Frozen VLA 相同，改变的主要变量是 Harness 本身，因此更接近控制变量实验。

---

## 思考 3

如果最终 task success 在 RELEASE 后触发，能否说明 VLA 没有贡献？

### 解析

不能。

例如：

```text
VLA 完成稳定抓取
↓
MOVE_TO 完成搬运
↓
RELEASE
↓
Success
```

最后一步由 analytic primitive 完成，并不否定 VLA 在最困难 contact phase 中的贡献。

---

## 思考 4

如果一个 VLA 在抓取透明软袋时完全不会，Harness 能否仅靠 re-staging 解决？

### 解析

通常不能。

这是：

```text
contact skill 本身 OOD
```

而不是：

```text
semantic / spatial / staging OOD
```

Harness 只能绕开一部分 OOD，而不能替代底层技能学习。

---

## 思考 5

为什么可以说 Harness 的核心不是“Memory”，而是“职责重分配”？

### 解析

Memory 是实现 Planner 稳定调用 primitive 的重要机制，但更深层的变化是：

```text
VLA：
从 end-to-end policy
变成 contact-rich tool
```

这重新定义了 VLA 在机器人系统中的位置。

---

# 44. 研究视角：阅读类似论文时应关注什么？

以后阅读 Agent + VLA 论文，应主动检查：

```text
1. VLA 是 frozen 还是 fine-tuned？
2. Planner 是 LLM、VLM 还是 learned policy？
3. Planner 输出 code、skill id 还是 primitive invocation？
4. Primitive Library 是 fixed 还是 expandable？
5. 哪些操作交给 analytic controller？
6. 哪些操作交给 VLA？
7. VLA invocation 如何终止？
8. 是否支持 retry？
9. retry 前是否 re-stage？
10. success / failure 如何检测？
11. 是否有 Task Specific Memory？
12. 是否有 Global Memory？
13. Memory 存 strategy 还是 trajectory？
14. 是否重新 grounding？
15. 是否依赖 depth / world map？
16. Agent 的闭环频率是多少？
17. VLA primitive 内部是否可实时干预？
18. benchmark 测的是 semantic shift、spatial shift 还是 contact shift？
19. 对比是否控制了底层 VLA backend？
20. system robustness 与 model generalization 是否被清楚区分？
```

如果能够自然提出这些问题，就已经不再只是“读懂 Harness VLA”，而是在形成 Agentic Robotics 的研究判断框架。

---

# 45. 本章总结

Harness VLA 的价值不在于训练了一个更强的 VLA，而在于重新思考：

> 一个已有 VLA 应该如何被整个机器人系统使用。

本章最核心的逻辑关系可以压缩为：

```text
端到端 VLA
在 deployment perturbation 下容易失效
↓
并不是所有阶段都需要 VLA
↓
非接触、几何明确的动作交给 analytic primitive
↓
接触复杂操作交给 Frozen VLA
↓
Agent 负责 semantic grounding、primitive composition 和 recovery
↓
Task Specific Memory 保存任务结构
Global Memory 保存跨任务经验
↓
VLA 失败后不直接让 rollout 崩溃
而是 diagnose → re-stage → retry
↓
系统级 robustness 提升
```

最终应形成以下五个核心结论。

第一，Harness VLA 并没有改变 Frozen VLA 参数，因此它提升的是 system-level robustness，而不是 VLA intrinsic generalization。

第二，VLA 在 Harness 中不再是“整个机器人系统的大脑”，而是一个局部 contact-rich tool。

第三，最重要的恢复机制不是简单 retry，而是 re-staging：主动改变下一次 VLA 调用的初始状态。

第四，Task Specific Memory 保存的是 task-level solution skeleton，Global Memory 保存的是 cross-task success rules 与 failure models。

第五，Harness 的真正研究价值在于“职责重分配”：让不同类型的问题交给最适合的模块处理，而不是强迫一个单一神经网络 policy 覆盖所有机器人任务结构。

---

# 46. 关键词

**Frozen VLA — 冻结 VLA — 参数不更新，只作为现有执行策略使用。**

**Agentic Planner — 智能体规划器 — 根据任务、当前观测和 Memory 选择下一 primitive。**

**Primitive Library — 原语库 — Agent 可以调用的固定机器人操作集合。**

**Analytic Primitive — 解析原语 — 基于 IK、几何或模型控制实现的确定性操作。**

**VLA_ACT — VLA 操作原语 — 把 Frozen VLA 封装成局部 contact-rich manipulation tool。**

**Contact-rich Manipulation — 富接触操作 — 需要复杂局部接触、视觉与动作协调的操作。**

**VLA-compatible Local State — VLA 兼容局部状态 — 更接近 VLA 熟悉 operating region 的局部操作状态。**

**Task Specific Memory — 任务特定记忆 — 保存某一任务成功 primitive composition 与策略总结。**

**Global Memory — 全局记忆 — 保存跨任务可复用的 success rules 与 failure models。**

**Re-grounding — 重新落地/重新绑定 — 根据当前观测重新确定语言目标和空间位置。**

**Retry — 重试 — 失败后再次尝试同一类操作。**

**Re-staging — 重新摆位 — 在再次调用 VLA 前先改变机器人、视角或局部状态。**

**Failure Recovery — 失败恢复 — 诊断失败并采取重新定位、摆位或重试。**

**Primitive-level Closed Loop — 原语级闭环 — 每个 primitive 执行后重新观察并做下一决策。**

**REPL — Read-Eval-Print Loop — 读取状态、执行命令、读取结果并继续决策的循环接口。**

**Division of Labor — 职责分工 — Agent、analytic controller 和 VLA 分别承担不同类型问题。**

**System-level Robustness — 系统级鲁棒性 — 通过系统组织提高成功率，而不等同于模型本身能力提升。**

**Learned Primitive Routing — 学习式原语路由 — 用可训练策略决定何时调用不同 primitive。**

**Operating Region — 有效工作域 — 某一 VLA 更容易稳定成功的局部状态区域。**

**Skill Discovery — 技能发现 — 从重复 primitive 组合中自动形成新的可复用技能。**
