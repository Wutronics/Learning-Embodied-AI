# 16｜OpenVLA：从开放 VLM 到可微调的机器人策略

## 1. 本章导言

OpenVLA 是理解现代 Vision-Language-Action（VLA）模型工程化形态的一座重要桥梁。

在此前的知识链中，RT-2 说明了一个关键事实：预训练 Vision-Language Model（VLM）不仅能够处理图像和语言，还可以通过将机器人动作离散化为 token 的方式，被改造成真正输出机器人动作的策略模型。Open X-Embodiment 则进一步解决了另一个问题：机器人数据虽然来自不同 embodiment、不同任务、不同控制接口，但仍然可以通过数据规范化、action 统一处理和 dataset mixture 的方式，被用于训练更具泛化能力的通用机器人策略。

OpenVLA 将这两条路线进一步工程化：

```text
预训练 VLM
+
大规模机器人数据
+
Action Tokenization
+
开源训练与推理链路
↓
可下载、可推理、可微调、可部署的 VLA
```

因此，本章并不只是介绍一个模型，而是要建立一个完整的工程心智模型：

```text
Image + Language
↓
Visual Encoder
↓
Multimodal Backbone
↓
Action Tokens
↓
Detokenization
↓
Continuous Robot Action
↓
Robot Controller
```

并进一步理解：

```text
Pretrained OpenVLA
↓
新机器人 / 新任务 demonstration
↓
Normalization
↓
Fine-tuning
↓
Evaluation
↓
Deployment
```

完成本章后，应能够从“模型架构”“动作表示”“训练目标”“数据格式”“微调方式”“真实机器人部署”六个角度分析 OpenVLA，并理解它为什么适合作为第一个真正进入 VLA 代码与实验阶段的模型。

---

## 2. 问题从哪里来：为什么需要 OpenVLA

### 2.1 RT-2 已经证明 VLM 可以变成 VLA

RT-2 的核心思想可以抽象为：

```text
Web-scale Vision-Language Knowledge
+
Robot Demonstrations
↓
Co-finetuning
↓
能够理解语义并输出机器人动作的 VLA
```

它证明了 VLM 中的视觉语义知识可以迁移到机器人控制任务中。

例如，模型不仅可以识别“红色杯子”，还可以在语言指令：

```text
Pick up the red cup.
```

的条件下输出机器人控制动作。

但 RT-2 对普通研究者来说存在一个现实问题：它并不是一个很容易被普通实验室下载、修改、重新训练并迁移到自己机器人上的开放工程平台。

因此，一个自然的问题出现了：

> 是否可以把“VLM → VLA”这条路线做成一个真正开放、可复现、可微调的系统？

OpenVLA 正是围绕这一问题出现。

### 2.2 OpenVLA 的定位

OpenVLA 是一个约 7B 参数规模的开放 VLA。其核心结构继承了预训练 VLM 的范式，并通过机器人示教数据将模型训练为视觉—语言条件下的机器人策略。

其整体思想不是重新设计一套完全不同的机器人网络，而是：

```text
已有 VLM
↓
引入机器人动作表示
↓
使用机器人 demonstration 继续训练
↓
得到 VLA
```

因此，理解 OpenVLA 时最重要的不是把它看成一个孤立的新架构，而是把它放在如下演化链中：

```text
LLM
↓
VLM
↓
RT-2：VLM 可以输出 Action Token
↓
Open X-Embodiment：跨机器人数据可被统一训练
↓
OpenVLA：开放、可训练、可部署的 VLA
```

---

## 3. OpenVLA 的整体架构

OpenVLA 的核心数据流可以概括为：

```text
Camera Image
      │
      ↓
DINOv2 + SigLIP
      │
      ↓
Visual Features
      │
      ↓
Projector
      │
      ↓
Visual Tokens ─────┐
                   │
Language Tokens ───┼→ Llama 2 Backbone
                   │
                   ↓
             Action Tokens
                   │
                   ↓
          Action Detokenization
                   │
                   ↓
         Continuous Robot Action
```

从机器人策略角度，可以写成：

$$
a_t = \pi_\theta(I_t, l)
$$

其中：

- I_t 表示当前时刻的 RGB 图像；
- l 表示语言指令；
- a_t 表示当前机器人动作；
- θ 表示 OpenVLA 模型参数；
- π_θ 表示由 OpenVLA 实现的机器人策略。

这里需要特别注意：原始 OpenVLA 的核心输入是图像与语言，并不是所有实现都必须显式输入 proprioception。

因此，不能把 VLA 机械理解为：

```text
Image + Language + Robot State
```

VLA 是一类机器人策略模型，而不是一种固定输入模板。

---

## 4. 视觉编码：为什么同时使用 DINOv2 和 SigLIP

OpenVLA 的视觉输入并不是直接送入 Llama，而是先通过视觉编码器转换为视觉 representation。

其视觉部分融合了 DINOv2 与 SigLIP。

可以建立一个近似但非常实用的心智模型：

```text
SigLIP
≈
更强调图像与语言之间的语义对应

DINOv2
≈
更强调视觉结构、局部特征和空间 representation
```

例如，在一个桌面抓取场景中：

```text
桌面物体：
红色杯子
蓝色碗
香蕉
```

语言指令为：

```text
Pick up the red cup.
```

模型需要同时解决两个问题：

```text
语义问题：
哪个物体是 red cup？

空间问题：
这个物体在哪里？
机械臂应该朝哪个方向移动？
```

因此，融合两种视觉 representation 的意义在于同时增强：

- 视觉—语言语义对齐；
- 物体与场景结构表征；
- 与 manipulation 相关的空间信息。

当然，这种“语义 vs 空间”的划分只是帮助理解的近似，不意味着两个视觉编码器只负责单一能力。

### 阶段思考 A

**问题 1：OpenVLA 的主要输入是什么？是否必须显式输入 proprioception？**

### 解析

如果回答“图像、语言和机器人状态”，说明仍然把“VLA 的一般定义”与“OpenVLA 的具体实现”混在一起。

更准确的理解是：

```text
VLA 作为大类：
输入可以包含 image、language、state 等多种信息

OpenVLA 原始核心形式：
image + language → action
```

因此，阅读 VLA 论文时必须区分：

```text
模型类别允许什么
```

和

```text
具体实现实际用了什么
```

---

**问题 2：为什么 OpenVLA 同时使用 DINOv2 和 SigLIP？**

### 解析

只回答“为了提取更好的视觉特征”不够具体。

更完整的因果链是：

```text
机器人操作需要语义理解
+
机器人操作需要空间结构感知
↓
单一视觉表征未必同时最优
↓
融合不同预训练视觉特征
↓
为后续语言条件动作预测提供更丰富的表示
```

---

## 5. Language Backbone：Llama 2 为什么可以预测机器人动作

OpenVLA 的语言主干使用 Llama 2。

乍看之下，这会产生一个问题：

> Llama 本来用于预测语言 token，为什么能预测机器人动作？

关键在于 OpenVLA 并不要求 Llama 直接输出浮点控制量，而是先把连续动作变成离散 action token。

普通语言模型的核心训练形式可以概括为：

$$
P(x_{t+1} \mid x_{\leq t})
$$

其中：

- x_t 表示 token；
- 模型根据已有 token 序列预测下一个 token。

在 OpenVLA 中，只需要把机器人动作也转化为 token，就可以继续沿用这种 next-token prediction 机制。

于是：

```text
普通 LLM：
文本 token → 下一个文本 token

OpenVLA：
图像 + 文本 token → action token
```

这一步非常重要，因为它说明 OpenVLA 并不是完全抛弃 LLM 的原有能力，而是在原有 token prediction 框架中扩展了“机器人动作”这一新的输出语义。

---

## 6. Action Representation：从连续动作到 Action Token

### 6.1 机器人动作本来是连续值

假设一个 7 维 Cartesian delta action 为：

$$
a_t =
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

其中：

- Δx、Δy、Δz 表示末端位置增量；
- Δr_x、Δr_y、Δr_z 表示姿态增量；
- g 表示 gripper 状态。

例如某一时刻的动作可能是：

```text
[0.02, -0.01, 0.03, 0.00, 0.02, -0.01, 1.0]
```

这是一个连续向量。

但 Llama 的自然输出是 token，而不是连续浮点向量。

因此需要：

```text
Continuous Action
↓
Normalization
↓
Discretization
↓
Action Token
```

---

## 7. 256 bins：连续动作如何离散化

OpenVLA 会把每个 action 维度的连续值映射到离散区间。

为了便于理解，可以假设某一动作维度在归一化后落在：

```text
[-1, 1]
```

然后将这个区间划分为 256 个 bin：

```text
bin 0
bin 1
bin 2
...
bin 255
```

于是：

```text
-0.90 → 某个较小编号 bin
-0.20 → 中间偏左 bin
 0.00 → 中间附近 bin
 0.63 → 中间偏右 bin
```

形式上可以写成：

$$
a_i \in \mathbb{R}
\rightarrow
k_i \in \{0,1,\ldots,255\}
$$

其中：

- a_i 表示某个连续动作维度；
- k_i 表示离散后的 bin 编号。

在实际系统中，OpenVLA 会根据训练数据中的动作统计量确定离散化与归一化方式，而不是简单假定所有维度天然处于相同数值范围。

---

## 8. 一个容易产生的误区：7D Action 是否意味着 256⁷ 个动作类别

不是。

OpenVLA 并不是把整个 7D action 看成一个大类别。

错误理解是：

```text
完整 7D action
↓
从 256⁷ 个类别中选 1 个
```

正确理解是：

```text
第 1 维 action → 256 选 1
第 2 维 action → 256 选 1
第 3 维 action → 256 选 1
...
第 7 维 action → 256 选 1
```

因此，一个 7 维动作通常对应一个 action token 序列：

```text
<ACTION_141>
<ACTION_89>
<ACTION_229>
<ACTION_130>
<ACTION_141>
<ACTION_115>
<ACTION_255>
```

可以把联合概率写成：

$$
P(a_1,\ldots,a_7 \mid I,l)
=
\prod_{i=1}^{7}
P(a_i \mid I,l,a_{<i})
$$

其中：

- I 表示图像输入；
- l 表示语言指令；
- a_i 表示第 i 个动作维度对应的 token；
- a_<i 表示此前已经生成的 action token。

因此，OpenVLA 的动作生成仍然遵循自回归序列建模思想。

---

## 9. 为什么不同动作维度可以共用同一套 Action Token

另一个常见误区是：

```text
Δx 需要 256 个 token
Δy 再需要 256 个 token
Δz 再需要 256 个 token
...
```

实际上，不同动作维度可以共用同一套离散 token。

原因在于，token 本身表示“归一化标量值落在哪个 bin”，而动作维度的语义由 action sequence 中的位置决定。

例如：

```text
第 1 个 action token
→ 对应 Δx

第 2 个 action token
→ 对应 Δy

第 3 个 action token
→ 对应 Δz
```

因此：

```text
ACTION_150
```

出现在第一个位置时，可以理解为 Δx 的某个归一化值；

出现在第二个位置时，则对应 Δy 的相同离散归一化值。

这说明：

> token 表达数值区间，position 表达动作维度语义。

### 阶段思考 B

**问题 1：为什么不是 256⁷ 个动作类别？**

### 解析

因为模型并没有对整个 7D 向量进行联合分类，而是在序列中逐维预测离散 token。

正确心智模型是：

```text
7D Continuous Action
↓
7 个标量
↓
每个标量单独离散
↓
7 个 Action Token
```

---

**问题 2：为什么不同 action 维度可以共用一套 token？**

### 解析

只回答“因为数据格式统一”并不准确。

更核心的原因是：

```text
token
→ 表示归一化后的离散数值

sequence position
→ 表示这个数值属于哪一个 action dimension
```

因此，token 语义与位置共同决定最终动作含义。

---

## 10. 为什么 Action Tokenization 之前必须 Normalization

不同机器人、不同动作维度的数值尺度可能完全不同。

例如：

```text
Δx：
[-0.05, 0.05] m

Δθ：
[-0.3, 0.3] rad

gripper：
[0, 1]
```

如果直接把这些原始数值放入同一套 256 bins，就会出现尺度不一致问题。

因此需要：

```text
Raw Action
↓
Dataset Statistics
↓
Normalization
↓
256-bin Discretization
```

推理时再反向恢复：

```text
Action Token
↓
Bin Value
↓
Normalized Action
↓
Unnormalization
↓
Physical Robot Action
```

可以把这一过程写为：

$$
a
\rightarrow
\tilde a
\rightarrow
k
$$

推理时：

$$
k
\rightarrow
\tilde a
\rightarrow
a
$$

其中：

- a 表示物理空间中的真实动作；
- \tilde a 表示归一化动作；
- k 表示离散 action token。

Normalization 的作用不仅是“让数据看起来更整齐”，而是让不同 action dimension 可以在统一 tokenization 机制下具有可比较的尺度。

---

## 11. 为什么错误的 Normalization Statistics 会直接导致部署失败

假设模型预测：

```text
ACTION_180
```

经过 detokenization 后，可能对应：

```text
normalized action = 0.4
```

如果某机器人 Δx 的真实范围是：

```text
[-0.05, 0.05] m
```

则 0.4 需要被恢复为对应物理位移。

如果使用了错误机器人或错误数据集的 normalization statistics，就可能出现：

```text
模型原本想表达：
“向右移动一点”

控制系统最终执行：
“向右移动一个过大的距离”
```

这说明：

> action token 预测正确，并不代表机器人最终执行正确。

在真实机器人部署中，action statistics、坐标系、控制频率和 gripper 定义都属于必须核对的系统级要素。

---

## 12. OpenVLA 的训练目标

### 12.1 本质仍然是 Behavioral Cloning

假设专家数据为：

```text
(image, instruction, expert action)
```

专家 action 经过：

```text
normalize
↓
discretize
↓
action token sequence
```

之后，OpenVLA 的训练目标就是让模型预测正确 action token。

设正确 token 序列为：

$$
y = (y_1,y_2,\ldots,y_D)
$$

模型学习：

$$
P_\theta(y_i \mid I,l,y_{<i})
$$

训练损失为：

$$
L
=
-\sum_{i=1}^{D}
\log
P_\theta(y_i \mid I,l,y_{<i})
$$

其中：

- D 表示 action 维度；
- y_i 表示第 i 个动作维度对应的正确 action token；
- I 表示图像输入；
- l 表示语言指令；
- θ 表示模型参数。

这就是标准 Cross Entropy / Next-token Prediction。

因此，从机器人学习角度看：

```text
OpenVLA Training
=
Behavioral Cloning / SFT
+
Action Tokenization
```

---

## 13. OpenVLA 与传统连续动作 BC 的区别

传统连续动作 BC 常见形式是：

$$
L_{\mathrm{MSE}}
=
\lVert
\hat a - a^*
\rVert^2
$$

其中：

- a^* 为专家动作；
- \hat a 为模型预测动作。

而 OpenVLA 先将专家动作离散化：

```text
Continuous Action
↓
Action Token
```

再通过分类式的 token prediction 训练。

因此两者的区别主要在动作表示与输出头：

```text
传统 BC：
continuous regression

OpenVLA：
discrete token prediction
```

但在学习范式上，它们都属于 imitation learning / supervised policy learning。

---

## 14. Open X-Embodiment 在 OpenVLA 中的作用

OpenVLA 的训练并不是只依赖一台机器人或一个 manipulation task。

Open X-Embodiment 提供了大规模、异构、多机器人、多任务的真实机器人 demonstration 数据。

其意义在于：

```text
Robot A 数据
Robot B 数据
Robot C 数据
...
↓
统一 preprocessing
↓
dataset mixture
↓
大规模 VLA 训练
```

OpenVLA 使用了约 970,000 条真实机器人 trajectory / episode 规模的数据。

因此，Open X-Embodiment 在 OpenVLA 中不是一个抽象的数据集理念，而是构成大规模机器人预训练的重要数据基础。

---

## 15. 跨机器人数据为什么仍然困难

即使所有数据最终都要进入一个 VLA，不同机器人仍然可能存在：

- 不同 action space；
- 不同 action scale；
- 不同相机视角；
- 不同 control frequency；
- 不同 gripper 编码；
- 不同 workspace；
- 不同 observation format；
- 不同任务定义。

因此，不能简单做：

```text
所有 raw data
↓
直接拼接
```

而需要：

```text
Dataset A
↓
Dataset-specific Transform
┐
Dataset B
↓
Dataset-specific Transform
├→ Standardized Format
Dataset C
↓
Dataset-specific Transform
┘
↓
Normalization
↓
Action Tokenization
↓
Training
```

这也是为什么学习 VLA 时，dataset pipeline 与 action representation 的重要性不低于模型本身。

### 阶段思考 C

**问题 1：Open X-Embodiment 在 OpenVLA 中解决了什么？**

### 解析

只回答“提供宏观数据集思想”不够准确。

更完整的理解是：

```text
提供真实的大规模多机器人 demonstration
+
提供跨 embodiment 数据混合的基础
↓
支持 OpenVLA 做大规模机器人预训练
```

---

**问题 2：OpenVLA 的训练目标与普通 BC 是否本质不同？**

### 解析

不是。

两者都在学习专家 demonstration。

区别是：

```text
普通 BC
→ continuous regression

OpenVLA
→ action token classification / autoregressive generation
```

因此，OpenVLA 本质上仍然属于 BC / SFT。

---

## 16. OpenVLA 与 RT-2 的训练差异

RT-2 的一个关键设计是：

```text
Web Data
+
Robot Data
↓
Co-finetuning
```

其目的之一是让模型在学习机器人动作时仍然保留已有的大规模视觉—语言知识。

OpenVLA 的训练则更加突出：

```text
Pretrained VLM
↓
Large-scale Robot Data
↓
Robot Policy
```

因此可以把两者的关注点近似概括为：

```text
RT-2：
强调 Vision-Language Knowledge 向机器人控制的迁移与保留

OpenVLA：
强调开放、可复现、可微调的大规模机器人策略训练
```

这并不意味着 OpenVLA 不使用预训练视觉—语言知识，而是说明两者在训练数据组织与研究目标上存在差异。

---

## 17. Inference：OpenVLA 如何真正控制机器人

训练完成后，OpenVLA 的推理流程为：

```text
当前 RGB 图像 I_t
+
语言指令 l
↓
Visual Encoder
↓
Multimodal Backbone
↓
Action Token
↓
Detokenization
↓
Unnormalization
↓
Continuous Action a_t
↓
Robot Controller
↓
机器人运动
```

执行动作后，环境发生变化：

```text
I_t
↓
a_t
↓
Robot / Environment
↓
I_{t+1}
↓
a_{t+1}
```

因此 OpenVLA 本质上仍然是 closed-loop robot policy。

形式上：

$$
a_t = \pi_\theta(I_t,l)
$$

机器人执行后：

$$
I_t
\rightarrow
a_t
\rightarrow
I_{t+1}
\rightarrow
a_{t+1}
$$

这与“一次输入图像、一次性生成完整轨迹”是不同的心智模型。

---

## 18. OpenVLA 在机器人控制系统中的位置

OpenVLA 通常不是直接输出电机 PWM。

更合理的控制层级为：

```text
Language Instruction
↓
OpenVLA
↓
EEF Delta / Joint Target / Gripper Action
↓
Robot Control Interface
↓
IK / Low-level Controller
↓
Joint Command
↓
Actuator
```

因此，OpenVLA 位于：

```text
高层语义
↓
学习策略层
↓
动作目标
↓
传统控制层
↓
执行器
```

这一点对控制专业背景尤其重要。

VLA 并没有替代所有传统控制器，而是把“从视觉与语言到机器人动作目标”这一部分学习化。

---

## 19. 为什么预训练 OpenVLA 不能直接控制所有新机器人

假设新机器人也是 7DoF。

这并不意味着它与 OpenVLA 预训练时的数据接口完全兼容。

两个 7DoF 机器人仍然可能不同：

```text
action definition 不同
coordinate frame 不同
action range 不同
control frequency 不同
camera viewpoint 不同
workspace 不同
gripper encoding 不同
robot appearance 不同
task distribution 不同
```

因此：

```text
相同 Action Dimension
≠
相同 Action Semantics
```

这也是为什么新机器人部署往往需要 robot adaptation 或 task adaptation。

---

## 20. 新 Robot / Task 的完整适配流程

对于一个新的机器人平台，可以按以下流程理解：

```text
① Collect Demonstrations
        ↓
② Dataset Conversion
        ↓
③ Define Action Space
        ↓
④ Compute Normalization Statistics
        ↓
⑤ Observation Preprocessing
        ↓
⑥ Load Pretrained OpenVLA
        ↓
⑦ Fine-tuning
        ↓
⑧ Save Checkpoint
        ↓
⑨ Closed-loop Evaluation
        ↓
⑩ Robot Deployment
```

其中最容易被低估的是前半部分。

很多真实机器人实验失败，并不是因为 Transformer 架构本身，而是：

- action 坐标系不一致；
- gripper 定义反了；
- normalization statistics 用错；
- 相机 resize 或 crop 不一致；
- timestamp 不同步；
- 数据采集时控制频率与部署时不同；
- action scale 与真实控制接口不匹配。

因此，VLA 工程的核心能力不仅是“会训练模型”，还包括：

> 能够检查 observation、action、dataset、normalization 与 robot interface 是否真正一致。

---

## 21. Fine-tuning：新任务为什么不需要重新发明 Training Objective

假设预训练 OpenVLA 参数为：

$$
\theta_{\mathrm{OpenVLA}}
$$

新机器人数据集为：

$$
D_{\mathrm{new}}
$$

Fine-tuning 的目标仍然是：

$$
\theta'
=
\arg\min_\theta
L(D_{\mathrm{new}};\theta)
$$

训练数据仍然是：

```text
Image
+
Language
+
Expert Action
```

训练目标仍然是让模型预测正确 action token。

因此：

```text
Pretraining：
大规模多机器人数据 → Action Token

Fine-tuning：
你的机器人数据 → Action Token
```

本质训练目标没有发生改变。

改变的是：

- 数据分布；
- robot embodiment；
- task distribution；
- 可训练参数范围；
- 学习率与训练策略；
- 数据规模。

---

## 22. Full Fine-tuning

Full Fine-tuning 的核心是更新模型的大部分甚至全部参数。

可以粗略表示为：

```text
Visual Encoder
↓
Projector
↓
Language Backbone
↓
Action Prediction

全部参与梯度更新
```

其优势是：

```text
模型可调整空间大
↓
更有能力适配新的 robot / task distribution
```

但代价包括：

- GPU memory 更高；
- optimizer state 更大；
- gradient memory 更高；
- 训练时间更长；
- 对数据量要求更高；
- 更可能发生 catastrophic forgetting。

对于 7B 级模型，full fine-tuning 的工程成本并不低。

---

## 23. LoRA：低秩参数高效微调

LoRA 的核心思想是：

> 冻结原始大矩阵，只训练一个低秩更新量。

设原始线性层参数为：

$$
W
$$

Full Fine-tuning 直接更新 W。

LoRA 则写为：

$$
W' = W + \Delta W
$$

其中：

$$
\Delta W = BA
$$

设：

$$
A \in \mathbb{R}^{r \times d}
$$

$$
B \in \mathbb{R}^{d \times r}
$$

且：

```text
r ≪ d
```

因此：

```text
原始 W
↓
冻结

新增 A、B
↓
训练
```

训练参数量显著减少。

---

## 24. LoRA 为什么能显著降低训练成本

Full Fine-tuning 需要为大量模型参数保存：

- gradient；
- optimizer state；
- 参数更新信息。

LoRA 只对少量新增低秩参数进行优化，因此：

```text
Trainable Parameters
↓
Gradient Memory
↓
Optimizer Memory
↓
Checkpoint Size
↓
```

需要注意：

> LoRA 并不意味着基础模型完全不占显存。

基础模型本身仍然需要被加载用于 forward。

LoRA 的核心优势在于减少可训练参数与训练相关内存，而不是让 7B 模型“消失”。

---

## 25. Full Fine-tuning 与 LoRA 的真正区别

可以用一个非常直观的类比：

### Full Fine-tuning

```text
一本 1000 页教材
↓
允许修改几乎所有页面
```

### LoRA

```text
原教材基本冻结
↓
在关键位置增加少量可学习修订
```

因此：

```text
Full Fine-tuning
→ 调整空间更大

LoRA
→ 成本更低、参数更少、实验迭代更快
```

不能简单认为：

```text
LoRA 一定比 Full Fine-tuning 好
```

也不能反过来认为：

```text
Full Fine-tuning 一定效果更好
```

最终效果取决于：

- 新 robot 与预训练分布差异；
- demonstration 数量；
- task complexity；
- embodiment shift；
- learning rate；
- trainable modules；
- evaluation protocol。

### 阶段思考 D

**问题 1：Full Fine-tuning 和 LoRA 最核心的区别是什么？**

### 解析

最核心的区别不是“一个效果好、一个效果差”，而是：

```text
Full Fine-tuning
→ 更新大范围原始参数

LoRA
→ 冻结原始大权重，只训练低秩增量参数
```

---

**问题 2：为什么 LoRA 显著降低训练显存？**

### 解析

只说“因为训练参数少”方向正确，但还可以继续追问：

```text
训练参数减少
↓
需要保存的 gradient 减少
↓
optimizer state 减少
↓
训练显存下降
```

因此，这是一个完整的参数高效微调链条。

---

## 26. 一个完整 Pick-and-Place 机器人实例

下面用一个简单任务贯穿 OpenVLA 的完整流程。

### 26.1 任务

场景：

```text
桌面上有：
红色杯子
蓝色盒子
其他干扰物体
```

语言指令：

```text
Put the red cup into the blue box.
```

机器人使用 7 维 Cartesian delta action：

```text
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
```

### 26.2 数据采集

每条 demonstration 包含：

```text
Episode
├── image_t
├── instruction
├── action_t
├── timestamp
└── success / metadata
```

例如某一步：

```text
Image:
camera_001.jpg

Instruction:
"Put the red cup into the blue box."

Action:
[0.02, -0.01, 0.03, 0.00, 0.02, -0.01, 1.0]
```

### 26.3 Action preprocessing

原始动作：

```text
[0.02, -0.01, 0.03, 0.00, 0.02, -0.01, 1.0]
```

经过：

```text
Normalization
↓
Discretization
↓
Action Token Sequence
```

得到类似：

```text
<ACTION_137>
<ACTION_122>
<ACTION_150>
<ACTION_128>
<ACTION_140>
<ACTION_119>
<ACTION_255>
```

### 26.4 训练

模型输入：

```text
Image
+
Instruction
```

目标输出：

```text
Action Token Sequence
```

损失为 Cross Entropy。

### 26.5 推理

部署时：

```text
Current Image
+
Instruction
↓
OpenVLA
↓
Action Tokens
↓
Detokenization
↓
Unnormalization
↓
Continuous Action
↓
Robot Controller
```

机器人执行后重新获取图像：

```text
New Image
↓
Next Action
```

形成 closed loop。

### 26.6 如果换成另一台机器人

即使也是 7DoF，也必须重新核对：

```text
action definition
coordinate frame
normalization statistics
camera viewpoint
control frequency
gripper encoding
```

因此，cross-robot deployment 的关键不是“输出维度一样”，而是“动作语义与数据分布真正兼容”。

---

## 27. 与传统控制、Robot Learning 的联系

OpenVLA 可以放入如下系统层级：

```text
High-level Task Instruction
↓
Vision-Language Understanding
↓
OpenVLA Policy
↓
Action Representation
↓
Robot Control Interface
↓
Low-level Controller
↓
Actuator
```

与传统 Robot Learning 的关系为：

```text
Behavioral Cloning
↓
Expert Demonstration
↓
Policy Learning
↓
OpenVLA 使用 Action Token 的方式实现 BC / SFT
```

因此，OpenVLA 并不是“抛弃 BC 的全新范式”，而是把大模型和 BC 连接起来：

```text
BC
+
VLM Backbone
+
Action Tokenization
+
Large-scale Robot Data
=
OpenVLA
```

---

## 28. 与 Diffusion Policy 的关键差异

OpenVLA 与 Diffusion Policy 都可以用于机器人动作生成，但 action generation 机制明显不同。

### OpenVLA

```text
Image + Language
↓
Autoregressive Token Prediction
↓
Discrete Action Tokens
↓
Continuous Action
```

### Diffusion Policy

```text
Observation
+
Noise Action Trajectory
↓
Conditional Denoising
↓
Continuous Action Trajectory
```

核心区别是：

```text
OpenVLA
→ 离散 action token
→ 自回归生成

Diffusion Policy
→ 连续 action trajectory
→ 迭代去噪生成
```

这为后续 π0 的学习提供了重要参照。

---

## 29. Repository 阅读顺序：为什么不要先啃 Transformer

第一次打开 OpenVLA 代码仓库时，最容易犯的错误是：

```text
直接打开 model definition
↓
面对大量 Transformer layer
↓
逐行阅读
↓
不知道整体数据流
```

更高效的方式是先建立 repository map。

推荐顺序：

```text
① README / Quick Start
↓
② Inference Example
↓
③ Action Tokenizer
↓
④ Dataset Pipeline
↓
⑤ Prompt / Image Processing
↓
⑥ Model Forward
↓
⑦ Loss
↓
⑧ Training Script
↓
⑨ LoRA / Fine-tuning
↓
⑩ Evaluation
```

---

## 30. 为什么优先阅读 Inference Pipeline

Inference 最容易暴露整个系统的数据路径。

先找到类似：

```python
image = ...
instruction = ...

action = model.predict_action(
    image,
    instruction
)

robot.step(action)
```

然后沿调用链向下追：

```text
predict_action()
↓
processor
↓
model.forward()
↓
action token
↓
decode
↓
unnormalize
↓
continuous action
```

这样会先得到一个稳定的“系统地图”。

之后再阅读 model definition，就能够知道：

- 视觉编码器为什么存在；
- projector 在哪；
- action token 在哪里生成；
- output logits 如何转成动作；
- dataset 中 action 为什么要 normalization。

因此，阅读 VLA 代码时应优先理解：

```text
输入是什么
↓
怎么处理
↓
怎么进入模型
↓
怎么得到动作
↓
怎么执行
```

而不是一开始追求逐层理解全部 Transformer 细节。

---

## 31. 常见误区与阶段纠偏

### 31.1 “VLA 一定要输入 Robot State”

错误。

VLA 是模型类别，不代表所有实现具有固定输入形式。

OpenVLA 原始核心形式更接近：

```text
Image + Language → Action
```

---

### 31.2 “256 bins 表示整个 7D 动作只有 256 类”

错误。

正确理解是：

```text
每一个连续 action dimension
↓
单独离散成 256 bins
```

一个 7D action 通常得到 7 个 token。

---

### 31.3 “不同动作维度需要不同的 256 个 token 集合”

不一定。

OpenVLA 可以让不同维度共用一套离散 action token。

区别通过 action token 在序列中的位置体现。

---

### 31.4 “Normalization 只是为了数值稳定”

不完整。

它还承担了更关键的作用：

```text
不同动作维度 / 不同机器人
↓
尺度不一致
↓
Normalization
↓
统一离散化与 tokenization
```

---

### 31.5 “Action Token 预测正确，机器人就一定执行正确”

错误。

还需要：

```text
正确 detokenization
+
正确 normalization statistics
+
正确 action semantics
+
正确 robot control interface
```

否则 token 正确也可能执行错误。

---

### 31.6 “LoRA 就是更弱版 Full Fine-tuning”

不准确。

LoRA 的核心是 parameter-efficient adaptation。

其价值主要在：

- 降低 trainable parameters；
- 降低 optimizer memory；
- 减少 checkpoint；
- 加速实验迭代。

它不是简单的“效果缩水版 Full FT”。

---

### 31.7 “读 VLA 代码应该先读 Transformer”

通常效率很低。

更合理的是：

```text
Inference
↓
Action Representation
↓
Dataset
↓
Forward
↓
Training
```

因为工程数据流比内部 Transformer 细节更能决定你是否真正理解系统。

---

## 32. OpenVLA 在完整 VLA 知识链中的位置

可以把本章放回整套知识体系：

```text
Robot Policy
↓
Transformer
↓
Vision-Language Model
↓
Behavioral Cloning
↓
Robot Dataset
↓
Action Representation
↓
Action Tokenization
↓
RT-2
↓
Open X-Embodiment
↓
OpenVLA
↓
π0 / π0.5
↓
Fine-tuning / Agent / RL
```

OpenVLA 是一个重要转折点。

此前学习更多是在回答：

> VLA 为什么成立？

从 OpenVLA 开始，问题变成：

> 如何把一个 VLA 真正下载、读代码、微调、部署到新机器人？

因此，它是从“概念学习”向“工程与研究实践”过渡的重要章节。

---

## 33. 局限与研究视角

OpenVLA 已经解决了几个重要问题：

- 将开放 VLM 变成机器人策略；
- 使用大规模真实机器人数据训练；
- 统一 action tokenization；
- 提供 inference 与 fine-tuning 链路；
- 支持 LoRA 与 full fine-tuning；
- 为新机器人 adaptation 提供工程基础。

但它仍然存在值得研究的问题。

### 33.1 离散 Action Token 的表示限制

连续机器人动作被离散化后，不可避免地存在量化误差。

因此阅读论文时需要问：

```text
bin 数量是多少？
action range 如何决定？
不同维度是否统一离散？
高精度 manipulation 是否受到影响？
```

### 33.2 Autoregressive Action Generation

动作 token 是序列生成的。

这意味着：

```text
动作维度之间存在 autoregressive dependency
+
推理存在 token generation 开销
```

与一次生成 continuous action chunk 的方法相比，存在不同的延迟与控制特性。

### 33.3 Cross-Embodiment 不代表 Zero-shot Universal Control

大规模多机器人数据能够提高泛化能力，但不能推出：

```text
训练过很多机器人
↓
任意新机器人都无需适配
```

新机器人仍可能存在：

- embodiment shift；
- observation shift；
- action shift；
- controller mismatch。

### 33.4 Fine-tuning 的参数效率问题

一个重要研究问题是：

> 当只有少量 demonstration 时，到底应该更新哪些模块？

可能的方案包括：

```text
Full Fine-tuning
LoRA
Freeze Vision Encoder
Freeze Language Backbone
Train Adapter
Train Partial Layers
```

这些问题将在后续 VLA Fine-tuning 中系统展开。

---

## 34. 综合思考

### 问题 1

为什么 OpenVLA 可以继续使用 Llama 2，而不是必须设计一个全新的机器人 decoder？

### 解析

因为连续机器人动作被映射成 action token 后，机器人控制问题可以被重写为条件 next-token prediction。

因此：

```text
VLM 已有 token generation 能力
+
Action Tokenization
↓
可以复用语言模型的序列建模框架
```

---

### 问题 2

如果两个机器人都是 7DoF，为什么仍不能认为它们 action space 相同？

### 解析

维度一致只说明向量长度相同。

还必须比较：

- 每一维的物理语义；
- 坐标系；
- action scale；
- gripper 定义；
- control frequency；
- absolute / relative action。

因此：

```text
same dimension
≠
same control semantics
```

---

### 问题 3

为什么 OpenVLA 的 Fine-tuning 仍然可以看成 BC？

### 解析

因为模型仍然从 expert demonstration 中学习：

```text
observation
+
instruction
→
expert action
```

只不过 action 被离散成 token，loss 使用 Cross Entropy。

---

### 问题 4

为什么 Repository 阅读时应该先看 inference？

### 解析

Inference 能最快暴露：

```text
Input
↓
Preprocessing
↓
Model
↓
Action Representation
↓
Decode
↓
Robot Interface
```

这是整个系统最重要的数据流。

在没有这张图之前阅读 Transformer 内部实现，很容易陷入局部代码细节。

---

## 35. 本章总结

OpenVLA 最重要的不是某一个具体模块，而是它把现代 VLM 与机器人策略真正连接成了一套开放工程链路。

第一，OpenVLA 的核心结构仍然建立在 VLM 上：

```text
DINOv2 + SigLIP
↓
Visual Representation
↓
Projector
↓
Llama 2
```

第二，机器人连续动作通过：

```text
Normalization
↓
Discretization
↓
Action Tokenization
```

被转换成 LLM 可以预测的 token sequence。

第三，OpenVLA 的训练目标并没有脱离 Behavioral Cloning：

```text
Expert Demonstration
↓
Action Token
↓
Cross Entropy
```

因此，可以把它理解为：

```text
VLM Backbone
+
Robot BC
+
Action Tokenization
+
Large-scale Cross-Embodiment Data
```

第四，真实部署中最危险的问题往往不是“模型架构看没看懂”，而是：

```text
action semantics
normalization
coordinate frame
camera preprocessing
control frequency
robot interface
```

第五，适配新 robot / task 时，可以选择：

```text
Full Fine-tuning
```

或

```text
LoRA
```

两者的核心区别是可训练参数范围与训练成本，而不是简单的效果优劣。

最终应形成如下心智模型：

```text
Open X Robot Data
        ↓
    OpenVLA
        ↓
Image + Language
        ↓
Action Tokens
        ↓
Continuous Action
        ↓
Robot

新 Robot / Task
        ↓
Demonstrations
        ↓
Normalization
        ↓
Fine-tuning
        ↓
Evaluation
        ↓
Deployment
```

当能够从输入、视觉编码、action tokenization、training objective、dataset、normalization、fine-tuning 与 deployment 八个角度解释 OpenVLA 时，就已经具备继续进入 π0、π0.5 与 VLA Fine-tuning 的基础。

---

## 36. 关键词

**Vision-Language-Action Model（VLA）— 视觉语言动作模型 — 根据视觉、语言等观测生成机器人动作的策略模型。**

**OpenVLA — 开放视觉语言动作模型 — 面向机器人操作的开放 VLA 系统，可进行推理与微调。**

**DINOv2 — 自监督视觉编码器 — 提供丰富视觉结构与表征能力。**

**SigLIP — 视觉语言编码模型 — 强调图像与文本之间的语义对齐。**

**Visual Encoder — 视觉编码器 — 将 RGB 图像转换成视觉 feature / token。**

**Projector — 投影层 — 将视觉特征映射到语言模型可以处理的表示空间。**

**Language Backbone — 语言主干 — 负责多模态序列建模与 token 生成。**

**Action Token — 动作 token — 连续机器人动作离散化后的 token 表示。**

**Action Tokenization — 动作词元化 — 将连续 action 转换成离散 token sequence。**

**Discretization — 离散化 — 将连续数值划分到有限的 bin 中。**

**Normalization — 归一化 — 将不同尺度动作映射到统一数值范围。**

**Unnormalization — 反归一化 — 将归一化动作恢复到真实机器人控制尺度。**

**Behavioral Cloning（BC）— 行为克隆 — 从 expert demonstration 中监督学习机器人策略。**

**Cross Entropy — 交叉熵 — OpenVLA action-token prediction 的主要监督训练损失。**

**Open X-Embodiment — 多机器人数据集合 — 为大规模 cross-embodiment robot learning 提供训练数据。**

**Cross-Embodiment Learning — 跨本体学习 — 利用不同机器人数据共同训练策略。**

**Full Fine-tuning — 全量微调 — 更新模型大范围甚至全部参数。**

**LoRA — Low-Rank Adaptation — 通过训练低秩增量矩阵实现参数高效微调。**

**Closed-loop Policy — 闭环策略 — 执行动作后重新观测环境，再生成下一动作。**

**Robot Adaptation — 机器人适配 — 将预训练策略迁移到新的 embodiment 与控制接口。**

**Task Adaptation — 任务适配 — 将预训练策略迁移到新的 manipulation task。**

**Inference Pipeline — 推理链路 — 从 observation 到 action 再到 robot execution 的完整数据流。**

---

## 37. 本章毕业检查

完成本章后，应能够独立回答：

1. OpenVLA 为什么可以看成 VLM 向机器人 policy 的工程化延伸？
2. DINOv2 与 SigLIP 在视觉编码中为什么具有互补性？
3. 为什么 Llama 2 可以预测机器人动作？
4. continuous action 如何转换成 action token？
5. 256 bins 的真正含义是什么？
6. 为什么一个 7D action 通常对应 7 个 token，而不是 256⁷ 个类别？
7. 为什么 action dimension 可以共享一套 action token？
8. normalization 为什么是 VLA 部署中的关键环节？
9. OpenVLA 的训练为什么本质仍然属于 BC / SFT？
10. Open X-Embodiment 在 OpenVLA 训练中发挥什么作用？
11. RT-2 与 OpenVLA 的训练重点有什么不同？
12. OpenVLA inference 如何形成 closed loop？
13. 为什么相同 7DoF 不代表两个机器人可以直接共用 policy？
14. 新机器人 adaptation 的完整工程链路是什么？
15. Full Fine-tuning 与 LoRA 的核心区别是什么？
16. 为什么 LoRA 能显著降低训练成本？
17. 为什么阅读 OpenVLA repo 时应该先看 inference pipeline？
18. OpenVLA 在完整 VLA 知识链中处于什么位置？
