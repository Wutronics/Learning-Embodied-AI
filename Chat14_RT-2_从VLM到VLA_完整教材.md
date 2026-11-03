# 14｜RT-2：从 VLM 到 VLA——Web 知识如何真正转化为机器人行为

## 本章导言

RT-2 是 Vision-Language-Action（VLA）发展过程中非常关键的一步。

在更早的机器人学习系统中，模型通常依赖机器人数据学习“看见什么、听到什么指令、应该执行什么动作”。RT-1 已经证明，可以用统一的 Transformer 策略处理大量语言条件机器人任务，并形成一种 generalist robot policy。但它的知识边界仍然很明显：模型能够利用的知识，主要来自机器人数据本身。

另一方面，大规模 Vision-Language Model（VLM）已经通过互联网图文数据获得了丰富的视觉概念、语言知识、物体属性、语义关系和一定程度的推理能力。例如，一个 VLM 可能知道“香蕉富含钾”“石头很硬”“2 + 1 = 3”，但这些知识本身并不能让机械臂完成抓取、搬运和放置。

RT-2 研究的核心问题因此可以表述为：

> 能否把已经具备 Web-scale 视觉语言知识的 VLM，直接改造成一个能够输出机器人动作的策略，使互联网知识真正参与机器人控制？

这一问题的重要性不在于“把模型做得更大”，而在于第一次非常明确地打通了下面这条链路：

```text
Web-scale Vision-Language Knowledge
↓
Semantic Understanding
↓
Robot Action Token
↓
Physical Robot Behavior
```

本章将围绕五条主线展开：

```text
为什么 RT-1 还不够
↓
如何把 Robot Action 变成 Token
↓
Web Data 与 Robot Data 如何 Co-finetuning
↓
Semantic Generalization 从哪里来
↓
RT-2 仍然不能解决什么
```

学完本章后，应能够独立解释：

- RT-2 相比 RT-1 的本质进步是什么；
- VLM 为什么能够被改造成 VLA；
- Robot Action 为什么要被 Token 化；
- Co-finetuning 的作用是什么；
- Web Knowledge 如何真正影响 Robot Behavior；
- Semantic Generalization 与 Motor Skill Generalization 有什么区别；
- RT-2 为什么仍然受 Robot Data、BC、Latency 和 Action Representation 限制。

---

# 1. 问题从哪里来：为什么 RT-1 之后还需要 RT-2

## 1.1 RT-1 已经解决了什么

RT-1 的重要意义在于，它证明了一个统一的机器人策略可以同时处理大量不同任务。

其基本形式可以概括为：

```text
Camera Image
+
Language Instruction
↓
Transformer Robot Policy
↓
Discrete Robot Action
```

因此，RT-1 已经不是“一项任务训练一个模型”的传统路线，而是向 generalist robot policy 迈进。

例如，如果训练数据中包含：

```text
pick the apple
pick the banana
put the cup in the sink
move the can to the tray
```

模型可以学会根据当前视觉输入和语言指令选择动作。

但这类能力仍然主要建立在 Robot Data 之上。

---

## 1.2 RT-1 的真正瓶颈不是“看不见”，而是“知识来源受限”

一个容易产生的误区是：

> RT-1 不能理解图像，所以 RT-2 要引入 VLM。

这个说法不准确。

RT-1 本身就以图像为输入，也能从图像中学习物体、位置和场景特征。真正的问题是：

> RT-1 的视觉语义能力主要由机器人训练数据塑造，而机器人数据规模远小于互联网图文数据，因此它能利用的常识、语义关系和开放世界知识非常有限。

例如，机器人数据里可能有：

```text
pick banana
move banana to plate
```

但不一定有：

```text
pick the fruit rich in potassium
```

如果模型只依赖 Robot Data，它可能知道“banana”这个词对应哪个物体，却不一定知道：

```text
rich in potassium
→ banana
```

而一个经过 Web-scale 预训练的 VLM 很可能已经获得这种知识。

因此，RT-2 的研究动机不是单纯“让机器人看懂图像”，而是：

> 让机器人策略能够利用 VLM 通过互联网学习到的视觉语言知识，从而提升对新语义、新指令和新组合任务的泛化能力。

---

## 阶段思考 A：RT-2 相比 RT-1，最核心的研究动机是什么？

一个只回答“Robot Data 很贵、很少”的答案方向正确，但不完整。

Robot Data scarcity 确实是背景之一，但 RT-2 更核心的研究问题是：

```text
Robot Data
数量少、语义覆盖有限
↓
Robot Policy 的知识边界受限
↓
能否利用 VLM 已经学到的 Web-scale Knowledge
↓
让机器人在没有见过对应机器人指令的情况下仍能理解任务语义
```

因此更准确的答案是：

> RT-2 希望把 Web-scale VLM 中已有的视觉、语言和常识知识迁移到机器人控制中，让机器人不再只能依赖有限 Robot Data 学习“任务是什么意思”。

这个问题检查的是：是否区分“视觉输入能力”和“开放世界语义知识”。

---

# 2. 从 VLM 到 VLA：最关键的接口是 Action Tokenization

## 2.1 VLM 原本在做什么

典型 VLM 的抽象形式是：

```text
Image
+
Text Prompt
↓
Vision-Language Model
↓
Text Token Sequence
```

例如：

```text
Image: 一张香蕉图片
Prompt: What fruit is shown?
Output: banana
```

从模型内部来看，它做的事情仍然可以抽象为：

```text
Multimodal Input
↓
Transformer
↓
Next Token Prediction
```

问题在于：机器人需要的不是自然语言，而是物理动作。

---

## 2.2 机器人动作和语言 Token 看起来完全不同

机械臂动作通常是连续数值。

例如，一个末端执行器增量动作可以表示为：

$$
a_t
=
\left(
d,
\Delta x,
\Delta y,
\Delta z,
\Delta r_x,
\Delta r_y,
\Delta r_z,
g
\right)
$$

其中：

- d：终止或继续执行的离散标志；
- Δx、Δy、Δz：末端执行器位置变化；
- Δr_x、Δr_y、Δr_z：末端姿态变化；
- g：夹爪状态或夹爪开合程度。

这 8 个量本质上属于 Robot Action Space，而不是语言词表。

如果直接让 VLM 输出浮点数，就需要改变原有输出机制。

RT-2 的关键思想是：

> 不去改变“Transformer 预测 Token”的基本范式，而是把 Robot Action 编码成 Token。

---

## 2.3 连续动作如何被量化

对于一个连续动作维度 x，设它的允许范围为：

$$
x \in [x_{\min},x_{\max}]
$$

将这个区间均匀划分为 N 个离散 bin。

若 N = 256，则可以使用 0 到 255 的离散索引表示动作。

一种概念性的量化形式为：

$$
b
=
\operatorname{round}
\left(
\frac{x-x_{\min}}
{x_{\max}-x_{\min}}
(N-1)
\right)
$$

其中：

- x：原始连续动作值；
- x_min：该动作维度最小值；
- x_max：该动作维度最大值；
- N：离散 bin 数量；
- b：离散后的 bin index。

如果 N = 256，则：

```text
b ∈ {0, 1, 2, ..., 255}
```

这里的 0 到 255 不是米、弧度或速度，而只是编码。

执行前需要反向映射回物理动作空间。

---

## 2.4 一个简单数值例子

假设某个方向的位移范围为：

$$
\Delta x \in [-0.1,0.1]
$$

单位为 m。

现在动作值为：

$$
\Delta x = 0.05
$$

将其划分为 256 个 bins，可以得到近似离散索引：

$$
b_x
=
\operatorname{round}
\left(
\frac{0.05-(-0.1)}
{0.1-(-0.1)}
\times 255
\right)
$$

结果约为：

```text
b_x ≈ 191
```

因此：

```text
0.05 m
↓
quantization
↓
bin index 191
↓
Action Token
```

需要强调：这里的动作范围只是为了说明量化原理的示例，并不代表 RT-2 的实际控制范围。

---

## 阶段思考 B：256 个 bins 是否意味着机器人动作只能在 0～255 米之间变化？

不是。

这个错误的根源是把“编码空间”和“物理空间”混为一谈。

正确的数据流是：

```text
Physical Action
例如 0.05 m
↓
Normalization / Quantization
↓
Discrete Index
例如 191
↓
Token Prediction
↓
De-tokenization
↓
恢复为对应物理动作
```

因此：

```text
191
```

只是类别索引，不携带“米”这个单位。

这个问题检查的是：是否理解 action token 只是动作表示方式，而不是动作本身。

---

# 3. 为什么 Action Tokenization 这么重要

## 3.1 最大优势：复用 VLM 原来的生成机制

RT-2 不需要把 VLM 的 Decoder 完全改造成一个新的连续控制网络。

原本：

```text
Image + Text
↓
VLM
↓
Text Token
```

现在可以变成：

```text
Image + Text
↓
VLM
↓
Action Token
```

因此，模型仍然在做一个熟悉的问题：

> 在当前多模态条件下，预测下一个 Token。

Robot Action 只是 Token 的新语义。

---

## 3.2 Robot Action 训练仍然是监督学习

假设一个动作被编码成 8 个 Token：

$$
Y
=
(y_1,y_2,\dots,y_8)
$$

模型可以自回归地建模：

$$
p_\theta(Y\mid I,l)
=
\prod_{j=1}^{8}
p_\theta
\left(
y_j
\mid
I,l,y_{<j}
\right)
$$

其中：

- I：视觉输入；
- l：语言指令；
- y_j：第 j 个动作 Token；
- y_<j：前面已经生成的动作 Token；
- θ：模型参数。

训练时可以使用 Token-level Cross Entropy：

$$
\mathcal L_{\text{robot}}
=
-\sum_{j=1}^{8}
\log
p_\theta
\left(
y_j^*
\mid
I,l,y_{<j}^*
\right)
$$

其中 y_j* 来自专家 demonstration 中的真实动作。

所以，从 Robot Learning 角度看，这仍然属于：

```text
Expert Demonstration
↓
Supervised Learning
↓
Behavioral Cloning / SFT
```

不是 RL，也不是 Diffusion Policy。

---

## 阶段思考 C：RT-2 在 Robot Action Data 上训练，本质是 BC、RL 还是 Diffusion Policy？

核心答案：BC / Supervised Fine-tuning。

虽然网络非常大，输入包含图像和语言，输出形式也变成 Token，但训练信号仍然是：

```text
给定 observation 和 instruction
预测 expert action
```

这和 Behavioral Cloning 的本质完全一致。

这一点非常重要，因为它直接意味着：

> RT-2 并不会因为使用了 VLM，就自动摆脱 BC 的 distribution shift 与 compounding error。

后面讨论局限时，这一点会重新出现。

---

# 4. 只用 Robot Data 微调为什么还不够

## 4.1 预训练 VLM 已经拥有大量 Web Knowledge

假设预训练 VLM 已经通过互联网图文学习了：

```text
banana → fruit
banana → yellow
banana → potassium

rock → hard
rock → can be used as a tool

3 → number
heart → symbol
```

这些知识并不是机器人 demonstration 教出来的。

它们来自 Web-scale pretraining。

---

## 4.2 如果只拿 Robot Data Fine-tune

如果训练阶段只不断输入：

```text
pick apple
pick banana
move cup
open drawer
...
```

模型会越来越适应机器人动作任务。

但同时存在一个风险：

> 原本已经学到的广泛视觉语言知识可能被削弱。

从直觉上可以理解为：

```text
Pretrained VLM
↓
大量 Web Knowledge

只用 Robot Data Fine-tune
↓
Robot Behavior Ability ↑
Web Knowledge Retention ↓
```

这种现象与后续 Fine-tuning 中会遇到的 catastrophic forgetting 有联系。

因此，RT-2 不选择“只拿 Robot Data 把 VLM 改成机器人模型”，而是使用 Co-finetuning。

---

# 5. Co-finetuning：Web Data 与 Robot Data 一起训练

## 5.1 两类数据共同存在

RT-2 的训练数据可以抽象成：

$$
D
=
D_{\text{web}}
\cup
D_{\text{robot}}
$$

Web Data 仍然执行视觉语言任务，例如：

```text
Image: 香蕉
Question: What fruit is this?
Target: banana
```

Robot Data 则执行动作生成任务：

```text
Image: 机械臂前方有香蕉
Instruction: pick the banana
Target: Action Tokens
```

两者共用同一个 VLM Backbone。

---

## 5.2 同一个模型学习两种输出

Web task：

```text
Image + Text
↓
VLM
↓
Natural Language Token
```

Robot task：

```text
Image + Instruction
↓
VLM
↓
Action Token
```

关键点不是“先让一个 VLM 理解，再调用另一个 Robot Policy”，而是：

> 同一套模型参数同时承担视觉语言建模和机器人动作生成。

因此，可以用一个概念性损失理解 Co-finetuning：

$$
\mathcal L
=
\lambda_r
\mathcal L_{\text{robot}}
+
\lambda_w
\mathcal L_{\text{web}}
$$

其中：

- L_robot：机器人动作 Token 预测损失；
- L_web：视觉语言任务损失；
- λ_r：Robot Data 权重；
- λ_w：Web Data 权重。

这个公式用于帮助理解“联合训练”机制。实际训练还会通过数据混合和采样策略控制不同数据来源的影响。

---

## 5.3 为什么 Robot Data 要提高采样权重

现实中 Web Data 数量远大于 Robot Data。

如果简单按原始数据量混合：

```text
Web Data  ██████████████████████████████
Robot Data █
```

训练梯度会被 Web Data 主导。

结果可能是：

```text
VLM 能力维持得很好
但 Action Prediction 学得不充分
```

因此需要提高 Robot Data 的采样权重，让机器人数据虽然数量少，但在训练中拥有足够大的影响。

---

## 阶段思考 D：如果 Web Data 远多于 Robot Data，会发生什么？

只回答“Robot 本体能力会比较弱”方向正确，但需要更精确。

真正的问题是：

> Robot Action Learning 信号会被数量巨大的 Web 数据淹没，模型难以充分学习 Image + Instruction → Action Token 的映射。

因此解决方法不是简单“增加机器人数据数量”，还包括：

```text
Robot Data Reweighting
↓
提高其训练采样概率
↓
保证 Action Learning 有足够梯度贡献
```

这个问题检查的是：是否理解 Co-finetuning 不只是“把两堆数据放一起”，还要处理数据规模严重不平衡的问题。

---

# 6. Web Knowledge 如何真正变成 Robot Behavior

这是 RT-2 最值得形成清晰心智模型的部分。

## 6.1 一个典型例子

假设机器人从未在 demonstration 中看到以下任务：

```text
Pick up the fruit rich in potassium.
```

但模型已经从 Web Data 学到：

```text
banana ↔ potassium
```

同时 Robot Data 又让模型学到：

```text
banana image
→ pick banana action
```

于是模型有机会组合这两部分能力：

```text
"rich in potassium"
↓
semantic knowledge
↓
banana
↓
visual grounding
↓
banana position
↓
robot skill
↓
pick banana
```

可以抽象为：

$$
\text{Web Knowledge}
+
\text{Robot Skill}
\rightarrow
\text{New Robot Behavior}
$$

这里的“New Robot Behavior”并不意味着凭空创造了全新运动技能，而是：

> 用新的语义条件重新调用已经学会的机器人技能。

---

## 6.2 Web Data 与 Robot Data 的职责分工

可以用一句非常重要的话概括：

```text
Web Data
→ 教模型“做什么”

Robot Data
→ 教模型“身体怎么做”
```

更严格地说：

```text
Web Data
→ semantic understanding
→ object property
→ relation
→ concept
→ symbol
→ instruction interpretation
```

而 Robot Data：

```text
Robot Data
→ visuomotor mapping
→ end-effector movement
→ gripper control
→ pick / place execution
```

这两部分能力在同一个网络中结合，才形成 VLA。

---

## 阶段思考 E：Web Knowledge 是怎样“转化”为 Robot Behavior 的？

一个好的答案应该包含完整链路，而不只是说“模型理解了网络信息”。

正确心智模型是：

```text
Web Knowledge
↓
解析指令的语义含义
↓
确定任务目标 / 目标物体
↓
调用 Robot Data 学到的已有运动技能
↓
生成 Action Tokens
↓
物理执行
```

因此，Web Data 并不直接教机器人“如何移动关节”。

它主要改变的是：

> 在当前语义条件下，应该对哪个对象执行哪个已有技能。

---

# 7. Semantic Generalization：RT-2 真正增强的泛化是什么

## 7.1 视觉泛化与语义泛化不是一回事

传统视觉泛化可能是：

```text
训练：红苹果
测试：绿苹果
```

如果模型仍能识别并抓取，这属于视觉层面的泛化。

语言泛化可能是：

```text
训练：pick apple
测试：grab the apple
```

如果模型还能理解，属于语言表达层面的泛化。

而 RT-2 更强调：

> 测试指令中出现了训练 Robot Data 没有直接监督过的语义关系，但模型仍然能利用 VLM 的知识完成操作。

例如：

```text
Pick the fruit rich in potassium.
```

或者：

```text
Move the banana to the sum of two plus one.
```

这时模型可能需要先获得：

```text
2 + 1
↓
3
↓
识别数字 3 的位置
↓
把香蕉移过去
```

其内部逻辑更接近：

```text
Instruction
↓
Semantic Interpretation
↓
Target Grounding
↓
Robot Action
```

这就是 Semantic Generalization。

---

# 8. Semantic Generalization 不等于新 Motor Skill

这是本章最重要的纠偏之一。

## 8.1 一个非常诱人的错误理解

看到 RT-2 能利用 Web Knowledge 后，很容易产生如下推论：

```text
VLM 在网上见过很多任务
↓
机器人应该也会做这些任务
```

这是错误的。

---

## 8.2 知道“怎么描述”不等于身体“会执行”

假设 VLM 知道：

```text
tie shoelaces
```

是什么意思。

它甚至可能能用语言详细解释系鞋带步骤。

但如果 Robot Data 从未包含：

```text
柔性绳索操纵
双指精细配合
接触控制
高精度闭环动作
```

那么 RT-2 不会凭空获得这一运动技能。

因此：

$$
\text{Semantic Knowledge}
\neq
\text{Motor Skill}
$$

RT-2 更擅长的是：

$$
\text{New Semantic Condition}
+
\text{Existing Robot Skill}
\rightarrow
\text{New Skill Combination}
$$

而不是：

$$
\text{Web Description}
\rightarrow
\text{Completely New Physical Skill}
$$

---

## 阶段思考 F：为什么 RT-2 的 Semantic Generalization 不等于“机器人会做任何新动作”？

核心因果链是：

```text
Web Data
提供语义和知识
↓
告诉机器人“目标是什么”
↓
Robot Data
决定模型真正掌握哪些控制能力
↓
如果某个 motor skill 从未被机器人数据覆盖
↓
模型不会仅凭语言知识自动获得该技能
```

这个问题检查的是：是否把“认知泛化”和“运动技能泛化”区分开。

---

# 9. RT-2 的推理过程

## 9.1 从相机图像到真实动作

训练完成后，推理阶段不需要继续访问 Web Data。

典型闭环流程为：

```text
Camera Image
+
Language Instruction
↓
Vision-Language Backbone
↓
Action Token Sequence
↓
De-tokenization / De-quantization
↓
Continuous Robot Action
↓
Low-level Controller
↓
Robot
↓
New Observation
↓
再次推理
```

因此 RT-2 仍然是一个 Robot Policy，而不是一次性任务规划器。

---

## 9.2 为什么它是 Closed-loop

设当前 observation 为：

$$
o_t
$$

策略输出：

$$
a_t
=
\pi_\theta(o_t,l)
$$

机器人执行动作后进入新状态：

$$
o_{t+1}
$$

然后再次预测：

$$
a_{t+1}
=
\pi_\theta(o_{t+1},l)
$$

所以系统不是：

```text
观察一次
↓
生成整个动作脚本
↓
完全开环执行
```

而是持续：

```text
Observation
→ Action
→ Environment
→ New Observation
→ Action
```

这也是现代 VLA 仍然属于 policy 的原因。

---

# 10. 为什么 Action Generation 时要限制输出 Vocabulary

VLM 原来的词表可能包含：

```text
apple
banana
table
hello
the
123
...
```

机器人控制阶段则只允许：

```text
Action Tokens
```

如果模型在 Robot Action 模式下可以任意输出整个语言词表，就可能生成非法内容，例如自然语言单词，而不是格式正确的动作。

因此推理时需要约束输出空间。

可以理解成：

```text
Full Vocabulary
████████████████████████████

Robot Action Mode
↓ mask

Legal Action Tokens
████
```

这保证输出满足 Robot Action 格式。

---

## 阶段思考 G：为什么不能让模型在动作阶段自由输出语言 Token？

因为机器人控制接口要求的是严格结构化的动作编码，而不是自然语言。

因此：

```text
"move left"
```

和：

```text
Action Token 37
Action Token 192
...
```

在执行层意义完全不同。

这个问题检查的是：

> 是否理解 VLM 的语言输出空间与机器人控制接口之间必须有明确的表示约束。

---

# 11. 一个完整机器人实例：抓取“富含钾的水果”

下面用完整 Pick-and-Place 任务串联 RT-2。

---

## 11.1 场景

桌面上有：

```text
banana
apple
cup
```

语言指令为：

```text
Pick up the fruit rich in potassium.
```

Robot Data 中可能存在：

```text
pick banana
pick apple
pick cup
```

但没有：

```text
pick the fruit rich in potassium
```

---

## 11.2 第一步：语义理解

VLM Backbone 利用 Web-scale Knowledge 获得：

```text
rich in potassium
→ banana
```

这是 Web Data 贡献的能力。

---

## 11.3 第二步：视觉定位

视觉输入中识别：

```text
banana
```

并将语言概念和图像中的目标物体对应起来。

可以把这一过程理解为 semantic grounding。

---

## 11.4 第三步：机器人动作生成

Robot Data 已经让模型学会：

```text
visual observation
+
target banana
↓
end-effector delta
+
gripper action
```

动作被编码为 Token Sequence。

例如：

```text
[continue, 128, 191, 103, 121, 127, 129, close]
```

这里只用于说明结构，不代表实际 RT-2 输出值。

---

## 11.5 第四步：Action Decode

Token 被映射回实际机器人动作：

```text
Δx
Δy
Δz
Δr_x
Δr_y
Δr_z
gripper
```

然后交给底层控制接口。

---

## 11.6 第五步：闭环执行

机械臂执行一小步动作。

相机获得新图像。

RT-2 再次根据新 observation 预测下一动作。

因此任务真正的数据流是：

```text
Web Knowledge
↓
理解 instruction
↓
视觉目标 grounding
↓
Robot Skill
↓
Action Tokens
↓
Action Decode
↓
Controller
↓
Robot
↓
New Observation
↓
再次进入 RT-2
```

这个完整例子说明：

> Web Knowledge 主要解决“目标是谁、任务是什么意思”，Robot Data 主要解决“如何把机械臂移动过去并执行抓取”。

---

# 12. RT-2 与传统机器人控制系统的关系

RT-2 并不是直接输出电机 PWM。

它通常位于：

```text
Language / Vision
↓
RT-2 VLA
↓
Robot Action Representation
↓
Robot Controller
↓
Joint Command
↓
Motor / Actuator
```

因此从控制系统角度看：

- RT-2：高层 learned policy；
- Action Tokenization：策略输出表示；
- De-tokenization：恢复控制目标；
- Low-level Controller：执行轨迹跟踪和稳定控制；
- Actuator：实际驱动机器人。

这点非常重要，因为“大模型控制机器人”不代表大模型直接承担所有底层控制环。

---

# 13. RT-2 相比 RT-1 的本质进步

可以先写成最简单的两条链：

RT-1：

```text
Robot Data
↓
Generalist Robot Policy
```

RT-2：

```text
Web Data
+
Robot Data
↓
VLM-based VLA
```

RT-2 的真正突破不是“动作格式变成 Token”本身，而是：

> 把 Foundation VLM 的 Web-scale Knowledge 真正接入了 Robot Action Generation。

因此 RT-2 不只是“会更多机器人任务”，而是开始展示：

```text
Internet Knowledge
+
Existing Motor Skill
↓
Previously Unseen Semantic Task
```

这正是 VLA 概念形成过程中的关键一步。

---

# 14. RT-1 与 RT-2 对比

| 维度 | RT-1 | RT-2 |
|---|---|---|
| 基本定位 | Generalist Robot Policy | Vision-Language-Action Model |
| 核心知识来源 | Robot Data | Web Data + Robot Data |
| Backbone | 面向机器人任务设计的 Transformer | 预训练 VLM Backbone |
| 输入 | Image + Language | Image + Language |
| 输出 | 离散 Robot Action | Action Tokens |
| 动作学习方式 | BC | BC / SFT |
| Web-scale Knowledge | 基本不直接利用 | 核心能力来源 |
| Co-finetuning | 无 | 有 |
| Semantic Generalization | 较弱 | 明显增强 |
| 新 Motor Skill | 不具备 | 仍不具备 |
| Cross-Embodiment | 未解决 | 仍未解决 |
| Action Representation | 离散化 | Tokenized Discrete Action |
| 主要价值 | 多任务统一机器人策略 | 把 VLM 知识迁移到机器人行为 |

最重要的演化逻辑是：

```text
RT-1
Robot Data Scale
↓
Generalist Robot Policy

RT-2
Web Knowledge + Robot Data
↓
Semantic Generalization in Robot Control
```

---

# 15. RT-2 的主要局限

## 15.1 Motor Skill 仍受 Robot Data 限制

RT-2 可以理解新的任务语义，但物理动作能力仍主要来自 demonstration。

所以：

```text
知道“该做什么”
≠
身体真的“会怎么做”
```

如果 Robot Data 中没有某种高度灵巧或复杂的 motor skill，Web Knowledge 并不会自动补全这一能力。

---

## 15.2 Robot Data Scarcity 仍然存在

RT-2 没有消除真实机器人数据昂贵的问题。

它解决的是：

> 让有限 Robot Data 能够借助大规模 Web Knowledge 获得更高的泛化价值。

但是：

```text
Image
→ Physical Action
```

这条映射仍然需要 Robot Data。

---

## 15.3 Action Tokenization 有量化误差

连续动作被离散到有限 bins：

$$
a
\rightarrow
b
$$

必然会产生精度损失。

例如两个非常接近的连续值可能落入同一个 bin。

因此对：

```text
精细插孔
精密装配
高速运动
接触丰富操作
```

离散 Token 可能不如连续生成自然。

这也是后续 VLA 逐渐发展出 continuous action generation、Diffusion 和 Flow Matching 路线的重要背景之一。

---

## 15.4 大模型与实时控制存在天然矛盾

更大的 VLM 往往带来更强语义能力，但也意味着：

```text
Model Size ↑
↓
Inference Cost ↑
↓
Policy Frequency ↓
```

而机器人控制希望：

```text
Feedback Frequency ↑
Latency ↓
```

因此形成矛盾：

```text
Large VLA
vs
Real-Time Robot Control
```

这也是后续 VLA Inference、Asynchronous Execution、Latency Compensation 等研究问题的重要来源。

---

## 15.5 Cross-Embodiment 仍未解决

RT-2 的动作空间仍绑定具体机器人控制接口。

不同机器人可能使用：

```text
Joint Position
Joint Velocity
EEF Delta
Torque
Dual-arm Action
Mobile Base Action
```

因此：

```text
RT-2
≠
One Model for Every Robot
```

这会自然引向后续 Cross-Embodiment Learning。

---

## 15.6 它仍然继承 BC 的问题

因为 RT-2 的机器人动作学习本质仍然是 BC，所以仍然存在：

```text
Distribution Shift
Compounding Error
OOD Observation
Failure Recovery Weakness
```

如果策略一次预测错误，机器人可能进入训练数据很少出现的状态，之后的预测继续恶化。

因此：

> VLM 很强并不意味着 BC 的经典问题消失。

这也是后续 VLA + RL、Agent Recovery、Online Adaptation 等方向存在的原因。

---

# 16. 容易混淆的概念

## 16.1 “RT-1 不会看图” vs “RT-1 缺乏 Web-scale Semantic Knowledge”

前者错误，后者准确。

RT-1 本来就使用视觉输入。

真正差异是知识来源：

```text
RT-1
主要依赖 Robot Data

RT-2
利用 Web-scale VLM Knowledge
```

---

## 16.2 Action Token vs Physical Action

Action Token 是编码。

Physical Action 是真实控制量。

二者关系：

```text
Physical Action
↓
Quantization
↓
Action Token
↓
Prediction
↓
De-quantization
↓
Physical Action
```

因此 Token 191 不等于 191 m，也不等于 191 rad。

---

## 16.3 Semantic Generalization vs Motor Skill Generalization

Semantic Generalization：

```text
新的任务描述
新的语义组合
新的目标关系
↓
模型仍能判断应该做什么
```

Motor Skill Generalization：

```text
模型会执行训练中从未学过的全新物理技能
```

RT-2 显著增强的是前者，而不是后者。

---

## 16.4 Web Knowledge vs Robot Skill

Web Knowledge：

```text
what / which / meaning / relation
```

Robot Skill：

```text
how to physically act
```

两者必须结合，才能形成完整 VLA 行为。

---

## 16.5 Co-finetuning vs 只用 Robot Data Fine-tuning

只用 Robot Data：

```text
VLM
↓
Robot Data
↓
Robot Policy
```

风险是削弱原有 Web Knowledge。

Co-finetuning：

```text
Web Data
+
Robot Data
↓
Joint Fine-tuning
↓
保持知识 + 学习动作
```

---

# 17. 本章内容在 VLA 知识链中的位置

本章前面的知识已经提供了：

```text
Transformer
↓
VLM
↓
Robot Learning / BC
↓
Action Representation
↓
Action Tokenization
```

RT-2 将这些模块第一次非常清晰地组合成：

```text
VLM Backbone
+
Robot Action Tokens
+
Robot Demonstrations
+
Web Co-finetuning
↓
Vision-Language-Action Model
```

而本章留下的问题会自然引向后续方向：

```text
Robot Data 稀缺
↓
Cross-Embodiment Learning

Action Token 离散化
↓
Continuous Action Generation

推理慢
↓
Real-Time VLA / Latency

BC 上限
↓
VLA + RL

Motor Skill 受限
↓
更强 Robot Pretraining / Cross-Embodiment / Post-training
```

---

# 18. 研究视角：阅读 VLA 论文时应该继续追问什么

RT-2 建立了一个很重要的分析框架。以后阅读新的 VLA 论文时，可以沿着以下问题判断它与 RT-2 的差异。

首先看输入：

```text
Visual Input 是什么？
Language 如何输入？
Robot State 是否加入？
```

然后看输出：

```text
Discrete Token？
Continuous Regression？
Action Chunk？
Diffusion？
Flow Matching？
```

再看训练：

```text
只有 Robot Data？
Web + Robot Co-training？
Cross-Embodiment？
SFT？
RL Post-training？
```

最后看真正的泛化：

```text
只是换背景？
换物体？
换指令？
语义推理？
换任务？
换机器人？
还是出现新的 Motor Skill？
```

“Generalization”这个词在 VLA 论文里非常宽泛，必须判断它到底指哪一类泛化。

---

# 19. 综合思考

## 思考 1

为什么说 RT-2 不是简单地“给 RT-1 加一个 VLM”？

### 解析

因为它不是两级串联系统：

```text
VLM
↓
独立 Robot Policy
```

而是让同一个 VLM Backbone 直接产生 Action Tokens。

因此：

```text
Vision-Language Modeling
和
Robot Action Generation
```

被统一进了同一个 Token Prediction Framework。

---

## 思考 2

为什么 Action Tokenization 是 RT-2 能够复用 VLM 的关键接口？

### 解析

因为 VLM 的原生能力是：

```text
Multimodal Input
→ Token Sequence
```

如果 Robot Action 也被编码为 Token，则不需要从根本上改变生成范式。

因此：

```text
Text Token
和
Action Token
```

在模型层面都成为离散输出符号。

---

## 思考 3

如果 Web Data 很强，为什么还必须要 Robot Data？

### 解析

Web Data 能提供：

```text
concept
semantics
relation
reasoning
visual understanding
```

但它不直接提供真实机器人的：

```text
actuation
kinematics-aware behavior
gripper timing
contact interaction
robot-specific control distribution
```

因此 Web Knowledge 不能替代 Robot Demonstration。

---

## 思考 4

为什么 RT-2 并没有解决“机器人什么都会”？

### 解析

因为：

```text
Semantic Knowledge
和
Motor Skill
```

是两类不同能力。

RT-2 可以用新的语义方式调用已有 skill，但不会因为互联网上存在一段“如何系鞋带”的文字，就自动获得精确绳索操作能力。

---

## 思考 5

RT-2 仍然是 BC，为什么这一点值得特别强调？

### 解析

因为这意味着它继承 BC 的核心问题：

```text
Training Distribution
≠
Policy Rollout Distribution
```

一旦执行误差把机器人带到训练分布之外，VLM 再强也可能失效。

因此“大模型能力”并不能替代 Closed-loop Distribution Robustness。

---

## 思考 6

如果用一句话总结 RT-2 的历史贡献，应怎样表述？

### 解析

一个准确的版本是：

> RT-2 证明了可以通过 Action Tokenization 和 Web/Robot Co-finetuning，把预训练 VLM 的 Web-scale 视觉语言知识真正迁移到机器人动作生成中，从而显著增强机器人策略的 Semantic Generalization。

---

# 20. 本章总结

RT-2 的核心不是某个复杂的新控制算法，而是重新定义了“机器人策略可以从哪里获得知识”。

RT-1 主要依赖：

```text
Robot Data
→ Robot Policy
```

RT-2 则建立：

```text
Web Knowledge
+
Robot Demonstration
→ VLA
```

要理解 RT-2，需要抓住以下六个结论。

第一，RT-2 的 Backbone 是预训练 VLM，真正新增的关键能力不是“看图”，而是利用 Web-scale 视觉语言知识。

第二，Robot Action 被离散化并编码为 Action Tokens，从而让原本用于生成文本 Token 的 VLM 可以直接生成机器人动作。

第三，Robot Action 学习在本质上仍然属于 Behavioral Cloning / Supervised Fine-tuning，而不是 RL 或 Diffusion。

第四，Co-finetuning 同时保留 Web Data 和 Robot Data，使模型既维持语义知识，又学习 Physical Action。

第五，RT-2 最重要的泛化是 Semantic Generalization：

```text
新的语义条件
+
已有运动技能
→
新的机器人行为组合
```

而不是凭空创造新的 Motor Skill。

第六，RT-2 仍然存在明显边界：

```text
Robot Data Scarcity
Action Quantization Error
Latency
Single Embodiment
BC Distribution Shift
Motor Skill Coverage
```

因此，RT-2 不是 VLA 的终点，而是一个非常重要的转折点。

最终应形成的心智模型是：

```text
Pretrained VLM
+
Action Tokenization
+
Robot Demonstrations
+
Web / Robot Co-finetuning
↓
Semantic Knowledge Transfer
↓
Vision-Language-Action Policy
```

更进一步压缩：

$$
\text{RT-2}
=
\text{VLM}
+
\text{Action Tokenization}
+
\text{Web/Robot Co-finetuning}
$$

其最终意义不是让机器人“凭空学会一切动作”，而是让机器人第一次能够更系统地把互联网世界中学到的知识用于物理世界的行动。

---

# 21. 关键词

**Vision-Language-Action Model（VLA）— 视觉语言动作模型 — 将视觉、语言与机器人动作生成统一起来的策略模型。**

**RT-2 — VLM-to-VLA 代表模型 — 将预训练 VLM 通过 Action Tokenization 与 Robot Data 适配为机器人策略。**

**Action Tokenization — 动作 Token 化 — 将连续机器人动作离散编码为可由语言模型预测的 Token。**

**Action Bin — 动作离散区间 — 将连续动作范围划分为若干区间后得到的离散索引。**

**Behavioral Cloning（BC）— 行为克隆 — 使用专家 demonstration 监督训练策略预测动作。**

**Supervised Fine-tuning（SFT）— 监督微调 — 用输入与目标输出对继续训练预训练模型。**

**Co-finetuning — 联合微调 — 同时使用 Web Data 与 Robot Data 训练模型。**

**Web Knowledge — 互联网知识 — VLM 从大规模图文数据中学到的视觉、语言、属性和语义关系。**

**Robot Data — 机器人数据 — 包含 observation、instruction 与 expert action 的真实或仿真 demonstration。**

**Semantic Generalization — 语义泛化 — 在新的语言语义、知识组合或目标关系下仍能正确决定机器人行为。**

**Motor Skill — 运动技能 — 机器人在物理空间中真正执行某类操作的控制能力。**

**Knowledge Transfer — 知识迁移 — 将 Web-scale 视觉语言知识用于机器人决策和动作选择。**

**Vocabulary Constraint — 词表约束 — 在动作生成阶段限制模型只输出合法 Action Tokens。**

**De-tokenization — Token 解码 — 将 Action Token 恢复为机器人可以执行的物理动作。**

**Quantization Error — 量化误差 — 连续动作离散化为有限 bins 后产生的精度损失。**

**Closed-loop Policy — 闭环策略 — 每次执行动作后重新获取 observation 并再次推理。**

**Distribution Shift — 分布偏移 — 策略部署时访问到训练数据分布之外的状态。**

**Cross-Embodiment — 跨本体 — 在不同机器人形态、传感器和动作空间之间共享或迁移策略能力。**
