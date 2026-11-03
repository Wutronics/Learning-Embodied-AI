# 03｜ViT、CLIP 与 Vision-Language Model

## 1. 本章导言

视觉—语言—动作模型（Vision-Language-Action Model, VLA）的核心任务，是把机器人看到的视觉信息、接收到的自然语言指令以及机器人自身状态共同转化为可执行动作。要理解这个过程，首先必须解决一个更基础的问题：**图像怎样进入 Transformer，视觉信息怎样与语言建立语义联系，又怎样进一步成为大语言模型可以联合理解和推理的多模态表示。**

本章围绕三条连续的技术主线展开。

第一条主线是视觉 Transformer（Vision Transformer, ViT）。它解决“二维图像怎样变成 Transformer 能处理的 token 序列”。

第二条主线是 CLIP / SigLIP 一类视觉—语言预训练方法。它们解决“视觉表示和语言表示怎样进入可比较的语义空间，从而建立 multimodal alignment”。

第三条主线是视觉—语言模型（Vision-Language Model, VLM）。它进一步通过 visual encoder、projector 和 LLM backbone，把图像和语言真正组合成可用于理解、推理和生成的统一模型。

这一章位于 Transformer 基础之后、Robot Learning 与 Behavioral Cloning 之前。Transformer 已经解决了“token 序列怎样进行上下文建模”的问题；本章进一步解决“图像怎样变成 token，以及视觉 token 怎样进入语言模型”的问题。后续 Robot Learning 才会继续回答“这些多模态表示怎样与机器人状态一起映射为 action”。

因此，本章真正需要形成的不是几个模型名称的记忆，而是一条连续的数据流：

```text
RGB Image
↓
Patch
↓
Visual Token
↓
Vision Encoder
↓
Vision-Language Alignment
↓
Projector
↓
LLM-compatible Visual Tokens
+
Language Tokens
↓
VLM
↓
后续 Robot Learning / VLA
```

完成本章后，应能够在阅读 VLM/VLA 论文时判断：

- 图像如何被切分并编码成 visual token；
- token 数量和 feature dimension 分别意味着什么；
- ViT、CLIP、SigLIP、DINO 各自在解决什么问题；
- visual encoder、projector、LLM backbone 如何连接；
- multimodal alignment 与 instruction tuning 分别承担什么任务；
- 为什么 semantic alignment 不等于 precise spatial grounding；
- 为什么一个会看图问答的 VLM 仍然不是 VLA。

---

## 2. 问题从哪里来：Transformer 为什么不能直接处理一张图片

Transformer 的输入可以抽象成一串 token embedding：

$$
X=[x_1,x_2,\ldots,x_N]
$$

其中每个 token 表示为：

$$
x_i\in\mathbb{R}^{D}
$$

这里：

- N 表示 token 数量；
- D 表示每个 token 的 embedding dimension；
- xᵢ 表示第 i 个 token 的向量表示。

自然语言很容易被表示成这种结构。例如：

```text
"pick up the red cup"
↓
[pick] [up] [the] [red] [cup]
↓
language token embeddings
```

但一张 RGB 图像并不是 token 序列。它通常表示为：

$$
I\in\mathbb{R}^{H\times W\times C}
$$

其中：

- H 是图像高度；
- W 是图像宽度；
- C 是颜色通道数。

例如一张 224 × 224 的 RGB 图像：

$$
I\in\mathbb{R}^{224\times224\times3}
$$

它本质上仍然是一个二维空间上的像素矩阵，而 Transformer 需要的是一串向量。因此第一个核心问题是：

> 如何把二维图像改造成 Transformer 可以处理的一维 token sequence？

ViT 的核心思想非常直接：

> 把图片切成许多固定大小的 patch，再把每个 patch 转换成一个向量 token。

---

## 3. 从图像到 Visual Token：ViT 的基本机制

### 3.1 Patchify：把二维图像切成局部块

设输入图像大小为：

$$
H\times W\times C
$$

patch 边长为 P。如果 H 和 W 都能被 P 整除，则 patch 数量为：

$$
N=
\frac{H}{P}
\cdot
\frac{W}{P}
$$

例如：

```text
H = 224
W = 224
P = 16
C = 3
```

那么横向和纵向分别有：

$$
224/16=14
$$

个 patch，因此总数为：

$$
N=14\times14=196
$$

完整过程是：

```text
RGB Image
[224, 224, 3]

        ↓ Patchify

14 × 14 patches

        ↓

196 patches
```

每个 patch 的空间尺寸为：

$$
16\times16\times3
$$

此时 patch 仍然只是原始像素块，还不是 Transformer 意义上的 token。

---

### 3.2 Flatten：把 Patch 展平成向量

一个 16 × 16 × 3 patch 一共有：

$$
16\times16\times3=768
$$

个数。

将其展平后：

$$
p_i\in\mathbb{R}^{768}
$$

一般情况下，一个 P × P × C patch 展平后：

$$
p_i\in\mathbb{R}^{P^2C}
$$

这里的 pᵢ 只是“一个 patch 的像素数据被写成向量”的结果。它还没有经过可学习的视觉表示变换。

---

### 3.3 Patch Embedding：从 Patch 到真正的 Visual Token

ViT 使用一个可学习的线性映射，把每个展平后的 patch 投影到 Transformer 的 embedding space：

$$
x_i=p_iW_E+b_E
$$

其中：

- pᵢ 是第 i 个 patch 展平后的输入向量；
- W_E 是可学习的 patch embedding 权重矩阵；
- b_E 是偏置；
- xᵢ 是第 i 个 visual token；
- D 是 visual token 的 embedding dimension。

如果 patch 原始维度是 768，而模型设定 token dimension 为 512，那么：

$$
\mathbb{R}^{768}
\rightarrow
\mathbb{R}^{512}
$$

对整张图片而言，196 个 patch 最终变成：

$$
X_{\text{vision}}
\in
\mathbb{R}^{196\times512}
$$

如果考虑 batch size B：

$$
X_{\text{vision}}
\in
\mathbb{R}^{B\times196\times512}
$$

这就是深度学习代码中非常常见的：

```text
[B, N, D]
```

其中：

- B：batch size；
- N：token 数量；
- D：每个 token 的 feature dimension。

---

### 思考节点 A｜Patch、Visual Token 与 Shape

#### 问题 1

一张 224 × 224 × 3 图片，patch size 为 16 × 16，一共会产生多少个 patch？

#### 问题 2

每个 16 × 16 × 3 patch 展平之后是多少维？

#### 问题 3

Patch 和 visual token 是否是完全相同的概念？

#### 问题 4

如果一张图片有 196 个 patch，patch embedding dimension 为 512，那么进入 Transformer 前的 token matrix shape 是多少？

### 解析与纠偏

第 1 题的计算为：

$$
\frac{224}{16}
\times
\frac{224}{16}
=
14\times14
=
196
$$

第 2 题为：

$$
16\times16\times3=768
$$

第 3 题是本节第一个重要误区。

Patch 是原始图像中的局部像素区域，而 visual token 是该 patch 经过 flatten 和 learnable linear projection 后得到的向量表示。

正确的数据流是：

```text
Patch
↓
Flatten
↓
Linear Projection
↓
Visual Token
```

因此，不能把 patch embedding 描述为“把 patch 非线性化”。最基本的 ViT patch embedding 本质上是一个线性映射。

第 4 题不能只回答“512”。512 只是**每个 token 的 feature dimension**。整张图片有 196 个 token，因此矩阵 shape 是：

$$
196\times512
$$

这一点在后续阅读 VLM/VLA 代码时极其重要。看到：

```text
[B, N, D]
```

应该形成条件反射：

```text
B = batch size
N = token 数量
D = 每个 token 的 feature dimension
```

---

## 4. Position Embedding：图像的空间位置不能丢失

仅把图像切成 196 个 token 还不够，因为 Transformer 的 self-attention 本身并不知道：

- 哪个 token 原本位于左上角；
- 哪个 token 位于右下角；
- 哪些 token 在空间上相邻。

因此需要加入位置表示：

$$
z_i=x_i+e_i^{\text{pos}}
$$

其中：

- xᵢ 是 patch embedding；
- eᵢᵖᵒˢ 是第 i 个 patch 对应的位置 embedding；
- zᵢ 是真正送入 Transformer 的带位置信息 token。

这一步对于机器人视觉尤其关键，因为：

```text
“图中有一个红杯子”
```

和：

```text
“红杯子位于夹爪左前方”
```

不是同一个层次的信息。

前者偏语义存在性，后者涉及空间关系。

---

### 思考节点 B｜为什么视觉 Token 需要位置编码

#### 问题

既然每个 patch 已经变成 visual token，为什么还需要 position embedding？

### 解析

仅靠 token 内容，Transformer 并不知道它原本来自图像什么位置。

对于语言，词序会影响含义；对于图像，二维空间布局同样重要。

因此 position embedding 的作用不是“告诉模型这个 patch 是什么”，而是补充：

> 这个视觉特征原本位于图像中的哪里。

对于机器人 manipulation，这直接关系到物体位置、夹爪位置以及物体之间的空间关系。

---

## 5. Self-Attention 在图像中到底“注意”什么

visual token 进入 Transformer 后，attention 机制与语言 Transformer 本质相同。

输入 token matrix X 先得到：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

然后计算：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

这里：

- Q 表示 query；
- K 表示 key；
- V 表示 value；
- QKᵀ 衡量不同 token 之间的匹配关系；
- softmax 将匹配分数转成 attention weight；
- V 提供真正被聚合的信息。

在 ViT 中，attention 的对象不是“一个原始 patch 图片和另一个原始 patch 图片”，而是：

> 不同 patch 对应的 visual token 之间的 learned relation。

例如机器人场景中：

```text
cup token
↕
gripper token
```

```text
cup token
↕
table token
```

模型可以通过 attention 建立这些原本在图像中相距较远区域之间的上下文关系。

---

### 5.1 Visual Token 并不天然等于一个物体

一个 patch 覆盖杯子的一部分，并不意味着对应 visual token 一开始就表示“cup”。

浅层 token 可能主要编码：

- 颜色；
- 边缘；
- 纹理；
- 局部形状。

随着多层 Transformer 建模：

$$
x_i^{(0)}
\rightarrow
x_i^{(1)}
\rightarrow
x_i^{(2)}
\rightarrow
\cdots
\rightarrow
x_i^{(L)}
$$

token 才不断融合全局上下文，逐渐形成更丰富的高层语义。

因此正确心智模型是：

```text
局部像素
↓
Patch
↓
Visual Token
↓
多层 Self-Attention
↓
Context-aware Visual Representation
```

---

### 思考节点 C｜Attention 在 ViT 中发生在哪里

#### 问题

ViT 中的 self-attention，是不是直接在一个 patch 图片和另一个 patch 图片之间计算相关性？

### 解析与纠偏

这种说法方向接近，但不够准确。

真正参与 Q、K、V 运算的是 patch 对应的 visual token，而不是原始像素块本身。

因此更准确的表述是：

> ViT 在不同 visual tokens 之间计算 attention，从而学习图像不同局部区域之间的关系。

---

## 6. CNN Encoder 与 ViT Encoder：两种视觉建模方式

在 ViT 之前，CNN 长期是视觉编码的主流。

理解两者区别，有助于理解为什么现代 VLM/VLA 中 ViT 类视觉编码器非常常见。

### 6.1 CNN：从局部模式逐层构造全局语义

CNN 使用卷积核在局部区域滑动。

浅层往往首先提取：

- 边缘；
- 纹理；
- 颜色变化。

更深层再逐步组合：

```text
边缘
↓
局部形状
↓
物体部件
↓
完整物体
↓
场景语义
```

CNN 天然带有较强的 locality bias，也就是：

> 相邻像素通常更相关。

这是一种很强的 inductive bias。

---

### 6.2 ViT：通过 Token-to-Token Attention 建立长距离关系

ViT 先把图像拆成 token，再用 self-attention 建模。

理论上，一个图像左上角 token 可以直接与右下角 token 建立关系。

因此两者可以粗略比较为：

```text
CNN
局部 → 局部组合 → 更大感受野 → 全局

ViT
token ↔ token
通过 attention 建立长距离关系
```

对于机器人场景，很多真正重要的关系本来就是跨区域的：

```text
gripper ↔ cup
cup ↔ table
drawer ↔ handle
robot ↔ obstacle
```

因此 global relation modeling 对机器人视觉非常有价值。

---

### 6.3 ViT 并不是在所有情况下都优于 CNN

不能简单理解为：

```text
ViT > CNN
```

CNN 仍然具有：

- 强局部先验；
- 小数据条件下较好的数据效率；
- 一些边缘设备上的计算优势；
- 对 low-level visual pattern 的有效建模。

ViT 的重要优势则包括：

- 适合大规模预训练；
- 图像被统一表示成 token sequence；
- 容易与语言 Transformer 接轨；
- 天然适合 multimodal architecture。

对 VLM/VLA 来说，最后两点尤其重要。

语言已经是：

```text
language token
language token
language token
```

ViT 又把图像变成：

```text
visual token
visual token
visual token
```

于是两种模态在计算形式上都变成：

```text
token sequence
```

这为视觉和语言进入同一个大模型奠定了基础。

---

## 7. CLIP：让图像和文字进入共享语义空间

ViT 已经解决：

```text
Image
↓
Visual Representation
```

但还没有解决：

> 模型怎样知道视觉中的“杯子”和语言中的 “cup” 是同一个语义概念？

CLIP 的核心任务就是建立这种视觉—语言语义对齐。

---

### 7.1 双编码器结构

CLIP 可以抽象成两个 encoder：

$$
v_i=f_{\text{img}}(I_i)
$$

$$
t_i=f_{\text{text}}(T_i)
$$

其中：

- Iᵢ 是第 i 张图片；
- Tᵢ 是与图片配对的文本；
- f_img 是 image encoder；
- f_text 是 text encoder；
- vᵢ 是 image embedding；
- tᵢ 是 text embedding。

图像和文字最终被映射到相同维度的向量空间。

例如：

$$
v_i,t_i\in\mathbb{R}^{512}
$$

此时两种模态已经可以进行相似度比较。

---

### 7.2 Cosine Similarity：怎样判断图文语义是否接近

CLIP 常用 cosine similarity 衡量两个表示是否相似：

$$
\operatorname{sim}(v,t)
=
\frac{v^Tt}
{\|v\|\|t\|}
$$

直觉上，它主要比较两个向量“方向是否一致”。

对于一张红杯子图片和文本：

```text
"a red cup"
```

希望：

$$
\operatorname{sim}
(v_{\text{cup}},t_{\text{cup}})
$$

较高。

对于同一张图片和：

```text
"a dog"
```

则希望：

$$
\operatorname{sim}
(v_{\text{cup}},t_{\text{dog}})
$$

较低。

---

### 7.3 Batch 内的图文相似度矩阵

假设一个 batch 里有 N 对正确配对的 image-text 数据：

$$
(I_1,T_1),
(I_2,T_2),
\ldots,
(I_N,T_N)
$$

图像得到：

$$
v_1,v_2,\ldots,v_N
$$

文本得到：

$$
t_1,t_2,\ldots,t_N
$$

所有图片与所有文本两两比较后，得到相似度矩阵：

$$
S\in\mathbb{R}^{N\times N}
$$

其中：

$$
S_{ij}
=
\operatorname{sim}(v_i,t_j)
$$

理想情况下：

$$
S=
\begin{bmatrix}
\text{high} & low & \cdots & low\\
low & \text{high} & \cdots & low\\
\vdots & \vdots & \ddots & \vdots\\
low & low & \cdots & \text{high}
\end{bmatrix}
$$

因为：

```text
Image 1 ↔ Text 1
Image 2 ↔ Text 2
...
```

才是真实配对。

如果 batch size 为 32：

$$
S\in\mathbb{R}^{32\times32}
$$

---

### 7.4 Positive Pair 与 Negative Pair

正确图文配对：

$$
(I_i,T_i)
$$

称为 positive pair。

不匹配的：

$$
(I_i,T_j),
\quad i\neq j
$$

称为 negative pair。

对比学习的核心直觉是：

```text
Positive Pair
→ 拉近

Negative Pair
→ 推远
```

因此模型学到的不是固定类别编号，而是更加丰富的视觉—语言语义结构。

---

### 思考节点 D｜CLIP 到底在学什么

#### 问题 1

为什么 CLIP 需要 image encoder 和 text encoder 两个网络？

#### 问题 2

batch 中有 32 对 image-text 数据时，相似度矩阵 shape 是多少？

#### 问题 3

为什么希望相似度矩阵的对角线较大？

#### 问题 4

什么是 positive pair，什么是 negative pair？

#### 问题 5

ViT 和 CLIP 的核心区别是什么？

### 解析

图像和文字属于不同模态，原始表示形式完全不同，因此需要各自的 encoder 先转换到可比较的向量空间。

32 对数据会形成：

$$
32\times32
$$

的相似度矩阵。

理想情况下第 i 张图对应第 i 条文本，所以对角线表示正确配对，应具有较高相似度。

ViT 和 CLIP 不是同一个层次的概念：

```text
ViT
回答：
图像怎样被编码？

CLIP
回答：
图像表示与文本表示怎样建立语义对齐？
```

ViT 可以作为 CLIP 的 image encoder，但：

```text
ViT ≠ CLIP
```

---

## 8. 为什么 CLIP 比固定类别图像分类更适合 VLM/VLA

传统图像分类通常学习：

```text
Image
↓
Classifier
↓
固定类别
```

例如：

```text
dog
cat
cup
car
```

而 CLIP 一类方法可以利用自然语言描述：

```text
"a red ceramic cup on a table"
"a small blue object beside the robot"
"a person holding a glass"
```

于是模型能够学习：

- 物体类别；
- 颜色；
- 材质；
- 属性；
- 场景；
- 物体之间的高层关系。

这对于 VLA 很重要，因为机器人 demonstration 数据昂贵。如果视觉系统能预先从互联网规模图文数据中学习：

```text
cup
drawer
banana
red
left
holding
table
```

等语义概念，那么机器人训练阶段就不需要完全从零学习这些视觉知识。

---

## 9. Semantic Alignment 不等于 Spatial Grounding

这是本章最关键的边界之一。

CLIP 能够告诉模型：

> 这张图片和 “a red cup” 在语义上高度相关。

但机器人真正抓取时还需要知道：

- 红杯子具体位于哪里；
- 杯子的轮廓是什么；
- 杯柄方向如何；
- 夹爪相对杯子处于什么位置；
- 应该向哪个空间方向移动。

这些并不是同一个问题。

---

### 9.1 Semantic Alignment

Semantic alignment 主要回答：

```text
这个视觉内容意味着什么？
```

例如：

$$
\text{Image of red cup}
\leftrightarrow
\text{"a red cup"}
$$

这属于高层语义关联。

---

### 9.2 Spatial Grounding

Spatial grounding 更关注：

```text
这个语义对象具体在哪里？
```

例如：

```text
red cup
↓
图像右上区域
↓
具体空间位置
```

经典 CLIP 的训练主要强调 image-level embedding 与 text-level embedding 的全局语义对应，并不等于提供：

```text
patch 37 ↔ "cup"
patch 89 ↔ "table"
```

这样的精确局部监督。

因此必须明确：

```text
知道“图中存在杯子”
≠
知道“杯子精确在哪里”
≠
知道“怎样控制机械臂抓住杯子”
```

---

### 思考节点 E｜为什么 CLIP 还不够完成机器人抓取

#### 问题

为什么说 CLIP 主要完成 semantic alignment，而不能简单认为它已经解决了机器人视觉中的精确目标定位问题？

### 解析与纠偏

CLIP 的核心训练信号主要告诉模型：

> 整张图片和某段文本是否匹配。

它并没有天然提供精确的 pixel-level 或 patch-level object grounding。

机器人 manipulation 对视觉的要求比“识别图中是什么”更高，还需要：

- 目标位置；
- 局部形状；
- 空间关系；
- 与机械臂状态的相对关系。

因此：

```text
Semantic Understanding
只是 Manipulation 的前提之一
```

---

## 10. CLIP、SigLIP 与 DINO：不同类型的视觉预训练能力

现代 VLA 通常不会从零训练 vision encoder，而会复用预训练视觉 backbone。

本章重点建立三个名字的第一层定位：

```text
CLIP / SigLIP
更偏 Vision-Language Semantic Alignment

DINO
更偏强视觉自监督 Representation
```

这只是帮助阅读 VLA 架构的粗略心智模型，并不是绝对能力边界。

---

### 10.1 CLIP：强视觉—语言语义连接

CLIP 重点学习：

$$
\text{Vision}
\leftrightarrow
\text{Language Semantics}
$$

因此面对：

```text
"Pick up the red cup."
```

CLIP 类视觉 representation 能帮助模型把语言里的：

```text
red cup
```

与视觉场景里的目标对象建立语义关联。

---

### 10.2 SigLIP：与 CLIP 同属图文对齐路线

SigLIP 仍然解决 image-text alignment，但其训练目标采用 sigmoid-based image-text objective。

可以把每一对 image-text pair 粗略理解为一个判断问题：

```text
这一对是否匹配？
```

因此当前阶段需要记住的是：

```text
CLIP
↓
Contrastive-style Image-Text Alignment

SigLIP
↓
Sigmoid-based Image-Text Alignment
```

两者都主要属于视觉—语言语义预训练路线。

---

### 10.3 DINO：更强调从图像本身学习强视觉表示

DINO 类方法的核心并不是直接用文字监督视觉，而是通过视觉自监督学习获得稳定、结构化的 image representation。

可以形成一个简单心智模型：

```text
同一个视觉内容
↓
不同 crop / augmentation
↓
模型学习稳定视觉表示
```

这类 representation 往往能很好保留：

- object structure；
- local feature；
- spatial correspondence；
- shape；
- general visual representation。

因此在机器人场景里可以粗略理解为：

```text
CLIP / SigLIP：
“这个视觉内容在语言上意味着什么？”

DINO：
“这个视觉世界本身具有怎样的结构和局部特征？”
```

二者并非绝对互斥。CLIP 也保留视觉空间特征，DINO 也能产生语义信息。这里只是在 VLA 阅读中建立功能上的第一层区分。

---

## 11. 从 CLIP 到真正的 VLM

CLIP 更接近：

$$
\text{Image}
\leftrightarrow
\text{Text}
$$

也就是：

```text
这张图与这句话是否匹配？
```

完整 VLM 则希望实现：

$$
\text{Image}
+
\text{Language}
\rightarrow
\text{Understanding / Reasoning / Generation}
$$

例如：

```text
Image:
桌面上有红杯子、蓝盒子和苹果

Question:
"Which object is red?"

VLM:
"The cup."
```

这已经不是简单的图文相似度比较，而是联合理解和生成。

---

## 12. VLM 的三个核心模块

一种常见 VLM 可以抽象为：

```text
Image
↓
Visual Encoder
↓
Visual Features / Visual Tokens
↓
Projector
↓
LLM-compatible Visual Tokens
              \
               \
                → LLM Backbone
               /
Language Tokens
↓
Text / Multimodal Representation
```

核心模块有：

1. Visual Encoder；
2. Projector；
3. LLM Backbone。

---

### 12.1 Visual Encoder：负责“看”

Visual encoder 输入：

$$
I\in\mathbb{R}^{H\times W\times3}
$$

输出：

$$
Z_v
\in
\mathbb{R}^{N_v\times D_v}
$$

例如：

$$
Z_v
\in
\mathbb{R}^{196\times1024}
$$

表示：

```text
196 个 visual tokens
每个 token 1024 维
```

visual encoder 可以来自：

- ViT；
- CLIP vision encoder；
- SigLIP vision encoder；
- DINO 类视觉模型。

---

### 12.2 Projector：负责把视觉特征映射到 LLM 可接受空间

假设 visual encoder 输出：

$$
Z_v
\in
\mathbb{R}^{196\times1024}
$$

而 LLM hidden dimension 为：

```text
4096
```

语言 token 则可能表示为：

$$
Z_l
\in
\mathbb{R}^{N_l\times4096}
$$

此时 visual token 与 language token 的 feature dimension 不一致。

因此需要 projector：

$$
Z_v'
=
f_{\text{proj}}(Z_v)
$$

把：

$$
196\times1024
$$

映射成：

$$
196\times4096
$$

如果语言序列包含 6 个 token：

$$
Z_l
\in
\mathbb{R}^{6\times4096}
$$

则视觉 token 和语言 token 可以组成：

$$
[Z_v';Z_l]
\in
\mathbb{R}^{202\times4096}
$$

再送给后续 LLM。

---

### 思考节点 F｜Projector 为什么存在

#### 问题 1

Visual encoder、projector、LLM backbone 各自负责什么？

#### 问题 2

如果 vision encoder 输出是：

```text
[196, 1024]
```

而 LLM hidden dimension 是 4096，projector 之后 visual tokens 的 shape 应该是什么？

#### 问题 3

为什么不能简单把原始 vision encoder 输出直接塞进 LLM？

### 解析与纠偏

Visual encoder 负责把原始图像变成高层视觉特征。

Projector 负责把这些视觉特征映射到 LLM 可兼容的 representation space。

LLM backbone 负责把视觉和语言 token 放到统一上下文中进行联合理解、推理和生成。

第 2 题结果为：

$$
196\times4096
$$

第 3 题首先涉及 tensor shape：

```text
Vision:
[196, 1024]

Language:
[N_l, 4096]
```

二者 feature dimension 不一致，不能直接沿 token 维拼接。

但这还不是全部。

即使 visual feature 和 language feature 恰好都是 4096 维，也不意味着它们天然位于同一个语义表示体系。

因此：

```text
相同 Dimension
≠
相同 Semantic Space
```

Projector 不只是“改 shape”，还承担模态接口的学习功能。

---

### 12.3 Projector 不是 Normalization

把 projector 描述为“把视觉向量和语言向量归一化到相同维度”是不准确的。

Normalization 通常指类似：

$$
x
\rightarrow
\frac{x}{\|x\|}
$$

或者 LayerNorm 一类尺度标准化。

而 projector 的核心是：

$$
D_v
\rightarrow
D_{\text{LLM}}
$$

例如：

$$
1024
\rightarrow
4096
$$

所以它更准确地叫：

```text
learnable projection / mapping
```

而不是 normalization。

---

### 12.4 LLM Backbone：负责联合上下文建模

完成 projector 后，模型现在拥有：

```text
Visual Tokens
+
Language Tokens
```

LLM backbone 负责对两者进行联合建模。

因此 VLM 不再只是回答：

```text
这张图和这句话像不像？
```

而可以回答：

```text
这张图里哪个物体是红色的？
哪个物体在盒子左边？
用户指令指向的是哪个对象？
```

可以用一句话区分：

```text
CLIP：
对齐

VLM：
联合理解 + 推理 + 生成
```

---

## 13. Pretrained Backbone：为什么现代 VLM 不从零训练

现代 VLM 通常不会同时从随机初始化开始学习：

```text
怎么看图
+
怎么理解语言
```

更常见的做法是复用：

```text
Pretrained Vision Encoder
+
Pretrained LLM
```

这些模型主体称为 pretrained backbone。

视觉 backbone 已经从大规模视觉数据中学到：

- 物体；
- 场景；
- 纹理；
- 视觉语义。

语言 backbone 已经学到：

- 语言结构；
- 世界知识；
- 指令理解；
- 文本生成；
- 某种程度的推理能力。

因此 VLM 训练的一个重要目标变成：

> 如何让这两个已经具备强能力的 backbone 正确协作。

---

## 14. VLM 的训练：Multimodal Alignment 与 Instruction Tuning

可以用两个阶段建立基本心智模型：

```text
Stage 1
Multimodal Alignment

↓  

Stage 2
Multimodal Instruction Tuning
```

现实模型的具体训练流程可能更复杂，但这两个概念足以支撑当前对 VLM/VLA 的理解。

---

### 14.1 Stage 1：Multimodal Alignment

视觉编码器输出：

$$
Z_v
$$

projector 将其转换为：

$$
Z_v'
=
f_{\text{proj}}(Z_v)
$$

但随机初始化的 projector 并不知道：

> 什么视觉特征应该以怎样的形式进入 LLM。

因此可以使用 image-text paired data：

```text
Image:
一只狗坐在草地上

Text:
"A dog sitting on the grass."
```

进行训练。

这一阶段的核心作用可以理解为：

> 教会 LLM 怎样读取视觉编码器传来的信息。

---

### 14.2 为什么常常 Freeze 部分 Backbone

如果 vision encoder 和 LLM 已经经过大量预训练，一开始就完全更新所有参数：

```text
Vision Encoder
+
Projector
+
LLM
```

会带来：

- 很高的计算成本；
- 更大的数据需求；
- 破坏原有视觉/语言能力的风险。

因此一种常见策略是：

```text
Vision Encoder：freeze

LLM：freeze 或 largely freeze

Projector：train
```

可以把这理解为：

```text
视觉专家已经会“看”
语言专家已经会“理解和表达”

第一步先训练二者之间的“接口”
```

---

### 14.3 Stage 2：Multimodal Instruction Tuning

完成基本视觉—语言连接后，模型还需要学会：

> 根据不同 instruction 完成不同多模态任务。

例如：

```text
<image>

User:
What color is the cup?

Assistant:
The cup is red.
```

或者：

```text
<image>

User:
Which object is to the left of the box?

Assistant:
The cup.
```

训练仍然可以理解为 autoregressive next-token prediction：

$$
P
\left(
y_t
\mid
y_{<t},
I,
x
\right)
$$

其中：

- I 是输入图像；
- x 是用户 instruction；
- y₍<t₎ 是此前已经生成的回答 token；
- yₜ 是当前需要预测的 token。

对应监督损失可以写为：

$$
\mathcal{L}
=
-\sum_t
\log
P
\left(
y_t^*
\mid
y_{<t}^*,
I,
x
\right)
$$

其中 yₜ* 是训练数据中的正确输出 token。

Instruction tuning 的作用，是让模型从：

```text
“我可以读取视觉信息”
```

进一步变成：

```text
“我可以根据视觉信息和用户指令完成任务”
```

---

### 思考节点 G｜Alignment 与 Instruction Tuning

#### 问题 1

什么是 pretrained backbone？

#### 问题 2

Multimodal alignment 主要解决什么问题？

#### 问题 3

Instruction tuning 又主要解决什么问题？

### 解析

Pretrained backbone 是已经在大规模数据上完成预训练的主体网络，例如 vision encoder 或 LLM。

Multimodal alignment 更侧重：

```text
把视觉表示转换成语言模型可以消费的多模态表示
```

Instruction tuning 更侧重：

```text
让模型学会按照 instruction 使用这些多模态信息完成任务
```

两者不是同一件事。

可以用：

```text
Alignment：
先让两个模态“接得上”

Instruction Tuning：
再让模型“会按照要求使用它们”
```

来建立直觉。

---

## 15. CLIP Alignment 与 VLM Alignment 的区别

“alignment”一词在两个地方出现，很容易混淆。

### 15.1 CLIP 风格 Alignment

CLIP 更关注：

$$
\text{Image Embedding}
\leftrightarrow
\text{Text Embedding}
$$

核心问题是：

```text
图像和文本在语义上是否匹配？
```

### 15.2 VLM Multimodal Alignment

VLM 更关注：

$$
\text{Visual Features}
\rightarrow
\text{LLM-compatible Representation}
$$

核心问题是：

```text
怎样让 vision backbone 的输出真正被 LLM 理解和使用？
```

因此，两者都叫 alignment，但作用层次不同。

---

## 16. 完整机器人实例：机械臂 Pick-and-Place

考虑一个简单任务：

```text
场景：
桌上有红杯子、蓝盒子和苹果

语言指令：
"Pick up the red cup."
```

从本章角度，完整前端数据流可以拆解为：

### 第一步：相机采集 RGB 图像

```text
Camera
↓
RGB Image
```

### 第二步：Patchify

例如图像大小为 224 × 224，patch size 为 16：

$$
224\times224
\rightarrow
196\text{ patches}
$$

### 第三步：Patch Embedding

每个 patch：

$$
16\times16\times3
=
768
$$

展平并投影到 D 维 visual token。

例如：

$$
768
\rightarrow
512
$$

最终：

$$
X_{\text{vision}}
\in
\mathbb{R}^{196\times512}
$$

### 第四步：Vision Encoder

经过多层 ViT：

```text
Visual Tokens
↓
Self-Attention
↓
Context-aware Visual Features
```

此时某些 token 可以逐渐编码：

```text
red cup
blue box
apple
table
relative visual relations
```

### 第五步：语言 Tokenization

语言指令：

```text
"Pick up the red cup."
```

转成 language tokens。

### 第六步：Projector

vision feature 被映射到 LLM compatible dimension。

例如：

$$
196\times1024
\rightarrow
196\times4096
$$

### 第七步：Visual Tokens 与 Language Tokens 进入 LLM

```text
Visual Tokens
+
Language Tokens
↓
LLM Backbone
↓
Multimodal Understanding
```

此时模型可能已经理解：

- 目标物体是红杯子；
- 场景中还有蓝盒子和苹果；
- 用户意图是进行 pick-up；
- 指令中的 “red cup” 对应场景中的某个视觉对象。

但是：

> 这仍然不等于已经得到机器人动作。

---

## 17. 为什么 VLM 还不是 VLA

一个 VLM 可能输出：

```text
"The robot should pick up the red cup."
```

这仍然属于：

$$
\text{Vision}
+
\text{Language}
\rightarrow
\text{Text / Representation}
$$

而机器人真正需要的是动作，例如：

```text
Δx
Δy
Δz
Δr_x
Δr_y
Δr_z
gripper
```

也就是：

$$
\text{Vision}
+
\text{Language}
+
\text{Robot State}
\rightarrow
\text{Robot Action}
$$

因此从 VLM 到 VLA 还需要引入：

- robot state / proprioception；
- action representation；
- robot demonstration；
- policy learning；
- control execution。

---

### 思考节点 H｜CLIP、VLM 与 VLA

#### 问题

请区分下面三个系统：

```text
CLIP
VLM
VLA
```

它们分别主要解决什么问题？

### 解析

可以建立如下三层心智模型：

```text
CLIP
Image ↔ Text
重点：语义对齐

VLM
Image + Text
↓
Understanding / Reasoning / Generation

VLA
Image + Text + Robot State
↓
Robot Action
```

这是阅读现代具身智能论文时非常重要的一条层级边界。

---

## 18. 与传统控制 / Robot Learning 的联系

### 18.1 本章位于机器人系统的上游感知与语义表示层

从完整机器人系统来看：

```text
Camera / Sensors
↓
Perception / Visual Representation
↓
Vision-Language Understanding
↓
Learned Policy / VLA
↓
Action / Trajectory Generation
↓
Controller
↓
Actuator
↓
Robot / Environment
```

本章主要位于：

```text
Perception
+
Vision-Language Understanding
```

而不是底层控制器。

---

### 18.2 与传统控制的本质差异

传统控制一般假设：

```text
状态 x
目标 x_d
误差 e
```

已经是结构化数值变量。

例如：

$$
e_t=x_d-x_t
$$

控制器再根据状态和误差计算 control input。

VLM/VLA 的前端面对的却是：

- RGB image；
- natural language；
- scene semantics。

因此本章方法解决的是：

> 如何把高维非结构化输入转换成对决策有意义的表示。

它并不是 PID、LQR、MPC 或底层电机控制器的替代品，而是位于更上游的信息理解层。

---

### 18.3 与 Robot Learning 的连接

现代 robot policy 可以抽象为：

$$
\pi(a_t\mid o_t)
$$

VLA 的 observation 往往不只是低维状态，而可能包含：

$$
o_t=
(I_t,l,s_t)
$$

其中：

- Iₜ 是视觉 observation；
- l 是 language instruction；
- sₜ 是 proprioception / robot state。

本章主要构建：

```text
I_t + l
↓
Multimodal Representation
```

后续 Robot Learning 再学习：

```text
Multimodal Representation
+
Robot State
↓
Policy
↓
Action
```

---

## 19. 容易混淆的概念

### 19.1 Patch 与 Visual Token

Patch 是原始图像区域。

Visual token 是 patch 经过 embedding 后的网络表示。

```text
Patch
≠
Visual Token
```

---

### 19.2 Token Dimension 与 Token Matrix Shape

如果每个 visual token 是 512 维：

```text
D = 512
```

而一张图有 196 个 token：

```text
N = 196
```

那么整张图的表示是：

$$
196\times512
$$

而不是单独的 512。

---

### 19.3 Patch Embedding 与“非线性化”

最基本的 patch embedding 是线性 projection：

$$
x_i=p_iW_E+b_E
$$

不能简单描述为“非线性化”。

---

### 19.4 ViT 与 CLIP

```text
ViT：
视觉编码架构

CLIP：
视觉—语言对齐训练框架
```

ViT 可以成为 CLIP 的一部分。

---

### 19.5 CLIP 与 VLM

```text
CLIP：
图像和文字是否语义匹配

VLM：
结合图像和语言进行理解、推理和生成
```

---

### 19.6 Semantic Alignment 与 Spatial Grounding

```text
Semantic Alignment：
“这是什么？”

Spatial Grounding：
“它具体在哪里？”
```

机器人操作通常两者都需要。

---

### 19.7 Projector 与 Normalization

```text
Projector：
learnable mapping

Normalization：
尺度或分布归一化
```

二者不是同一个概念。

---

### 19.8 相同 Dimension 与相同 Semantic Space

两个向量同为 4096 维，只说明：

```text
shape compatible
```

不说明：

```text
semantic representation compatible
```

真正的模态兼容需要训练建立。

---

### 19.9 VLM 与 VLA

```text
VLM：
理解世界

VLA：
还必须产生可以驱动机器人的 action
```

“会描述怎么做”与“真正能输出动作”是两件不同的事。

---

## 20. 本章在整个 VLA 学习路线中的位置

本章对应的知识链是：

```text
Robot Policy
↓
Transformer / Attention
↓
ViT
↓
CLIP / SigLIP / DINO
↓
VLM
↓
Robot Learning / Behavioral Cloning
↓
Action Representation
↓
Action Chunking
↓
Diffusion / Flow Matching
↓
VLA
↓
Fine-tuning / Agent + VLA / VLA + RL
```

本章真正解决的是：

```text
Image
↓
Visual Representation
↓
Vision-Language Representation
```

下一阶段将继续解决：

```text
Demonstration
↓
Observation / State / Action
↓
Behavioral Cloning
↓
Policy
```

因此本章是从“Transformer 会处理 token”走向“机器人 policy 能处理视觉语言输入”的桥梁。

---

## 21. 局限与研究视角

### 21.1 语义能力强，不代表 manipulation 能力强

一个 vision-language backbone 即使能准确识别：

```text
red cup
drawer
handle
banana
```

也不意味着机器人已经能可靠抓取。

真正的 manipulation 还需要：

- spatial grounding；
- geometry；
- robot state；
- action generation；
- control；
- closed-loop correction。

因此论文如果声称“generalization 很强”，需要进一步问：

```text
泛化的是语义理解？
目标定位？
动作策略？
还是整个闭环任务？
```

---

### 21.2 Vision Encoder 的选择会影响 VLA 的能力结构

阅读 VLA 论文时应关注：

- vision encoder 是 CLIP、SigLIP、DINO 还是其他模型；
- 是否 freeze；
- 是否使用多个 vision encoder；
- visual token 是否保留局部空间结构；
- visual feature 如何进入 action policy。

不同视觉 backbone 的训练目标不同，因此可能为 VLA 提供不同类型的表示能力。

---

### 21.3 Projector 也是一个需要学习的关键接口

Projector 虽然通常比 backbone 小得多，但它承担：

```text
Vision Representation
↓
LLM-compatible Representation
```

这个关键接口。

阅读论文时可关注：

- linear projector；
- MLP projector；
- 更复杂 resampler / adapter；
- projector 是否单独预训练；
- 哪些模块与 projector 一起更新。

本章不提前展开这些具体架构，但应该形成：

> “模态接口本身也是模型设计问题”

这一研究意识。

---

### 21.4 Freeze / Unfreeze 是后续 Fine-tuning 的重要变量

视觉 backbone 和 LLM backbone 是否冻结，会直接影响：

- 训练成本；
- 数据需求；
- domain adaptation；
- 原有能力保持；
- 新 robot / 新 task 的适应能力。

这将在后续 VLA Fine-tuning 中进一步系统讨论。

---

### 21.5 VLM 到真机闭环仍然存在巨大距离

即使 VLM 完美回答：

```text
“The red cup is on the left side of the table.”
```

机器人仍然需要解决：

```text
机器人当前关节状态是什么？
↓
Action Space 是什么？
↓
目标动作怎样生成？
↓
是否预测 Action Chunk？
↓
底层控制器怎样执行？
↓
执行后如何得到新 Observation？
↓
环境改变后怎样重新修正？
```

因此视觉—语言理解是 VLA 的基础，但不是完整机器人系统。

---

## 22. 综合思考

### 综合题 1

给定输入图像：

```text
[3, 224, 224]
```

patch size 为 16，visual embedding dimension 为 768。

请说明从原始图像到 ViT 输入 token matrix 的完整 shape 变化。

#### 解析

图像被切成：

$$
14\times14=196
$$

个 patch。

每个 patch：

$$
16\times16\times3=768
$$

维。

因此：

```text
[3, 224, 224]
↓
196 patches
↓
[196, 768]
↓
Patch Projection
↓
[196, 768]
```

这里原始 patch flatten dimension 和 embedding dimension 恰好都为 768，只是数值维度相同，不代表 projection 没有意义。

---

### 综合题 2

某 VLM 使用：

```text
SigLIP Vision Encoder
+
Projector
+
LLM Backbone
```

请解释三个模块大概分别负责什么。

#### 解析

SigLIP vision encoder 负责把图像转成带有强视觉—语言语义信息的 visual features。

Projector 负责把视觉 feature 映射到 LLM-compatible representation space。

LLM backbone 负责把 visual token 和 language token 联合建模，进行理解、推理和生成。

---

### 综合题 3

为什么一个 VLM 已经能回答：

```text
“The red cup is on the left.”
```

仍然不能直接认为它已经能完成：

```text
Pick up the red cup.
```

#### 解析

因为前者主要证明：

```text
Vision-Language Understanding
```

而后者还要求：

```text
Robot State
↓
Action Representation
↓
Policy
↓
Trajectory / Action Generation
↓
Controller
↓
Closed-loop Execution
```

也就是说：

```text
语义理解
≠
机器人控制能力
```

---

### 综合题 4

如果一个 VLA 在语言指令泛化上很强，但对相机高度变化非常敏感，这种现象可能说明什么？

#### 解析

这通常说明语义层能力可能较强，但视觉空间表示或 policy 对视觉 observation distribution 的鲁棒性仍然有限。

因此研究 VLA 时不能只检查：

```text
模型是否认识目标物体
```

还应检查：

```text
视觉视角变化
相机高度变化
空间布局变化
局部 grounding
action policy robustness
```

这也是从“会理解图像”走向“能稳定控制机器人”的关键区别。

---

## 23. 本章总结

本章的核心不是记忆 ViT、CLIP、SigLIP、DINO 等模型名称，而是建立一条从像素到多模态表示的连续信息流。

第一，图像不能直接作为普通 Transformer token sequence，因此必须经历：

```text
Image
↓
Patchify
↓
Flatten
↓
Patch Embedding
↓
Visual Tokens
```

ViT 再通过 self-attention 建立不同视觉区域之间的上下文关系。

第二，仅有视觉表示还不足以连接语言。CLIP / SigLIP 通过 image-text pretraining 建立：

```text
Vision
↔
Language Semantics
```

但 semantic alignment 不能简单等同于 precise spatial grounding。

第三，完整 VLM 通过：

```text
Visual Encoder
↓
Projector
↓
LLM-compatible Visual Tokens
+
Language Tokens
↓
LLM Backbone
```

实现真正的视觉—语言联合理解。

第四，multimodal alignment 和 instruction tuning 分别解决：

```text
“两个模态如何接起来”
```

和：

```text
“模型如何按照指令使用多模态信息”
```

第五，VLM 仍然不是 VLA。VLA 还需要继续完成：

```text
Vision + Language + Robot State
↓
Policy
↓
Robot Action
```

因此本章最终应该形成如下核心框架：

```text
RGB Image
↓
Patchify
↓
Patch Embedding
↓
Visual Tokens
↓
ViT / Vision Encoder
↓
Visual Representation
↓
CLIP / SigLIP / DINO 等预训练能力
↓
Projector
↓
LLM-compatible Visual Tokens
+
Language Tokens
↓
LLM Backbone
↓
VLM
↓
后续 Robot Learning / VLA
```

只要能够解释这条链上每一层的：

```text
输入是什么
输出是什么
shape 是什么
解决什么问题
为什么需要这一层
```

就已经具备阅读大量 VLM/VLA 架构图的基础。

---

## 24. 关键词

**Vision Transformer (ViT) — 视觉 Transformer — 将图像切成 patch 并作为 token sequence 进行 Transformer 建模。**

**Image Patch — 图像块 — 从原始图像中切出的局部像素区域。**

**Patch Embedding — 图像块嵌入 — 把展平后的 patch 线性映射到 Transformer embedding space。**

**Visual Token — 视觉 token — 图像局部区域在网络表示空间中的向量表示。**

**Position Embedding — 位置嵌入 — 给 visual token 注入原始图像中的空间位置信息。**

**Vision Encoder — 视觉编码器 — 将 RGB 图像转换为高层视觉特征或 visual tokens 的模型。**

**CLIP — Contrastive Language-Image Pre-training — 通过图文对比学习建立视觉与语言的语义对齐。**

**SigLIP — Sigmoid Loss for Language-Image Pre-training — 使用 sigmoid-based image-text objective 进行图文预训练的方法。**

**DINO — Self-Supervised Vision Representation — 通过视觉自监督学习获得强视觉结构和特征表示的路线。**

**Cosine Similarity — 余弦相似度 — 用向量方向衡量两个 representation 是否接近。**

**Positive Pair — 正样本对 — 正确匹配的 image-text pair。**

**Negative Pair — 负样本对 — 不匹配的 image-text pair。**

**Contrastive Learning — 对比学习 — 通过拉近正样本、推远负样本学习 representation。**

**Multimodal Alignment — 多模态对齐 — 建立不同模态表示之间的可兼容映射关系。**

**Semantic Alignment — 语义对齐 — 建立视觉内容与语言概念之间的高层语义对应。**

**Spatial Grounding — 空间指代 / 定位 — 把语言概念落到图像中的具体区域或空间位置。**

**Projector — 投影 / 映射模块 — 将 vision feature 映射到 LLM 可以接收的 representation space。**

**Pretrained Backbone — 预训练主干网络 — 已经在大规模数据上训练好的 vision 或 language 主体模型。**

**Vision-Language Model (VLM) — 视觉—语言模型 — 联合处理图像和自然语言并进行理解、推理或生成。**

**Instruction Tuning — 指令微调 — 使用 instruction-response 数据训练模型按照任务要求输出结果。**

**Multimodal Instruction Tuning — 多模态指令微调 — 让模型根据图像和语言指令共同完成任务。**

**LLM Backbone — 大语言模型主干 — 负责多模态上下文建模、语言理解、推理与生成。**

**Vision-Language-Action Model (VLA) — 视觉—语言—动作模型 — 将视觉、语言和机器人状态进一步映射为机器人动作的策略模型。**
