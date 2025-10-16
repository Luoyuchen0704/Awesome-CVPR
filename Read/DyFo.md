# [CVPR2025]DyFo：一种无需训练的动态焦点视觉搜索，用于增强 LMM 的细粒度视觉理解
[DyFo论文](https://arxiv.org/abs/2504.14920)
[DyFo框架代码](https://github.com/PKU-ICST-MIPL/DyFo_CVPR2025)
## 1 摘要

人类能够轻松地在杂乱的环境中定位目标物体，这依赖于一种名为视觉搜索的认知机制，它能够有效地过滤掉无关信息，并聚焦于与任务相关的区域。受此启发，我们提出了 Dyfo（动态聚焦），这是一种无需训练的动态聚焦视觉搜索方法，能够增强大型多模态模型 (LMM) 中的细粒度视觉理解能力。与现有需要额外模块或数据收集的方法不同，Dyfo 利用 LMM 与视觉专家之间的双向交互，使用蒙特卡洛树搜索 (MCTS) 算法模拟类似人类的焦点调整。这使得 LMM 能够聚焦于关键视觉区域，同时过滤掉无关内容，而无需引入因词汇扩展或集成专门的定位模块而产生的额外训练。实验结果表明，Dyfo 显著提升了 LMM 中的细粒度视觉理解能力，并减少了幻觉问题，在固定分辨率和动态分辨率模型中均取得了卓越的性能。

## 2 代码解读

### 2.1 代码架构总览

```
.
├── dyfo
│   ├── scripts
│   └── src
├── figs
├── requirements.txt
└── README.md
```

* `dyfo/scripts/`：包含启动 `LMM server`、`visual expert server`、批量评估与流式推理脚本（例如 `pope/…_batch.sh`、`pope/stream_pope_*.sh`、`vstar/stream_vstar_*.sh`）

* `dyfo/src/`：存放核心模块代码（Focus Adjuster、Focus Tree Search、通信接口等）

* `figs/`：存放论文中的可视化结果或示例图像

* `requirements.txt`：项目依赖的 Python 包

* `README.md`：项目说明文档
  总体来看，该仓库采用模块化设计，把推理服务、脚本调度、算法核心分别隔离，便于扩展与调试。
### 2.2 关键技术实现
  
**1. 通信/服务框架**  

* 使用独立的server脚本分别启动LMM和视觉专家模型的服务端,`dyfo/scripts/lmm_server/<qwen/llava>_server.sh`,`dyfo/scripts/expert_server/start_server.sh`  

* 客户端通过RPC/HTTP/socket与各服务交互（传递图像片段、文本提示、返回响应）  
**2. 聚焦调节器（Focus Adjuster）**

* `dyfo/src/`中聚焦模块：根据当前聚焦状态和语义提示生成"动作"、

* 动作空间包括**语义聚焦**和**语义扩散**，模拟人类在视觉搜索中的聚焦/扩散行为

* LMM会给出当前关注区域的理解或文本提示，视觉专家根据提示产生局部mask/bounding boxes/分割

* 调节器根据LMM与视觉专家之间的"对齐度"或一致性决定下一步动作
**3. 聚焦树搜索（Focus Tree Search / MCTS）**

* 使用蒙特卡洛树搜索(MCTS)构建搜索树：
  
  * 选择阶段：依据UCT（上限置信区间）公式在树中选择节点
  * 扩展阶段：为当前叶节点扩展动作子节点
  * 模拟/评估阶段：对于新节点，基于LMM和专家一致性计算奖励
  * 反向传播：更新节点访问频次与价值

* 公式类似`a* = argmax [ Q(f,a) + w * sqrt( ln N(f) / N(c(f,a)) ) ]`

* 搜索结束后使用多节点投票机制整合多个子树节点的结果，决定最终聚焦区域或答案
**4. 投票与输出融合**

* 多个搜索路径可能给出不同聚焦区域或答案，系统最后用投票方式（多数同意）融合结果

* LMM在聚焦区域上生成最终回答输出
**5. 异步 / 并行支持**

* 推理过程可能采用异步接口或batch异步调用，提高效率
  
### 2.3 使用实例
  
可能的使用流程
  
```
# 假设有 dyfo 客户端模块接口 dyfo_client
from dyfo_client import DyFoClient

# 初始化客户端（连接 LMM server & expert server）

client = DyFoClient(lmm_address="http://localhost:port1",
                    expert_address="http://localhost:port2",
                    config={…})
# 给定图像与问题文本，开始动态聚焦搜索
result = client.infer_with_dyfo(image_path="input.jpg",
                               question="Is there a person on the bike?",
                               max_steps=10,
                               mcts_budget=100)
# 返回结构可能为：
# {
# "focusing_regions": [ (x1,y1,x2,y2), … ],
# "final_answer": "Yes, a person on bike",
# "search_path": [... steps metadata …]
# }

print("Answer:", result["final_answer"])
print("Regions:", result["focusing_regions"])
```
评估shell脚本调用
```

# 启动服务（两个 tmux 窗口）

bash dyfo/scripts/lmm_server/qwen_server.sh
bash dyfo/scripts/expert_server/start_server.sh

# 对 POPE 数据集进行批量评估

bash dyfo/scripts/pope/qwen_batch.sh

# 对单个子集进行流式推理

bash dyfo/scripts/pope/stream_pope_qwen.sh mcts False gqa/gqa_pope_random
```
## 3 文章分析

近年来，大规模多模态模型（Large Multimodal Models, LMMs）如 GPT-4V、Qwen-VL、LLaVA 等，在图文理解、视觉问答、图像描述等任务上取得了显著进展。然而，它们在**细粒度视觉识别和局部注意能力**方面仍存在不足，即所谓的"视觉幻觉"（Visual Hallucination）与"注意扩散"（Attention Diffusion）问题：

* 模型常常对图像中的无关区域进行冗余关注，导致错误推理；

* 在复杂场景或小目标识别时，模型缺乏动态聚焦能力，无法模仿人类的视觉搜索过程。

为了解决这些问题，论文提出 **DyFo (Dynamic Focusing)**，一种**无需训练**的后处理式框架，模拟人类"视线移动（saccade）"的动态视觉机制，使 LMM 在推理时能逐步聚焦到关键视觉区域，从而提升其对细粒度视觉问题的理解能力。

### 1 BLIP-2

* 2023年的论文，多模态人工智能领域的非常重要的里程碑模型

* 让图像和语言模型高效对话，无需从零联合训练一个多模态Transformer

* 核心结构：视觉编码器->Q-Former->大语言模型

* 视觉编码器是一个预训练的图像特征提取器，冻结参数不再训练可以减少计算成本

* Q-Former是一个中间翻译层，把视觉特征转成语言模型能理解的token，包含一组可学习查询向量，可视作视觉信息的摘要器

* 冻结的LLM大语言模型

* 高效、可扩展、灵活、迁移性强

* 适用任务：图像描述、视觉问答、视觉推理、图文匹配

* 是许多多模态模型的桥梁原型，如论文使用的LLaVA、Qwen-VL等

* 一种高效的视觉-语言连接框架，对接图像编码器和大语言模型，实现通用的多模态理解与生成能力
  
### 2 ViT

* Vision Transformer，计算机视觉CV领域最重要的基础模型之一，2020年

* 核心思想：把一张图片像文本句子一样处理，将图片拆分成固定大小的小块patch，每个patch输入到Transformer中，通过自注意力机制进行特征提取

* 将图像看成patch tokens的Transformer模型

* ViT结构与NLP的Transformer Encoder几乎一致，输入图像-》Patch切块+展平+线性映射-》加入位置编码-》Transformer编码器堆叠-》取CLS Token分类

* 图像分块，把224 * 224 * 3的图片按固定大小16 * 16分成小块，每个patch拉平成一个向量，作为视觉token，维度为16 * 16 * 3=768

* 线性映射，把每个patch的向量映射到固定embedding维度，如768

* 位置编码，显式加入位置信息，让模型知道patch在图像中的位置

* Transformer编码器堆叠，类似BERT结构，包括多头自注意力，前馈神经网络，残差连接，层归一化

* 分类Token，在所有patch token前加入一个特殊的CLS向量，经过多层Transformer后，汇总图像的语义信息，最后输入到分类头预测类别

* ViT优势在于全局感知能力强，迁移性好，结构简单统一，与语言模型兼容性强

* CNN擅长捕捉局部纹理特征，ViT擅长全局结构和语义关系
  
### 3 Qwen2-VL

* 阿里巴巴推出的通义千问的视觉-语言升级版本

* 支持处理任意分辨率图像

* 对位置编码的统一机制

* 对长视频的理解能力

* 多语言支持
  
### 4 LLaVA-Next

* 微软推动的大型多模态模型项目

* 支持高输入分辨率

* 支持图像-文本交错格式
  
### 5 Transformer

* 现代人工智能，自然语言处理NLP和多模态模型的核心架构，一种神经网络架构

* Google在2017年的论文,"Attention Is All You Need"

* 改变了深度学习的格局，提出一种完全基于注意力机制Attention的架构，取代了循环神经网络RNN和卷积神经网络CNN在序列建模中的作用

* 核心思想，让模型通过注意力机制自动学习不同部分之间的依赖关系，而不依赖顺序结构，如RNN

* 自注意力机制Self-Attention理解输入序列（图像patch、音频帧等）的元素关系

* 基本结构：Encoder编码器和Decoder解码器

* Encoder将输入序列（文本、图像patch）转换为高维特征表示，包含多层自注意力+前馈网络，常用于理解任务BERT

* Decoder生成输出序列，多层结构有交叉注意力，关注编码器的输出，常用于生成任务

* 每层Transformer Block包括：

* 多头注意力，并行学习

* 残差连接+归一化，稳定训练，防止梯度消失

* 前馈全连接层，对每个位置独立非线性映射

* 位置编码，注入顺序信息，Transformer没有顺序结构，不能像RNN自然感知顺序

* 优势：全局建模能力强，训练并行化（快），泛化能力好，可扩展性强
### 6 SEAL
* 克服MLLMs在处理高分辨率图像或视觉密集图片的局限性
## 4 研究方法
  
> 在不重新训练LMM的前提下，通过与外部视觉专家的交互式搜索，动态调整模型的视觉焦点。  

两个主要模块

1. 聚焦调节器
* 用于决定当前聚焦策略，在视觉空间中"放大"或"扩散"注意力
* 动作空间包括**语义聚焦**和**语义扩散**，由LMM的语义输出引导
* 该模块以LMM当前输出与视觉专家反馈之间的一致性作为奖励信号
2. 聚焦树搜索
* 借鉴蒙特卡洛树搜索MCTS的思想，对不同聚焦路径进行模拟和优化
* 通过UCT公式在节点间权衡"探索"和"利用"，逐步寻找最优聚焦路径
### MCTS
* Monte Carlo Tree Search（MCTS，蒙特卡罗树搜索）,是一种智能决策算法
* 结合了随机采样和树搜索
* 用于在高维、不确定、无法完全枚举的搜索空间中寻找最优决策，如游戏AI、强化学习和规划问题
* 通过模拟估计节点的好坏，通过选择平衡探索与利用
**选择select**
* 从根节点开始，选择最有潜力的子节点，直到未完全扩展的节点
* 常用选择策略置信度上限UCB1，节点访问次数少，UCB变大，鼓励探索；访问次数多，奖励高，保持利用
**扩展expand**
* 到达一个尚未完全展开的节点时，选择一个未探索的动作创建新的子节点
**模拟simulate**
* 从该新节点开始，使用随机策略或启发式策略模拟达到终止状态，模拟结果评估该节点价值
**回传reward**
* 将模拟结果沿路径反向传播到跟节点，更新经过节点访问次数和平均奖励值
迭代若干次后，最终在根节点选择访问次数最多的子节点作为决策输出
模拟和回传可以使用神经网络
### Focus Adjuster
