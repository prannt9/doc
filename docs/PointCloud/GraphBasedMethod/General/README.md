### A Gentle Introduction to Graph Neural Networks(2021)

> 本文是一篇博客上发表的文章, 对GNN作了总体的介绍。
>
> 地址: https://distill.pub/2021/gnn-intro/
>
> 作者还发了另一篇x相关文章:https://distill.pub/2021/understanding-gnns/
>
> 地址注释: 该博客下的所有文章包括的图均为交互图, 因此可以双击看效果。

### Sub Title

神经网络被用于处理图的结构和性质上，本文旨在探究构建一个GNN需要哪些模块，以及其背后的思想。

从交互图中可以看出：每一个node(节点)都是由下一层中自身node和邻居node组成的。

### 正文

`A set of objects, and the connections between them, are naturally expressed as a graph.` GNN的应用包括`抗菌剂的发现`、`物理仿真`、`假新闻辨别`、`交通预测`、`推荐系统`。

本文分为以下四块进行介绍：

- 什么样的数据可以表示为一张图；
- 图和别的数据有什么不一样；
- 构建GNN, 看看每个部分是什么样子，从一个`bare-bone`的实现到`SOTA`的GNN；
- 提供了一个GNN的`playground`， 可以在这里体验真实的任务和数据集，了解GNN模型的每个组件是如何对其做出的预测做出贡献的。

> Definition: A graph represents the relations (*edges*) between a collection of entities (*nodes*).

#### Images as graphs

- 通常将`image`视为带有图像通道的矩形网格，将其表示为阵列（例如244x244x3）；
- 将`image`视为具有规则结构的图，其中每个像素表示一个`node`，并通过`edge`连接到相邻像素。

#### Text as graphs

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/3.png)

如上图所示：每个词向量都被指向下一个词向量。

#### Examples

1. 分子结构的图表示(香料、咖啡因)；
2. 社交网络的图表示；
3. 文章引用的图表示。

### 图结构数据的问题

- graph-level；
  - 预测整个图的属性。例如，对于一个以图表示的分子结构，我们想要预测该分子的气味，或者它是否会与疾病相关的受体结合。这类似于`MNIST`和`CIFAR`的图像分类问题，我们希望将标签与整个图像相关联。对于文本，一个类似的问题是情绪分析，我们想立刻识别整个句子的情绪或情绪。
- node-level；
  - 预测图中每个节点的一些属性。给定一个图，预测每个顶点上的属性。
- edge-level
  - 预测图中边的属性或存在性。给定一个图，预测每条边上的属性。

### Challenge

将`graph`用到神经网络上最核心的问题：如何让`graph`和神经网络变得兼容。`graph`包含四种信息:`nodes`， `edges`， `global-context` and `connectivity`，前面三种都很好表示，而`connectivity`表示起来相对较难，最明显的选择就是使用`邻接矩阵`，但是`邻接矩阵`有一些缺点，如`node`数量过多，每个`node`连接的边也变化非常大，这就会导致非常稀疏的邻接矩阵,空间利用率会很低。

GNN应该：无论`graph`怎么变化，表示的都是同一个`image`(相当于`同分异构体`)。

> **批注：**也正因如此，使用GNN进行点云语义分割不需要PointNet中的`max pooling`层，因为`max pooling`层是用来保证点云排列不变性的。
>
> **问题：**上面的批注针对`image`可以，针对三维点云也可以？？？

### Graph Neural Network

重要摘抄：

> 1. Now that the graph’s description is in a matrix format that is permutation invariant, ... 
> 2. A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances).

GNN会对属性(node, edge, global)进行变换，但是不会改变图的连接性(边连接哪些顶点信息不会改变)。

#### The simplest GNN

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/截屏2022-05-29 20.11.23.png)

对于全局向量(Un)、顶点向量(Vn)、边向量(En),分别构造一个`MLP`(MLP输入和输出大小一致)，得到的输出作为下一次更新(经历`MLP`之后属性都是被更新过的,但是`graph`并未发生变化)，这三个`MLP`组成了一个`GNN`层。由于`MLP`是对每个向量单独作用的，因此不会考虑所有的连接信息，所以对顶点做排序并不会改变结果(permutation invariant)。

`最后一层输出怎样得到预测值?`

如果是`二分类`问题， 加一个输出维度为`2`的全连接层(所有顶点共享一个全连接层)， 再加一个`softmax`；

如果是`n分类`问题，加一个输出维度为`n`的全连接层(所有顶点共享一个全连接层)，再加一个`softmax`；

在GNN中,不管图有多大，一层里面只有`3个`MLP，所有顶点共享一个MLP，所有边共享一个MLP，全局使用一个MLP。

#### 稍微复杂点的GNN

假如没有`node`信息，只有`edge`的信息，但仍需对`node`做预测，我们就需要从`edge`中汇聚(pooling)信息,并且把该信息给到`node`。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/4.png)

如上图所示，假设想要对某一`node`做预测，该点本身并没有向量信息，就可以把和该点连接的边的向量拿出来，把全局向量拿出来(共5个向量)，把这5个向量加起来,就得到了顶点的向量。下图为转换过程。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/5.png)

类似的，若仅有`node`信息，无`edge`信息，但想要预测`edge`信息，如下图所示：

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/6.png)

完整的过程如下：

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/7.png)

**描述：**给定一个输入的图，首先经历GNN层(由三层MLP组成)，就得到了保持图结构的输出(但是所有的属性已经改变)，最后根据要对哪个属性做预测，添加合适的输出层；如果确实信息,就加入合适的pooling层。

**局限性：**在网络结构中没有使用图的连接信息(对属性变换时，就是每个属性自己进入MLP，并未看到该顶点是和哪些边相连)

#### Passing messages between parts of the graph

使用信息传递，邻域节点和边可以交换信息并影响到彼此的下一次更新。

`message passing`的步骤：

- 对于图中的每个点,聚合(gather)邻域节点信息；
- 通过聚合函数聚合所有信息;
- 所有的池化信息都进入更新函数。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/8.png)

**描述：**假设对某一顶点的向量进行更新，之前的做法是就把该向量本身拿出来进入MLP层，得到更新之后的向量。在信息传递中，把该节点本身的向量和邻居节点的向量加在一起得到一个汇聚的向量，把汇聚的向量再进入MLP得到该点向量的更新。











