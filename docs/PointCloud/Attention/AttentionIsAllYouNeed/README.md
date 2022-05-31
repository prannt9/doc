## Attention Is All You Need(2017)

> 论文地址：[传送门](https://arxiv.org/abs/1706.03762)

<hr>

本文是`注意力机制`的开山之作。

<hr>



### **Abstract**

1. 序列转录模型依赖于包含`encoder-decoder`结构的复杂的递归或卷积神经网络；
2. 本文提出了`Transformer`，仅依赖于注意力机制，而不用考虑`RNN`或`CNN`；
3. `Transformer`首次提出是应用于`machine learning`方面，后面才用到了其他领域。

<hr>



### **Conclusion**

1. 介绍了`Transformer`模型，第一个做`序列转录`的模型，仅仅依赖于`注意力机制`，把所有的`recurrent layers`换成了`multi-headed self-attention`；
2. 对于翻译任务，`Transformer`比基于`CNN`或`RNN`的方法训练都快得多，在实际任务上的表现也很好

<hr>



### **Introduction**

1. RNN，尤其是LSTM和GRNN，在序列转录模型中成为了`state of the art`方法(2017年)；
2. 循环模型通常沿输入和输出序列的符号位置分解计算。在计算时间内将位置与步骤对齐，生成一个隐藏状态的序列`ht`，作为前一个隐藏状态ht−1的函数和位置t的输入。这种固有的顺序性难以并行化，这在序列长度较长的情况下变得至关重要；
3. 本文提出了`Transformer`，一种避免递归的模型架构，完全依赖于`注意力机制`来绘制输入和输出之间的全局依赖关系。`Transformer`允许更多的并行化

<hr>



### **Background**

1. `Extended Neural GPU`、`ByteNet`、`ConvS2S`使用`CNN`来替换`RNN`，以减少时序计算。但是使用`CNN`对比较长的序列难以建模；
2. 提出`multi-head Attention`可以模拟`CNN`中多输出通道的效果；

<hr>



### **Model Architecture**

1. `Auto-Regressive`：过去时刻的输出作为当前时刻的输入；`Transformer`也使用了`encoder-decoder`结构，把`self-attention`、`Point-Wise`、`Fully-Connected-Layers`堆在一起（`encoder`和`decoder`都用了这种方式）；
2. 画网络结构图可以使用`Visio`；
3. `Transformer`网络结构

![Transformer网络结构](https://cdn.jsdelivr.net/gh/prannt99/blog/img/1.png)

左侧为`encoder`，右侧为`decoder`。

`encoder`部分：

每个`encoder`包括`N = 6`，两个`sub-layer`：`Muti-Head Attention` + `MLP`，每个`sub-layer`都使用了残差连接，最后再使用`Normalization`，残差结构需要输出和输入大小一致(如果不一致还要做投影)，因此为了简单起见，把每层的输出维度变为`512`。	

`Input Embedding`：进来是一个一个词，表示成一个一个向量；

`Positional Encoding`：下面会专门讲；

`Batch Normalization` 和 `Layer Normalization`对比：

![沐神画的](https://cdn.jsdelivr.net/gh/prannt99/blog/img/2.png)

`Batch Normalization`是竖着切，`Layer Normalization`是横着切，`Layer Normalization`用的更多。

`decoder`：做了一个自回归，即当前时刻的输入是上一时刻的输出(在预测时，不能看到之后时刻的输出)，但是在`attention mechanism`中，每一次都能看到完整的输入输出(包括未来时刻的输入输出)，所以要避免这种情况。方法是加入一个`masked attention`，保证在`t`时刻的输入不会看到`t`时刻之后的输入输出。

4. Attention

注意力函数是一个映射：把`query`、`key-value pairs`映射为一个`输出`，其中`query`、`key`、`value`、`output`都是向量。`output`是`value`的加权和（输出维度和`value`维度一致），对每一个`value`，权重是`key`和`query`的相似函数算出的，对于相似函数（compatibility function），不同的注意力机制有不同的算法。

不同的相似函数会导致不一样的注意力算法。接下来论文讲的是，论文自己用到的注意力算法。

`Scaled Dot-Product Attention`：

输入为维度都为`dk`的`query`和`key`以及维度为`dv`的`value`，然后让一个`query`和所有的`key`做内积，再除以一个常数，最后通过一个`sorftmax`层获得权重，再把权重作用于`value`上得到最终的输出。

最后注意力机制的公式长这样：
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

在图1中，三个`multi-head attention`的区别：

1. 编码器中的`multi-head attention`是自注意力的，将本身复制三分，即作为`key`，也作为`value`，还作为`query`；
2. 解码器中的`masked multi-head attention`也是自注意力的，`masked`作用是：防止看到未来时刻的输出；
3. 解码器中的`multi-head attention`不是自注意力的，因为`key`和`value`来自于编码器的输出。

在`Attention`的过程中，感兴趣区域已经被抓取出来，并且做一次`aggregation(聚合)`，因此后面的`MLP`、`projection`可以独立做。

`RNN`和`Transformer`的区别：`RNN`是把上一时刻的输出传入下一时刻作为输入，`Transformer`是先通过`Attention`层，然后再全局的抓取信息，最后再独立做`MLP`和`Projection`。

**Embedding and Softmax**

把`词`映射为`向量`，维度为`512`(编码器和解码器，以及softmax之前都需要`Embedding`)，三个`Embedding`使用的是相同的权重

**Positional Encoding**

加入这个模块是因为`Attention`不具有时序信息。`Transformer`的输出是`value`的加权和，而权重是`query`和`key`之间的距离，和序列信息无关。换句话说，给定一个句子，把顺序打乱之后，`Attention`的结果都是一样的。显而易见，这在处理时序数据的时候是有问题的(例如，给定一句话，把句子打乱之后，语义肯定会变化，但是`Attention`不会处理这种情况)。`Positional Encoding`出现在输入之后，进入网络之前的部分，把词在句子中的位置获得到。

### **Why Self-Attention**

|   Layer Type   |  Complexity  | Sequential Operations | Maximum Path Length |
| :------------: | :----------: | :-------------------: | :-----------------: |
| Self-Attention |   O(n·n·d)   |         O(1)          |        O(1)         |
|   Recurrent    |   O(n·d·d)   |         O(n)          |        O(n)         |
| Convolutional  | O(k·n·n·d·d) |         O(1)          |      O(log(n))      |
| Self-Attention |   O(r·n·d)   |         O(1)          |       O(n/r)        |

### **Experiment**

没啥重点，就是介绍参数。

### 待读论文

 [一文读懂「Attention is All You Need」| 附代码实现]( [一文读懂「Attention is All You Need」| 附代码实现 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247486960&idx=1&sn=1b4b9d7ec7a9f40fa8a9df6b6f53bbfb&chksm=96e9d270a19e5b668875392da1d1aaa28ffd0af17d44f7ee81c2754c78cc35edf2e35be2c6a1&scene=21#wechat_redirect) ) 

 [Attention is All You Need | 每周一起读](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIwMTc4ODE0Mw%3D%3D%26mid%3D2247484985%26idx%3D1%26sn%3Df8cb392ffbeb26c954d7ee3059364be1%26chksm%3D96e9d9b9a19e50af671441ac18b6ec9a3ec7a30722ff6f71120f8b2c3afd063c5c9f96c0998a%26scene%3D21%23wechat_redirect) 

[Implementation](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

[深度学习中的注意力机制](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzA4Mzc0NjkwNA%3D%3D%26mid%3D2650783542%26idx%3D1%26sn%3D3846652d54d48e315e31b59507e34e9e%26chksm%3D87fad601b08d5f17f41b27bb21829ed2c2e511cf2049ba6f5c7244c6e4e1bd7144715faa8f67%26mpshare%3D1%26scene%3D1%26srcid%3D1113JZIMxK3XhM9ViyBbYR76%23rd) 

[transformer的实现(pytorch版， 基本上复现了大部分思想)](https://link.zhihu.com/?target=https%3A//github.com/yyHaker/pytorch-transformer) 

[The Illustrated Transformer](https://link.zhihu.com/?target=https%3A//jalammar.github.io/illustrated-transformer/) 