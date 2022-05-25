## Attention Is All You Need(2017)

> 论文地址：传送门

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

`Positional Encoding`：；

`Batch Normalization` 和 `Layer Normalization`对比：

![沐神画的](https://cdn.jsdelivr.net/gh/prannt99/blog/img/2.png)

`Batch Normalization`是竖着切，`Layer Normalization`是横着切，`Layer Normalization`用的更多。

`decoder`：做了一个自回归，即当前时刻的输入是上一时刻的输出(在预测时，不能看到之后时刻的输出)，但是在`attention mechanism`中，每一次都能看到完整的输入输出(包括未来时刻的输入输出)，所以要避免这种情况。方法是加入一个`masked attention`，保证在`t`时刻的输入不会看到`t`时刻之后的输入