### PCT: Point Cloud Transformer(2021)

> 论文地址：传送门

### Abstract

`PCT`基于`Transoformer`，`Transformer`天生是`permutation invariant`的，这更适合点云处理。为了更好地捕捉局部上下文，使用`FPS`和`KNN`进行`embedding`。

### Introduction

`Transformer`是`encoder-decoder`结构，包含三个主要模块，分别是`embedding`、`positional encoding`、`self-attention(核心)`，其中`self-attention`基于全局上下文为其输入特征生成精细的注意力特征。首先，`self-attention`以`input embedding`和`positional encoding`的总和作为输入，并通过线性层计算每个`word`的三个向量：`query`、`key`和`value`。然后，`query`矩阵和`key`矩阵相乘得到`attention weight`。最后，注意力特征是所有的`key`对应的`value`的加权和。显然，输出的注意力特征取决于所有的输入特征，这样一来，就能够学习到全局上下文信息。

本文基于`traditional Transformer`，关键思想是利用`Transformer`天生的`permutation invariance`来避免对点云的顺序重新定义，并且特征学习是通过注意力机制得来的。

由于点云和自然语言有很大不同，因此`PCT`必须做出一些调整：

- 基于坐标的特征编码；
  - 把`Transformer`中原始的`positional encoding`和`input embedding`合并至基于坐标的`input embedding`中，这可以生成`可辨别`的特征，因为每个点都有唯一的空间坐标表示。
- 可优化的`offset-attention`；
  - 用`self-attention`模块的`输入`和`注意力特征`之间的`offset(补偿)`来替换`注意力特征`。
  - 优点一：`相对坐标`相较于`绝对坐标`更具鲁棒性。
  - 优点二：拉普拉斯矩阵(度矩阵和邻接矩阵之间的补偿)在图卷机中非常有效。
  - 将点云看作是带有float邻接矩阵的图
- 邻居嵌入(局部信息提取)
  - 句子中的每一个单词都包含基本的语义信息，但是点云不行。
  - 注意力机制在提取全局特征上很有效，但是会忽视局部信息，使用点集之间的注意力模块进行特征提取。

### Transformer for Point Cloud Representation

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/15.png)

#### Encoder

目的：把输入点转换至更高维度的特征空间中。

首先，通过`input embedding`对输入坐标进行编码。使用`Conv1d`对每个点进行`特征提取`。

```python
self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

x = self.relu(self.bn1(self.conv1(x))) # B, D, N
x = self.relu(self.bn2(self.conv2(x)))
```

然后进入四个一样的`Attention`模块，为每个点学习语义丰富、具有区分性的表示。

```python
self.sa1 = SA_Layer(128)
self.sa2 = SA_Layer(128)
self.sa3 = SA_Layer(128)
self.sa4 = SA_Layer(128)

x1 = self.sa1(x)
x2 = self.sa2(x1)
x3 = self.sa3(x2)
x4 = self.sa4(x3)
```

然后用`残差连接`拼在一起。

```python
x = concat((x1, x2, x3, x4), dim=1)
```

然后进入线性(Linear + Batch Normalization + Relu)层生成`输出特征`。

```python
self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                               nn.BatchNorm1d(1024),
                               nn.LeakyReLU(scale=0.2))
x = self.conv_fuse(x)
```

`encoder`部分的结构和原始的`Transformer`基本一致。

#### Classification

对编码部分得到的特征分别进行`最大池化`和`平均池化`，将池化的结果(全局特征)放入2个`串联的LBRD层`，然后进入`Linear`层，得到最后的预测结果。

```python
x = jt.max(x, 2)
x = x.view(batch_size, -1)
x = self.relu(self.bn6(self.linear1(x)))
x = self.dp1(x)
x = self.relu(self.bn7(self.linear2(x)))
x = self.dp2(x)
x = self.linear3(x)
```

#### Segmentation

首先将得到的`全局特征(最大池化和平均池化结果)`和`每个点的特征`进行`拼接`。对需要进行分割的物体的类别进行编码(也就是说，用一串向量表示目前正在进行语义分割的物体的种类)。对该编码使用`1D卷积`进行特征提取，并将提取的特征与前面的`全局特征`、`局部特征`进行拼接。最后使用`1D卷积`对每个点的特征进行处理，得到每个点的不同种类的分类得分，进而得到最终的分类结果。

```python
x_max = jt.max(x, 2)
x_avg = jt.mean(x, 2)
x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
cls_label_one_hot = cls_label.view(batch_size,16,1)
cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
x_global_feature = concat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64
x = concat((x, x_global_feature), 1) # 1024 * 3 + 64 
x = self.relu(self.bns1(self.convs1(x)))
x = self.dp1(x)
x = self.relu(self.bns2(self.convs2(x)))
x = self.convs3(x)
```

