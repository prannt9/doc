### Vision Transformer(2021)

> 原文链接：[传送门](https://arxiv.org/pdf/2010.11929.pdf)
>
> 看模型的表现：www.paperswithcode.com

**Transformer**是`Google`团队在2017年提出的一种`NLP`经典模型，现在比较火热的`Bert`也是基于`Transformer`。

**Transformer**模型使用了`Self-Attention`机制，**不采用RNN**的顺序结构，使得模型可以并行化训练，而且能够**拥有全局信息。**

但对于视觉任务来说，训练和部署这些模型的成本非常高，也就产生了设计轻量化Transformer的需求。

### Abstract

在计算机视觉领域，`Attention`要不就是和CNN一起用，要不就是替换其他经典网络的的某一部分，但是保持整体结构不变。本文证明了以上两种方式都是不必要的。一个纯的Transformer直接作用于一系列图像块的时候，在图像分类任务上也能表现的很好。尤其是在大规模数据集上先做预训练，然后再迁移到中小型数据集时，ViT能获得和最好的CNN相媲美的结果。

### Introduction

#### 回忆Transformer

> **1.如何把2d的图片变成1d的序列？**
>
> - 把像素点当作Transformer输入中的单词，然后直接把2d图片拉直，放到Transformer中自己和自己学。这样做的缺点是复杂度太高。假设图像的像素为`224 x 224`，则序列长度为50176，是BERT序列长度的100倍。这还只是分类任务，如果是语义分割、检测任务，复杂度更加难以想象；
> - 最近的工作要么把Transformer和CNN混在一起用(不用图片当作Transformer的直接输入，把网络中间的feature map当作Transformer的输入，例如ResNet，其最后一层的feature map size只有14 x 14)，要么把整个卷积层都换成自注意力(孤立自注意力和轴注意力)。
>   - Stand-Alone Attention：复杂度高是因为使用整张图作为输入。那现在就不用整张图了，而是用一个局部的小窗口，这个复杂度是可以控制的(相当于回到卷积的那一套操作了)
>   - Axial Attention：图片的复杂度高是因为他的序列长度为`N = H x W`，现在希望把N这个2d的矩阵拆分成2个1d的向量。所以轴注意力先在H维度上做一次self-attention，再在W维度上做一次self-attention。(相当于把在2d矩阵上进行自注意力操作转换为两个1d的顺序操作)
> - 这些方法其实都是在做一件事：上面提出的问题是序列长度太长，没法把Transformer用到视觉中，这些方法都是在想办法降低序列长度。

本文是将标准的Transformer直接作用于图片，尽量不做修改。ViT将一张图片打成很多个patch，每个patch的大小为`16 x 16`，则新的图片长度为`N = 14 x 14 = 196`，把每一个patch当作一个元素，输入进`fc layer`，得到一个`linear embedding`，把`linear embedding`当作输入传入给`Transformer`。这些Image patch可以视为NLP里的单词。本文采用的是有监督的训练。

Inductive Bias(归纳偏置)：指的是一种先验知识，提前做好的假设。比如对于CNN来说，有两个归纳偏置：

- locality：由于CNN是以滑动窗口的形式在图片上进行卷积的，所以假设图片上相邻的区域会有相邻的特征(比如桌子和椅子通常上在一起的，靠的越近的物体相关性就越强)
- translation equivariance(平移不变性)：无论是先做卷积操作，还是先做平移操作，结果是不变的。即`f[g(x)] = g[f(x)]`。

但是对于Transformer来说，并没有这些先验信息。作者选择在大规模的数据集上做预训练，且预训练的效果比归纳偏置好。

> 总结：
>
> - 因为Transformer在NLP领域扩展的很好，自然就会有一个问题，如果把Transformer应用到视觉中，是不是视觉中的性能也能得到大幅提升呢？
> - 讲前人的工作，这么好的方向不可能之前没人研究过，讲清楚自己的工作和别人的工作区别在哪里；
> - ViT就是用了一个标准的Transformer模型，只要对图片做一些预处理(把图片打成patch)，送入Transformer就可以了，不需要别的改动，这样就可以把视觉问题理解为NLP问题；
> - 结果很好。

### Conclusion

和之前的工作不同之处在于：除了在刚开始把图片打成patch，并未引入图像特有的归纳偏置，这样的好处就是不需要对Vision领域有什么了解或者domain knowledge。

### Related Work

目前大多数的基于Transformer的方法都是先在大规模数据集上做预训练，然后在目标任务上做微调。有两项工作比较有名，分别是BERT(完形填空，有一个句子，划掉其中的某些词，再predict这些词)和GPT(已经有了一个句子，去预测下一个词是什么)。

待读论文：MAE(何凯明：Masked Autoencoders Are Scalable Vision Learners)，该论文研究的是生成式模型，而非判别式模型。

### Method

在模型设计上，尽可能和原始的Transformer保持一致，这样做的好处是可以把NLP领域中成功的架构拿过来用，就不用自己再魔改模型了。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/19.png)

流程：

- 把原始图片打成九宫格的patch，然后把patch变成一个序列；
- 经过线性投影，每个patch会得到一个特征(patch + position embedding)
- 由于自注意力是所有的元素两两之间做交互，所以本身并不存在顺序问题，但是图片是一个整体（九宫格是有自己的顺序的，顺序颠倒就不是原来的图片了）
- 加上位置编码后，token就既包含了图片块原有的图像信息，又包含了该图片块所在的位置信息。
- 得到了一个一个的token之后，接下来的工作和NLP就一样了，输入进**Transformer Encoder**，**Transformer Encoder**会得到很多输出。那问题来了，这么多输出，该拿哪个输出去做最后的分类呢？借鉴BERT，BERT中有`Extra Learnable Embedding(分类字符)，图中用0*代替`
  - embedded patches -> Layer Normalization -> Multi-Head Attention
- 因为token和token之间在做交互，所以`分类字符`能够从别的`embedding`里学到信息，从而只需要根据`分类字符`的对应输出做最后的判断就可以了