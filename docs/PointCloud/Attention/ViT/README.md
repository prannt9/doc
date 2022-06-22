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

和之前的工作不同之处在于：除了在刚开始把图片打成patch，并未引入图像特有的归纳偏置，这样的好处就是不需要对Vision领域有什么了解或者domain knowledge。这一简单、扩展性好的策略跟大规模预训练结合起来时，效果出奇的好。

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

前向过程总结：

- 假设有张图片(224 x 224 x 3)，如果使用16 x 16的patch size，就会得到14 x 14 = 196个图像块，每个图像块的维度为16 x 16 x 3 = 768，所以原始图片就被变为了：224 x 224 x 3 -> 196 x 768；
- 线性投射层就是全连接层(E)，其维度为：768 x 768(后面的768就是文中说的D，D是可以变的，但是前面的768是从图像的patch算出的(16 x 16 x 3)，这个是不变的)；
- 经过线性投射层就得到了patch embedding(X · E = 196 x 768)，然后再乘以(768, 768)的矩阵，最后得到的还是(196, 768)，意思是：有196个图像块，每个图像块的维度是768；
- 到现在为止，就成功的把Vision的问题变为了NLP的问题。输入说一系列**1d**的token，**而非2d**的图了；
- 除了图像本身的token，还要加一个额外的cls token(借鉴BERT，维度为1 x 768)，可以直接拼接。所以最后序列的长度就是整体进入Transformer的长度+1(196 + 1 = 197)，最后的尺寸为197 x 768；
- 1，2，3，...，9只是序号，而非我们真正使用的位置编码，因为不可能把数字传给Transformer去学。具体的做法是，维护一张表，表里的每一行代表了1，2，3，...，9这些序号，每一行是一个向量(该向量的维度 = D的维度(768))，这个向量是可以学的，然后把位置信息和token信息进行相加(注意是sum，不是concat)，加完之后的序列还是197 x 768；
- 如上图(右图)所示，也就是说Transformer Encoder的输入Embedded patches的维度上197 x 768，进入Layer Norm，输出还是197 x 768，然后进入Multi-Head Attention，变成了三份(k,q,v)，每一份都是197 x 768，由于做的是多头自注意力，所以最后的维度并非768，假设用的是ViT的base版本(多头用了12个头)，维度就变成了768➗12=64，也就是说(k,q,v)都变为了197 x 64(12个(k,q,v)去做self- attention)，最后再把这12个头的输出拼接起来，又变成了197 x 768。再过一层Layer Norm，还是197 x 768，再过一层MLP，先把维度放大(放大4倍)，变为197 x 3072，再缩小(投射回去)，变为197 x 768；
- 进去是197 x 768，出来还是197 x 768，然后叠加多个这样的BLOCK，想加多少加多少。L层的Transformer Block就是Transformer Encoder。

### Experiment

#### Ablation Experiment 

##### 1.class token 

为了和原始的Transformer模型尽可能保持一致，本文也使用了class token，当作图像的整体特征。该class token的输出通过一层MLP就可以做分类预测了，MLP使用**tanh**当作非线性激活函数。原来不是这么做的，比如有个残差网络(ResNet50)，假设其包含Res2、Res3、Res4、Res5四个Block，Res5出来的是一个feature map(14 x 14)，在该feature map之上做了一个GAP(global average-pooling)操作，池化后的特征被拉直称一个向量，拿这个特征去做分类。现在对于Transformer来说，进去N个元素，出来也是有N个元素。问题来了，为什么不能用这N个输出去做GAP，得到最后的特征，而是在前面加一个class token，最后用class token去做输出？作者通过实验得出的结论：其实这两种方式都可以。

##### 2.Positional Embedding

作者对比了三种位置编码：1-d positional embedding，2-d positional embedding和relative positional embedding。

1-d positional embedding：NLP中常用的位置编码，也是本文从头到尾都在使用的位置编码；

2-d positional embedding：把一个图片打成9宫格，表示为：1-9，而2-d位置编码表示为11，12，13，21，...，33；

relative positional embedding：token和token之间的距离可以用相对距离来表示(offset)。

作者通过实验得出的结论：其实这三种方式都可以。

