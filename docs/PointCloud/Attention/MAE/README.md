### Masked Autoencoders Are Scalable Vision Learners

> 原文链接：https://arxiv.org/pdf/2111.09886.pdf

本文和之前文章的**区别**：

**Transformer：**基于纯注意力机制的编码器和解码器，在机器翻译任务上的表现比RNN好；

**BERT：**将Transformer的编码器拓展到更一般的NLP任务上，使用类似于**完形填空**的自监督训练，也就不需要使用标签信息；

**ViT：**将Transformer模型用到CV领域，把整张图片打成许多**16 x 16**的patch，然后送入Transformer中；

**MAE：**可以认为是**BERT的CV版本**，**基于ViT**，且把整个训练拓展到**无标签**的数据上；

**标题：带掩码的自编码器是可拓展的视觉学习器**

注1：假设算法比较快，就可以在标题里写**efficient**，假设做的东西比较大，就可以在标题里写**scalable**；

注2：masked一词来自于BERT，每次mask掉一部分，然后去预测这一部分；

注3:在Transformer、BERT中，用的都是encoder这个词，auto在这里不是自动的意思，而是标签(y)和样本(x)来自于同一个东西。比如在语言模型中，每次用前面那些词去预测下面一个词。由于x和y都是来自于同一句子里面的词，所以叫auto；

注4：本标题的模板：xxx are xxx(最近流行的梗，把结论放在了title里，且这种句式比较客观)。

### Abstract

本文提出了MAE，是CV领域可拓展的自监督学习器。随机盖住输入图像中的patch，然后重构这些被盖住的patch**的像素**。和BERT不同的是，本文盖住的是patch，预测的是patch中的所有像素。两个核心设计：一是开发了一个不对称的`encoder- decoder`架构，编码器仅作用在可见的(visible)patch子集中，如果该patch被mask掉了，编码器就不会对该patch进行编码(这样子能省一点点的计算时间)，解码器能够从masked token中重建原始的图片。二是如果mask掉大量的原始图片(e.g. 75%)，就会得到一个有意义的自监督任务(如果只遮住几块的话，那查下值就能出来了，模型也学不到太多有意义的东西；但是把一大半遮住，就会迫使学到一些更好的特征)。这两个设计合在一起就能高效地训练大的模型。



