### PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies(2022)

> 论文地址：https://arxiv.org/pdf/2206.04670v1.pdf
>
> 代码地址：https://github.com/guochengqian/PointNeXt

### Abstract

**PointNet++**是理解点云最有影响力的神经架构之一。虽然PointNet++的准确性已被**PointMLP**和**Point Transformer**等最新网络大大超越。但作者发现，性能提高的很大一部分是由于**改进了训练策略**，即数据增强和优化技术，以及增加了模型尺寸，**而非架构创新**。因此，PointNet++的潜力还有待挖掘。本文通过对**模型训练**和**缩放策略**的系统研究，重新审视了经典的PointNet++，并提供了两个主要贡献。首先，提出了一组改进的训练策略，显著提高了PointNet++的性能；其次，在**PointNet++**中引入了**inverted residual bottleneck**，以实现高效的模型缩放，并提出了PointNets的下一个版本**PointNeXt**。**PointNeXt**可以灵活扩展，在3D分类和分割任务上都优于最先进的方法。对于**分类**，PointNeXt在**ScanObjectNN**的总体准确率达到87.7%，超过PointMLP 2.3%，同时推理速度快10倍。对于**语义分割**，PointNeXt在**S3DIS**上以74.9%的mIoU取得了**SOTA**，优于最近的Point Transformer。

### Conclusion 

本文证明了通过**改进训练**(improved training)和**缩放策略**(scaling strategies)，可以提高PointNet++的性能，使其超过当前的SOTA。更具体地说，作者量化了目前广泛使用的每种数据增强和优化技术的效果，并提出了一套改进的训练策略。这些策略可以很容易地应用于提高PointNet++和其他代表性作品的性能。还将**Inverted Residual MLP block**引入PointNet++以开发PointNeXt。证明了PointNeXt在保持高吞吐量的同时，在各种基准上比PointNet++具有更高的性能和可扩展性。

### Introduction

PointNet++体系结构太简单，无法学习复杂的点云表示。本文重新审视了PointNet++这一经典且广泛使用的网络，并发现其潜力尚待挖掘，这主要是由于两个因素：1.卓越的训练策略；2.有效的模型缩放策略。这两点在PointNet++提出的年代是不具备的。

通过对各种基准的综合实证研究，例如**ScanObjecNN**用于对象分类，**S3DIS**用于语义分割，我们发现训练策略，即数据增强和优化技术，对网络性能起着重要作用。事实上，最先进的方法相对于PointNet++的性能提升很大一部分是由于改进了训练策略。例如：在训练阶段**随机去除颜色(randomly dropping colors)**，会使PointNet++的测试性能提高5.9%的mIoU；采用**标签平滑(label smoothing)**可以将ScanObjectNN的总体准确度提高1.3%。这些发现启发作者重新审视PointNet++，并为其配备当今广泛使用的、新的高级训练策略。如下图所示，在ScanObjectNN上，仅使用改进的训练策略就可以将PointNet++的OA提高8.2%（从77.9%提高到86.1%），从而在不改变架构的情况下建立新的SOTA；对于S3DIS分割任务，通过6倍交叉验证在所有领域评估的mloU可以增加13.6%（从54.5%增加到68.1%），优于PointNet++之后的许多现代架构，如PointCNN和DeepGCN。

![](1.jpg)

此外，当前流行的点云分析模型使用的参数比原始点网多得多。有效地将PointNet++从原来的小规模扩展到更大的规模，通常更大的模型能够实现更丰富的表示和更好的性能。然而，在PointNet++中使用更多构建块或增加通道大小的天真方法只会导致延迟开销，而准确性没有显著提高。为了有效地扩展模型，作者将**残差连接**、**inverted bottleneck**和**可分离的MLP**引入PointNet++中。这种现代化的体系结构被命名为PointNeXt。PointNeXt可以灵活扩展，在各种基准上都优于SOTA。PointNeXt在S3DIS上将原来的PointNet++提高了20.4%的mloU（从54.5%提高到74.9%），是原来的6倍，在ScanobjecNN上实现了9.8%的OA增益，超过了SOTA Point Transformer和PointMLP。

贡献：

- 本文是首次对点云域中的训练策略进行系统研究的论文。结果表明，仅采用改进的训练策略，PointNet++就能扳回一局。改进后的训练策略具有通用性，可以方便地应用于其他方法。
- 提出了PointNeXt。

### Preliminary: A Review of PointNet++

PointNeXt基于PointNet++，它使用类似于**U-Net**的架构，带有编码器和解码器，如下图所示。编码器部分使用许多**set abstraction块**分层提取点云的特征，而解码器通过相同数量的特征传播块逐步**插值**提取到的特征。SA块包括一个用于对输入点进行下采样的下采样层、一个用于查询每个点的邻居的分组层、**一组用于提取特征的MLP(将PointNet换成MLP)**以及一个用于聚合邻域特征的**reduction layer**。

![](2.jpg)

### Methodology: From PointNet++ to PointNeXt

本文聚焦于两件事：

- 训练上的改进：改进**数据增强**和**优化**技术；

  - Data Augmentation

    最原始的PointNet++对各种基准测试使用了随机旋转、缩放、平移和抖动等数据增强的简单组合。最近的方法采用了比PointNet++中使用的方法更有力的**数据增强技术**。

  - Optimization Techniques

    优化技术包括损失函数、优化器、学习率和超参数，对神经网络的性能也至关重要。由于机器学习理论的发展，现代神经网络可以使用理论上更好的优化器（如AdamW）和更好的损失函数（带有标签平滑的交叉熵）进行训练。作者团队还量化了每种现代优化技术对PointNet++的影响。首先对学习率和权重衰减进行顺序超参数搜索。然后，对标签平滑、优化器和学习速率进行加性研究。我们发现了**一组改进的优化技术**，可以进一步提高性能，提升幅度相当可观。标签平滑、AdamW和余弦衰减的交叉熵通常可以提高各种任务的性能。

- 架构上的改进：感受野缩放和模型缩放

  - Receptive Field Scaling

    在CNN中，感受野是十分重要的因素。在点云处理中，至少有两种方法可以缩放感受野：（1）采用更大的半径查询邻域；（2）采用层次结构。由于原始的PointNet++采用了层次结构，因此本文主要研究**更大的半径查询邻域**。

  - Model Scaling

    PointNet++是一个相对较小的网络，其中编码器仅由分类结构中的2个阶段和语义分割结构中的4个阶段组成。每个阶段仅由1个**SA块**组成，每个块包含3层MLP。用于分类和分割的PointNet++模型大小都小于2M，这比通常使用超过10M参数的现代网络要小得多。作者发现，添加更多**SA块**或使用更多通道都**不会**显著提高精度，同时导致吞吐量显著下降，主要原因是梯度消失和过拟合。

    本文提出在每一个SA之后加一个InvResMLP，以实现有效的模型缩放。InvResMLP和SA之间有三个区别：

    - 在输入和输出之间添加一个残差连接，以解决梯度消失问题。
    - 为了减少计算量和加强逐点特征提取，引入了可分离的MLP。虽然原始SA块中的所有3层MLP都是基于邻域特征计算的，但**InvResMLP**将MLP分为一个用于计算邻域特征的单一层（在grouping layer和reduction layer之间）和两个用于点特征的层（在reduction之后），这是受ASSANet和ConvNeXt的启发。
    - 利用**inverted bottleneck**将第二个MLP的输出通道扩展了4倍，以提取丰富的特征。事实证明，与附加原始SA块相比，附加**InvResMLP**块可以显著提高性能。

除了**InvResMLP**之外，我们还介绍了在宏观体系结构中的三个变化。

- 统一了用于分类和分割的PointNet++编码器的设计，即将用于分类的SA块的数量从2个缩放到4个，同时保持分割的SA块不变(4个)。
-  我们使用了一个对称解码器，其中它的信道大小被改变以匹配编码器。 
- 我们添加了一个**stem MLP**，即在体系结构开始处插入的附加MLP层，以将输入点云映射到更高的维度。

```bash
nohup CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml >> /root/PointNeXt/out.log 2>&1 &
```

