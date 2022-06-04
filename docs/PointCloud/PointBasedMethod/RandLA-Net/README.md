### RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds(2020)

> 论文地址:[传送门](https://arxiv.org/pdf/1911.11236.pdf)
>
> 代码地址:[传送门](https://github.com/QingyongHu/RandLA-Net)

### Abstract

本问的关键是使用`随机点采样`，而`不是更复杂的点选择方法`。虽然计算和内存效率显著提高，但`随机采样`可能会丢弃关键点。为了克服这个问题，引入了一个`局部特征聚合模块(local feature aggregation module)`，以逐步增加每个点的感受野，从而有效地保留几何细节。

### Introduction

深度传感器获得的原始点云通常是`不规则采样`、`非结构化`和`无序`的。`PointNet`是使用`MLP`学习每个点的特征，如该没有块分区(类似的预处理步骤)，很难扩展至大规模点云中。这种限制的原因有三个方面。1）这些网络采用的下采样方法要么计算昂贵，要么内存效率低下。例如，广泛使用的`最远点采样(FPS)`需要200多秒才能取样100万点中的10%。2）大多数现有的局部特征学习器通常依赖于昂贵的`内核化`或`图形构建`，因此无法处理大量点。3）对于通常由数百个对象组成的大型点云，现有的局部特征学习器要么无法捕获复杂的结构，要么由于感受野有限而效率低下。

本文首先采用`RS + local feature aggregation`的组合进行下采样(还能保证关键信息不丢失)。

过程：

- 对于每个点，首先采用`LocSE`单元显式地保存`局部几何结构`；
  - Finding Neighbouring Points
  - Relative Point Position Encoding
  - Point Feature Augmentation
- 然后使用`attentative pooling`层自动保留有效的局部特征；
- 最后把`LocSE`和`attentive pooling`堆叠在一起作为`dialated residual block`来增大每个点的感受野；
- 这些组件都是用`shared MLPs`实现的。

### Related Work

从点云中提取特征的方法：

- 传统方法：手工提取特征
- 集于学习的方法：
  - Projection
  - Voexl
  - Point
    - neighboring feature pooling；
    - graph message passing；
    - kernel-based convolution；
    - attention-based aggregation。

### RandLA-Net

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/12.png)

如上图所示，从密集的大规模点云下采样到稀疏的点云，先经过`FPS`，再经过`local feature aggregator`保留图出的特征。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/13.png)

`RandLA-Net`网络结构如上图所示，由`LocSE` + `Attentive Pooling`组成。

#### LocSE

给定一个点云`P`以及每个点的特征，`LocSE`会显式地嵌入所有邻域点的空间坐标，这样一来，相应的点特征总是会知道他们的相对空间位置。这使得`LocSE`显式地察觉到局部几何模式，整个网络也都能够学习复杂的局部结构。`LocSE`的输出是一组新的邻域特征，其显式地编码中心点`Pi`的局部几何结构，以增强相邻的点特征。

#### Attentive Pooling

作用：用于聚合邻近点的特征。

#### Dialated Residual Block

由于点云将被大幅下采样，因此最好显著增加每个点的`感受野`，这样即使一些点被丢弃，输入点云的几何细节也更有可能被保留。使用`跳连接`堆叠多个`LocSE`和`Attentive Pooling`作为`Dialated Residual Block`。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/14.png)

如上图所示，第一次`LA`之后，可以观察到红色点的`K`个邻居，第二次`LA`之后观察到`K·K`个邻居，为了防止过拟合，本文只堆叠两个`LA`。

流程：

- 输入(N，3 + d) -> Shared MLP(N，d<sup>out</sup> / 2) -> 3D coordinates -> LocSE(N，d<sup>out</sup>)
  - shared MLP将d<sup>in</sup>变为d<sup>out</sup> / 2
  - 然后进行`3d`坐标嵌入
  - LocSE
    - (N，3 + d) -> (1, 3 + d) -> KNN -> (K, 3 + d)
      1. -> (k，d)(仅取K个点的特征)
      2. -> (K, 3)(仅取K个点的三维坐标) -> Positional Encoding -> (K，d)
      3. concatenate(1，2)
- -> Attentive Pooling(N，d<sup>out</sup> / 2)
  - 上一步concatenate得到(K，2d) -> 点乘size也为(K，2d)的Attention Scores -> 得到Attention Features(K，2d) -> 权重求和 -> Shared MLP -> Aggregated Feature(1，d<sup>out</sup>)
    - Attention Scores是使用`g()`函数计算`上一步concatenate得到(K，2d)`得到的，`g()`本质上就是`shared MLP` + `softmax`
- -> 3D coordinates -> LocSE(N，d<sup>out</sup>) + Attentive Pooling(N，d<sup>out</sup>) -> Shared MLP(N，2d<sup>out</sup>)
- -> + Input Point Features -> Leaky Relu 
- -> Aggregated Features(N，2d<sup>out</sup>)



