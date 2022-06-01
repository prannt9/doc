## PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation(2017)

> 论文地址:https://arxiv.org/pdf/1612.00593.pdf
>
> 代码地址:https://github.com/fxia22/pointnet.pytorch
>
> 作者汇报:https://www.bilibili.com/video/av29834089/

![Application of PointNet](https://cdn.jsdelivr.net/gh/prannt99/blog/img/WeChatc545d1aa5b26183c9c708617062eb46f.png)

### Abstract

本文设计了一种神经网络结构,直接处理三维点云(而非投影或变换),可实现`物体分类`、`部分分割`、`语义分割`.PointNet的基本思想:学习每个点的空间编码，然后将所有`单个点`特征聚合到全局点云上。通过这种设计，PointNet不会捕获局部结构。然而，事实证明，利用局部结构对卷积架构的成功很重要。

### Introduction

典型的神经网络架构需要`高度规则`的输入数据格式,但是点云和`mesh`两种数据结构是无序的.大多数研究在把输入送进网络前,会先把点云转换为体素网格,这样做会带来很多问题.`PointNet`直接将点云作为输入(x, y, z),但是进入网络之前会先进行数据对齐(T-Net).网络在初始阶段以`相同和独立的方式`处理每个点.输出可以是类标签也可以是语义标签.`PointNet`能够根据一组稀疏的关键点来概括输入点云.

方法的关键:对称函数 + 最大池化.

### Related Work

3D数据有许多流行的表示,所以产生了各种各样的(基于学习的)方法.有基于体积的、基于多视图的、

定义:从数据结构的角度来看,点云是无序集合组成的向量.

> We design a deep learning framework that directly consumes unordered point sets as inputs. A point cloud is represented as a set of 3D points {Pi | i = 1, ..., n}, where each point Pi is a vector of its (x, y, z) coordinate plus extra feature channels such as color, normal etc. For simplicity and clarity, unless otherwise noted, we only use the (x, y, z) coordinate as our point’s channels.

### Architecture

![Arichitecture](https://cdn.jsdelivr.net/gh/prannt99/blog/img/WeChatc5c2d29c070d4bc63ccb45a577c82c26.png)

从架构图中可以看出:`分类网络`和`分割网络`有很大一部分是一样的.网络由三个关键的模块组成,分别是`max pooling layer`、`local and global information aggregation`、`two joint alignment networks`.

1. **Max Pooling Layer**:作为对称函数来聚合所有点的信息;

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/WeChat03188f4c7b30a08c6faea5ae938e76d4.png)

上图显示了各种方法作为对称函数的精度.

1. **local and global information aggregation**:1024 + 64 = 1088(维) 
2. **joint alignment networks**:T-Net对齐网络

接下来是理论分析:主要说明了一件事,那就是对输入点点小扰动不应该对输出造成很大影响.

### Experiment

实验由四个部分组成:

1.PointNet进行`分类`、`部分分割`、`语义分割`

- **分类实验**
  - 数据集:`ModelNet40`,测试集:训练集 = 9843:2468,均匀采样1024个点.
  - 数据增强:旋转、抖动
- **部分分割实验**
  - 数据集:`ShapeNet`
- **语义分割实验**
  - 数据集:`S3DIS`
  - 首先根据房间拆分point,然后把每一个房间划分为`1x1`大小的区域(block)
  - 在训练阶段,从每个`block`中随机采样`4096`个点;在测试阶段,在所有的点上进行测试
  - 使用`K`折交叉验证

2.结构设计分析

- 消融实验:加不加`T-Net`对结果的影响并不大
- 鲁棒性测试:损失50%的点,精确度仅下降2.4%

3.可视化PointNet(略)

4.时间、空间复杂度分析











