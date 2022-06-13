### **Contrastive Boundary Learning for Point Cloud Segmentation (CVPR 2022)**

> 论文地址：[传送门](https://arxiv.org/pdf/2203.05272.pdf)
>
> 代码地址：[传送门](https://github.com/LiyaoTang/contrastBoundary)

### Abstract

 3D点云的分割方法在场景边界的表现不佳，因此拖累了整体分割精度。 

> However, current 3D point cloud segmentation methods usually perform poorly on scene boundaries, which degenerates the overall segmentation performance.

本文的重点是对场景边界进行分割，为了解决在边界上表现不好的问题，本文提出`CBL`框架。具体来说，`CBL`通过对比多尺度场景上下文，增强了边界点之间的特征区分。

### Introduction

背景：

> - Despite that various point cloud segmentation methods have been developed, little attention has been put on boundaries in 3D point clouds.
> - a clean boundary estimation can be beneficial for overall segmentation performance.
> - 当前流行的分割指标缺乏对边界性能的具体评估，这就使得现有方法无法更好的展示边界分割的质量。

贡献：

- 探索了当前3D点云分割中存在的边界问题，并考虑边界区域的指标对其进行量化。**结果表明，目前的方法在边界区域的准确性比它们的整体性能差得多。**
- 提出了一种新颖的对比边界学习（CBL）框架，它通过对比场景边界上的点特征来改进特征表示。因此，它提高了边界区域周围的分割性能，从而提高了整体性能。
- 通过全面的实验证明CBL可以在边界区域以及所有baseline的整体性能上带来显着且一致的改进。这些实验结果进一步证明了CBL对提高边界分割性能是有效的，准确的边界分割对于鲁棒的3D分割很重要。

核心思想：

- 边界分割
  - 由于当前大部分工作都集中在改进一般指标，例如mIoU、OA和mAP，点云分割中的边界质量通常被忽略。最近的边界相关工作仅给出边界的定性结果；
  - 本文作者引入了一系列用于量化边界分割质量的指标，包括`mIoU@boundary`、`mIoU@inner`和来自2D实例分割任务的边界`IoU(B-IoU)`；
  - 基于ground-truth数据，如果在其邻域中存在具有`不同注释标签`的点(作者将其视为边界点)。类似地，对于模型预测，如果附近存在具有不同预测标签的点，也将这个点视为边界点。这里将点云记为X，第i个点记为xi，其局部邻域为Ni=N(xi)，对应的GT标签为li，模型预测标签为pi。将GT中的边界点集定义为Bl，预测分割中的边界点集定义为Bp。
  - 为了检查边界分割结果，一种直观的方法是计算边界区域内的mIoU，即mIoU@boundary。为了进一步比较模型在边界和非边界（内部）区域的性能，作者进一步计算了内部区域的mIoU，即mIoU@inner。
  - 然而，`mIoU@boundary`和`mIoU@inner`没有考虑模型预测分割中的错误边界。受2D实例分割的边界IoU的启发，为了更好地评估，作者考虑了分割预测中的边界与GT数据中的边界之间的对齐。通过`B-IoU`进行评估。
- CBL框架
  - 对比边界学习（CBL）框架通过`对比学习`来增强跨边界的特征的`辨别能力`。然后，为了深入提高模型在边界上的性能，作者通过子场景边界挖掘在下采样点云（即子场景）中使用CBL。