### Point Cloud Saliency Detection by Local and Global Feature Fusion(2019)

### Abstract

本文提出一种新方法，用于检测3D点云上突出区域。首先，根据局部环境的差异来评估每个点的局部差异；其次，将点云分解为`小集群(small clusters)`，并计算每个集群的初始全局`稀有度值(rarity value)`；然后使用`random walk ranking method`将所有集群中的每个点引入集群级全局`rarity refinement`；最后，提出了一个优化框架，将局部差异值和全局稀有度值集成在一起，以获得点云的最终显著性检测结果。

**本文的中心思想：local distinctness + global rarity。**

### Conclusion

本文提出了一种新的点云`显著性`检测方法，该方法综合利用心理学证据支持的局部显著性和全局稀有性线索。本文考虑了局部几何特征，相较于超体素分割，本文能减少不精确分割对全局稀有度计算的影响。此外，还提出了一种`自适应`优化框架，通过考虑不同显著性线索之间的内在关系，可以有效地**组合**不同的显著性线索，并获得最先进的显著性检测结果。

### Introduction

> Saliency detection for 3D point clouds can be regarded as finding for perceptually important regions that are unique with respect to their surrounding regions

本文将显著性检测应用于非结构化的3D点云数据中。

### Method

本文使用的方法都是传统方法：FPFH、KNN、supervoxel、random walk ranking method。