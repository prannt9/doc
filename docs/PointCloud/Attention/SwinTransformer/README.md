### Swin Transformer(2021)

> 论文地址：https://arxiv.org/pdf/2103.14030.pdf

**本文是ICCV 2021的年度最佳论文。**

### Abstract

本文提出一种新的**Vision Transformer**，叫做**Swin Transformer**，可以作为CV领域**通用**的骨干网络。但是把**Transformer**用于CV领域是有巨大挑战的。挑战主要来源于两个方面：**一个是**尺度上的问题(如给你一张街景的图片，里面有很多车辆和行人，而行人和车辆的尺寸不同，这种现象在NLP中不存在)；**另一个**是图像的分辨率太大。为了解决这两个问题，本文提出**hierarchical Transformer**，通过移动窗口的形式学习特征，移动窗口的好处是效率提升了(因为自注意力是在窗口内算的，所以序列长度大大降低；通过移动操作能让相邻的窗口之间有了交互，所以上下层之间就有了**cross-window connection**，从而变相达到了全局建模的能力)。这种层次化结构不仅非常灵活，可以提供各个尺度的特征信息，同时自注意力是在滑动窗口内做的，所以计算复杂度是随着图像大小线性增长的(而非平方级增长)。

### Introduction

第一段：**CNN**长久以来都在CV领域占据着主导地位。

第二段：NLP领域的发展却走了一条不同的路线，**Transformer**如今占据主导地位。Transformer在NLP领域的成功也在驱动研究者们将其应用到视觉领域，最近Transformer在视觉任务(主要是classification)上也表现出了很好的效果。

第三段：本文探究了Transformer在视觉任务上的应用并将其作为CV的通用模型，**就像其在NLP中的表现和CNN在vision中的表现一样**。将Transformer从NLP领域迁移至视觉领域有**两个**困难(Abstract中提到的)。诸如图像分割、目标检测等任务需要在像素级别做出密集的预测，但Transformer无法胜任，因为平方倍的计算复杂度。为了解决这一问题，本文提出了高分辨率通用的Transformer骨架网络，**Swin Transformer**，由**层次化**的**特征图**构成，并且拥有**线性时间复杂度**。如下图所示，Swin Transformer从较小的patches开始，随着层数的加深，逐渐融合邻近的patches。有了这些层次化的特征图，Swin Transformer能够轻易的使用诸如`FPN`网络进行目标检测任务，或`U-Net`进行语义分割任务。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/20.png)

上图中，ViT把图片打成patch(ViT中使用的patch size是16 * 16的)，图中的16x表示16倍的下采样率。这也就意味着，在ViT中每一个patch(或token)自始至终尺寸都是一样的，每一层的Transformer Block所看到的token的尺寸都是16倍的下采样率。虽然ViT可以通过全局的Self-Attention操作达到全局的建模能力，但对多尺寸特征的把握能力就会弱很多。在ViT里，它处理的特征都是单一尺寸，而且是低分辨率的，所以不适合处理密集预测型的任务。同时，ViT的Self-Attention始终都是在最大的窗口(整张图)上进行，所以ViT是全局建模，时间复杂度是平方倍的增长。

Swin Transformer(简称ST)借鉴了很多CNN的设计理念以及先验知识，如为了减少序列长度，**ST**选择在小窗口内算Self-Attention(而不是像ViT一样，在整张图上算)，这样一来，只要窗口的尺寸固定，Self-Attention的计算复杂度就固定。整张图的计算复杂度会和图片大小呈线性复杂度。如上所述，是利用了CNN中locality的inductive bias(同一个物体的不同部位或者语义相近的不同物体大概率会出现在相邻的地方)。如何生成多尺寸的特征呢？可以回想CNN，CNN为什么会有多尺寸的特征呢？因为有**pooling**操作，可以增大每个卷积核的感受野，从而使得每次池化后的特征抓住物体的不同尺寸。**ST**在这里也提出了类似于pooling的操作，叫作patch merging(把相邻的小patch合成一个大patch，这一个大patch就能看到之前几个小patch看到的内容，感受野就增大了，同时也能抓住多尺寸的特征)。如上做图所示，ST刚开始的下采样率是4倍，然后变为8倍、16倍，一旦有了这种多尺寸的信息，就可以把这些多尺寸的特征图输入给FPN去做检测任务，或输入给U-Net去做语义分割任务。

第四段：**ST**的一个关键设计是**shift window(如下图所示)**， 在第L层中，把特征图分为如下左图的四个小窗口，每个窗口由基本的patch构成，每个patch的尺寸为4*4，红色的框是一个window，在本文中，一个window里有7x7=49个patch。在第L+1层，就是把第L层的整体特征图向右下角移动2个patch。右图中window与window之间可以互动(cross-window connection)，因为在左图(没有shift)中window和window之间都是不重叠的，如果每次Self-Attention操作都在左图的window里进行，那么该window里的patch就无法注意到别的window里的patch。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/21.png)

第五段：卖结果。

第六段：展望未来。

### Conclusion

提出了Swin Transformer，能够产生层次化的特征图，复杂度和输入图片的尺寸呈线性增长。本文最重要的贡献是基于Self-Attention的shift window。接下来的工作就是把Swin transformer用到NLP里。

### Method

#### Overall Architecture

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/22.png)

- 假设输入图片的尺寸为H x W x 3 = 224 x 224 x 3，像ViT一样经过Patch Partition把图片打成patch(patch size = 4 x 4)，尺寸为56 x 56 x 48(56 = 224/4，48 = 4x4x3)；

- 经过Linear Embedding层，把patch转换为向量形式，尺寸变为56 x 56 x 96(C = 96，token为每个向量的维度)，56 x 56会被拉直(变为序列长度)，所以尺寸为3136 x 96；

- **注：**Patch Partition + Linear Embedding相当于ViT中Linear Projection那一步操作，在代码里也是使用一次卷积操作就完成了；

- 目前的序列长度已经达到了3136，目前的Transformer是无法接受这么长的序列长度的，所以**ST**引入了基于窗口的Self-Attention，每个窗口都有7x7=49个patch，所以序列长度变为8*8=64；

- 经过ST Block，输出为56 x 56 x 96

- 想要有多尺寸的特征信息，就要构建多层次的Transformer，需要像CNN中的pooling一样，本文提出了**patch merging**；

  - patch merging的过程如下图所示：假设有个张量(里面的数字表示序号)，下采样的时候每**隔**一个点选一个(stride=2)，原来的一个张量就变为4个张量，然后沿着C这个维度拼接这4个张量，尺寸变化为：H x W x C -> H/2 x W/2 x C x 4(用空间尺寸换通道数)，CNN中的池化操作之后，通道数都是翻倍，而不是变为原来的4倍，所以在C这个维度上使用1x1卷积，把通道数将为2C，最终的尺寸变化为：H x W x C -> H/2 x W/2 x C x 2(空间尺寸减半，通道数翻倍)。在经历了第一次patch merging之后，尺寸由原来的56 x 56 x 96变为28 x 28 x 192。

    ![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/23.png)

- 经历第二次ST Block，输出依旧为28 x 28 x 192；

- 第三阶段同理，输出为14 x 14 x 384；

- 第四阶段同理，输出为7 x 7 x 768；

- 在得到最后一层特征图(7 x 7 x 768)之后，做了一次全局池化操作(global average pooling)，尺寸变为1 x 1 x 768

#### Shifted Window based Self-Attention

**窗口划分**

56 x 56 x 96 -> 8 x 8 x 96(整张图共有64个窗口)，在64个窗口中分别算注意力。 

**移动窗口**

Transformer Block的安排是有讲究的：每次都先做一次基于窗口的Self-Attention，再做一次基于移动窗口的Self-Attention，这样就达到了窗口和窗口之间互相通信的目的。如Overall Architecture的图b所示：输入进来之后，先做一次Layer Normalization，然后做基于窗口的Self-Attention，然后再做一次Layer Normalization，经过MLP。本次的窗口就结束了。接下来输入到下一次的**shift window**中，得到最终输出。两个Block加起来才算是Swin Transformer的一个基本计算单元。

如第二张图所示，原始的图像有四个window，而shift之后得到了9个窗口，窗口的数量增加了，且每个窗口里的patch数量大小不一。在这种情况下，如果想做快速运算，把这些窗口全部压成一个batch直接去算Self-Attention就不行了(因为窗口尺寸不一样)。

解决方法有两个：

- 在小窗口周围填充0(padding操作)，使得所有的window尺寸一样，这样就还能把他们压成一个batch，运算就会快很多。但是这样一来，计算复杂度无形之中就提高了，因为原来计算基于窗口的Self-Attention，只需要算4个窗口的，但是shift之后需要算9个窗口的，复杂度提升了2倍多；
- 怎样才能让shift之后的窗口数量还是4个，且每个窗口里的patch数量还保持一致呢？作者在这里提出了一种非常巧妙的masking(掩码)方式，如下图所示。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/24.jpeg)

当通过shift得到9个大小不一的window时，不在这9个window上去算Self-Attention，先做一次循环移位(cyclic shift)，具体做法是：把A和C移到正下方，再把B和A移动到右边(七巧板)。这样一来，就得到了四个尺寸相同的window。但是现在新的问题就来了，左上角的window里的元素是紧挨着的，元素之间可以两两做自注意力，但是剩下的三个窗口就不一样了，它们里面的元素是从别的地方搬过来的，所以元素之间按道理来说，是不应该做自注意力的(它们之间不应该有什么联系，比如左下角下方元素和上方元素)。为了解决这一问题，作者设计了几种掩码方式，从而能让一个窗口之中不同的区域之间也能用一次前向过程就能把自注意力算出来，并且互相之间不干扰。就是第三个小图中的masked MSA(具体过程下面细说)，最后还需要把循环移位还原回去(为了保持图片的相对位置不变，整体图片的语义信息也是不变的)。

**掩码操作**

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/26.png)

如上图所示，是已经经过循环移位之后的图像，左上的四个格子是原来的图，其他三部分是移过来的。区域0中的元素都是相邻的，是可以相互之间做自注意力的，但是对于区域1而言，左边是原图，右边是从其他地方移过来的，因此1和2之间不应该做自注意力。同理，3和6、4和5和7和8都不应该做自注意力。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/27.png)

如上图所示，假设原始的图像中每个window包含7 x 7个patch，仅考虑3号和6号区域。分别将3号和6号区域拉直成向量，由于shift window的步距为为窗口大小的一半(7 / 2)，则3号区域被拉直后尺寸为4 x 7 = 28，6号区域被拉直后尺寸为3 x 7 = 21。接下来把左边的向量转置得到中间的向量，做自注意力(左边乘以中间)，得到右边的图。对于右边的图，右上和左下是不希望得到的，因此会被mask掉，保留左上和右下的区域。现在已经知道了哪些区域想要，哪些区域不想要，作者针对这个形式设计了一个掩码的模板，如下图所示。

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/28.png)	 

然后让上面两张图相加，由于36、63那个矩阵和上图的矩阵相加之后，右上和左下会是负数，然后通过softmax操作之后就变为0了。