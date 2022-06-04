### Generative Adversarial Nets

> 论文地址：[传送门](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)

<center><strong>没错， 就是干!</strong></center>

GAN(生成对抗网络)，全篇的写作都值得效仿!当然摘要也值得效仿！

### Abstract

本文提出了一个新型的`framework`以`对抗`的方式`估计`生成模型，在该过程中会同时训练两个模型： 一个是用于捕捉数据分布的生成模型`G`， 另一个是辨别模型`D`， 用于估计一个样本是从训练数据中来，还是生成出来的数据(`G`中的数据)。 `G`的作用是最大可能的让`D`犯错。在任何函数空间中`G`和`D`中，存在唯一解，使得`G`能够找出真是数据中的分布， 而`D`始终等于`0.5`， 如果`G`和`D`被定义为`MLP`的话， 整个系统可以通过反向传播来训练。本论文不使用任何`马尔可夫链`或展开一个近似的`推理网络`(说白了就是**我比别人简单**)。

> 评论: 教科书📗式的摘要.

### Introduction

深度学习的作用是发现`丰富`的、`层次化`的模型，这些模型能够表示`AI应用`中各种数据的概率分布。

> 评论：深度学习不光是讲`深度神经网络`，更多的是对整个数据分布的特征表示。

迄今为止，深度学习领域显著的成功集中于`辨别模型`，这些成功都依赖于反向传播和`dropout`算法。而深度`生成模型`做的还比较差，因为很难近似`最大似然估计`和相关策略中出现的概率计算，并且很难在生成环境中利用分段线性单元的优势。

> 评论：这一段实际上在说，不去近似`似然函数`了，可以使用别动方法来：得到计算上更好的模型。

在`对抗网络`架构中，有两个模型：一个是`生成模型`，一个是`判别模型`。 `生成模型`是`造假币`的人， `判别模型`是`警察👮‍♀️`， `判别模型`的任务是把假币找出来， 和真币区分开来. 在这一过程中， `造假者`和`警察`都会不断学习， 最后希望的是`造假者能赢`(这时生成出来的数据就是真是数据)。

> 评论：造假者和警察互相内卷。目标：不断的调整G和D，直到D不能把事件区分出来为止，在该过程中需要：
>
> - 优化G，使它尽可能的让D混淆；
> - 优化D，使它尽可能的能区分出假冒的东西。

在`GNN`中， 生成模型是一个MLP， 输入是随机噪声， 且判别模型也是一个MLP， 这种特例被称为`对抗网络`. 由于2个模型都是MLP， 所以在训练时可以通过`反向传播`和`dropout`完成(不需要近似推理或马尔可夫链)。

### Related Work

有个小剧场。

### Adversarial Nets

![](https://cdn.jsdelivr.net/gh/prannt99/blog/img/20220530231342.png)

上面四张图a，b，c，d. 黑色的点状线代表真实数据的分布，绿色线表示生成模型`G`，蓝色线代表分类模型`D`。

- a图表示初始状态；
- b图表示保持`G`不动，优化`D`，直到分类的准确率最高；
- c图表示保持`D`不动，优化`G`，直到混淆程度最高；
- d图表示多次迭代后，终于使得`G`能够完全产生`D`无法分辨的数据，此时`D = 0.5`。

#### 公式化

```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async>
</script>
</head>
<body>
<p>
\min_{G} \max_{D}V(D,G) = E_{x\sim p_{data}(x)}[logD(x)] + E_{z\sim p_{z}(z)}[log(1-D(G(z)))]
</p>
</body>
</html>
```

同时训练`G`和`D`，使得混淆程度变高、辨别能力增强。

分析：优化`D` -> logD(x) ↑ -> V(D，G) ↑；优化`G` -> G(z) ↑ -> D(G(z)) ↑ -> log(1 - D(G(z))) ↓(自己的理解)

以下是数学推导过程：式中G(z)是`fake data`。在训练`D`的时候，`G`是固定的。

```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async>
</script>
</head>
<body>
<p>
target:\max_{D}V(D,G) = E_{x\sim p_{data}(x)}[logD(x)] + E_{z\sim p_{z}(z)}[log(1-D(G(z)))]\\=\int_{x}p_{data}(x)·logD(x) + \int_{x} p_{g}(x)·log(1-D(x))dx\\=\int_{x}[p_{data}(x)·logD(x)+p_{g}(x)·log(1-D(x))]dx\\对于f(g)=a·log(g)+b·log(1-g),有:\\ \nabla f = a·\frac{1}{g}-b·\frac{1}{1-g}=\frac{a-(a+b)·g}{g(1-g)}\\ 当且仅当g=\frac{a}{a+b}时，\nabla f=0，f(g)可取到最大值,所以当D训练的足够好时，有：\\ D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}，使得V(D,G)最大
</p>
</body>
</html>
```

同时训练`G`和`D`，使得混淆程度变高、辨别能力增强。

分析：优化`D` -> logD(x) ↑ -> V(D，G) ↑；优化`G` -> G(z) ↑ -> D(G(z)) ↑ -> log(1 - D(G(z))) ↓(自己的理解)

以下是数学推导过程：式中G(z)是`fake data`。在训练`D`的时候，`G`是固定的。

对于有当且仅当时，，可取到最大值所以当训练的足够好时，有：，使得最大

训练`G`的时候情况类似。

GANs是通过最小化`JS散度`去拟合两个分布。 当p<sub>g</sub>(x) = p<sub>data</sub>(x)时，V(G, D<sup>*</sup>)取最小值`-log4`，此时`D = 0.5`

### 思考

在应用上，是否可以使用GAN来生成点云?或者代替一些数据增强的方法.



训练`G`的时候情况类似。

GANs是通过最小化`JS散度`去拟合两个分布。 当p<sub>g</sub>(x) = p<sub>data</sub>(x)时，V(G, D<sup>*</sup>)取最小值`-log4`，此时`D = 0.5`

### 思考

在应用上，是否可以使用GAN来生成点云?或者代替一些数据增强的方法.











