

## week3 作业

### Coding

1. 【必做】使用numpy实现Softmax（假设2个样本，给定[[1, 2, 3], [2，1，3]]);
2. 【必做】使用torch.nn.functional.softmax() 验证numpy实现Softmax是否一致；
3. 【选做】了解目标检测xml的标注文件，提取目标框和图片长宽，把图像上的物体框出来。

使用浏览器打开文件【test_00000335.xml】

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701104808779.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)



### 自行学习

++++++++++++++++ **以下不是作业，是额外的一些好东西，不用提交，自行学习** ++++++++++++++++ 

#### 一.PyTorch 中的亦混 loss

Pytorch 中提供了很多类型的 loss (Criterion)，如 l1, l2, crossentropy 等。同时，同一种 criterion 可 能被包含在不同的功能模块下，如 torch.nn，torch.nn.functional 等。

这里，希望同学们能够区分和辨别其中的差异，以及他们之前相互的关联，不至以后感觉模糊。

1. 明确 F.xx 和 nn.xx (xx 表示某 criterion 如 cross_entropy 等)在使用时的区别

2. 明确各常用 F.xx 以及 nn.xx 的具体内容。可以从传入参数、数学形式、作用目的等不同方

   面总结。
    常用 criterion 需包括(但不限于[欢迎更多的总结]): F.softmax, F.cross_entropy, F.log_softmax, F.nll_loss, nn.CrossEntropyLoss, nn.NLLLoss, nn.LogSoftmax 等

#### 二.一些 criterion 的数学

 为进一步强化某些格外重要的 criterion 的理解，现补充一些数学求导，对面试很有帮助:

1. (数学意义上的)softmax 的求导。

2. (数学意义上的)cross_entropy+softmax 的求导。请注意，需包含 cross_entropy+softmax 的

   原始形式，以及求导后的结果。另外，请注意，下标的正确对应以及分类讨论, 以及分类 讨论后的整合。