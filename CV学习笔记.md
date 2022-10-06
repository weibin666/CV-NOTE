



### 1、RCNN算法解析

![RCNN](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_19.jpg)

> ```python
> # 听课笔记：
> @）、传统机器学习方法：input-->feature--->classic ML tools---->target
> @）、深度学习方法：input--->CNN---->target
> 1、RCNN流程：
> input image-->extract region proposals(~2K大概就是2000个)-->compute CNN features-->classify regions(Classification+Detection)   
> A1.Region Proposal：Selective Search
>    通过Selective Search方法找到近似2000个region proposal（同时要wraped to fixed size）
> A2.Feature Extraction
>    将这2000个RP每一个(注意是每一个这也是网络慢的原因)都放进CNN网络中进行提取特征feature
>    Train Tips:
>     a.Pretrain on ILSVRC2012 dataset
>     b.Finetune(微调) on real dataset
>     c.Batch size:128=32pos(正样本)+96neg(负样本)(background)[iou<0.5][1:3 is classic]
> A3.Classification+Detection
> 	SVM Classification + NMS + BBox Regression
> 	a. SVM Classification
>         。Do classification for each class
>         。
>     	  Pos:Ground Truth   【Different from CNN part】
>           Neg: iou < 0.3[others ignored]
> 	b.  Non-Maximum Suppression(NMS)
>     	。sort bbox according to score[here class score]
>         。get the 1st one,remove those which has higher iou with it[会通过一个阈值去除掉它]
>         。then repeat the last step until there‘s no bbox.
> 	c. Bbox regression
>     	。regression targets of(dx,dy,dw,dh)
>         。coordinates need to be normalized 
>         。we’ll discuss in detail later
> Q:Why iou threshold is different here comparing to training AlexNet?
>     we need more data when training CNN，but here more data may be harmful.
> Q:Why use SVM?AlexNet has softmax at last,why not use directly?
>     AlexNet focuses on classification，Not accurate on localization
>     Use hard negative mining when using SVM
>     # Fine Tuning时与GT的iou最大且大于0.5的候选框为正样本。其余候选框为负样本，训练各类别SVM分类器时，GT框为该类正样本，与GT的iou小于0.3的候选框为该类负样本，忽略与GT的iou大于0.3的候选框。
> 2、NMS 
> Non-Maximum Suppression
> 
> ```
>

#### RCNN的优缺点

1. slow：need to run full forward pass if CNN for each region proposal
2. SVMs and regressors are post-hoc:CNN features not updated in response to SVMs and regressors.
3. Complex multistage training pipeline

### 2、NMS的原理与发展

```python
# NMS听课笔记
1、Traditional NMS[here we use.Need to know how to code in C++/Python]--most important
2、Soft NMS[2017,Better to know]--more important
3、IOU-Net[2018,Locatization Confidence + PreROI Pooling,better to know]
4、Soft-NMS[2019,kl-loss+soft-NMS,better to know]
# 传统NMS
通过score挑出最大的那个bounding box，然后进行循环分别和剩余的bounding box进行iou的计算，如果某个iou大于阈值的话，我们就将这个bounding box给取出来
该算法具体的步骤如下：

1、假定有6个带置信率的region proposals，并预设一个IOU的阈值如0.7。
2、按置信率大小对6个框排序: 0.95, 0.9, 0.9, 0.8, 0.7, 0.7。【[B0,0.98],[B1,0.14]...】
3、设定置信率为0.95的region proposals为一个物体框；
4、在剩下5个region proposals中，去掉(其实质是将其score置0，但在soft-max里面是将其输入到一个函数f里面进行置于0.几)与0.95物体框IOU大于0.7的。(大于0.7说明我们这个0.95更接近于真实物体框)
5、重复2～4的步骤，直到没有region proposals为止。
6、每次获取到的最大置信率的region proposals就是我们筛选出来的目标。
这个操作的直观结果，就是图片上每个待检查物体，仅仅保留一个置信率最大region proposals。
凡是与这个置信率最大的框相交面积大于某个阈值的其它框，统统去掉。(那其他小于这个阈值的框怎么处理呢？？？---不作处理等待下一轮操作继续处理)
因此，调整这个IOU的阈值，往往是图像目标检测算法的一个调整点。

删除分数最高的，(这两个条件必须同时满足)同时IOU达到一定阈值的框，为什么要这样做呢？**因为**有可能预测其他物体的备选框分数也很高，但是跟我们前面这个分数高的备选框IOU为0，所以这些其他框暂时就被保留了下面，这样我就理解了。

# 两步IOU，一个是在CNN里面，一个是和真实值进行IOU

# rethinking about NMS
就是有俩物体重叠的时候应该怎么解决呢？
这样我们就提出来了Soft-NMS
————————————————
# soft-NMS： 是为了解决重叠物体而产生的
搞出来一个函数si=si x (高斯核)；高斯核=e的-sigma分之iou(M,bi)的平方
这样就能保留多个bounding box，尽量保留分数较高的bounding box

An algorithm which decays the detection scores of all other objects as a continuous function of their overlap with M. 换句话说就是用稍低一点的分数来代替原有的分数，而不是直接置零。

————————————————
原文链接：https://blog.csdn.net/u014380165/article/details/79502197
# NMS的弊端如下图所示
```

![这里写图片描述](https://img-blog.csdn.net/20180309200527621?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##### 理解NMS和soft-NMS的原理

​	Figure2是Soft NMS算法的伪代码。首先是关于三个输入B、S、Nt，在FIgure2中已经介绍很清楚了。D集合用来放最终的box，在boxes集合B非空的前提下，搜索score集合S中数值最大的数，假设其下标为m，那么bm（也是M）就是对应的box。然后将M和D集合合并，并从B集合中去除M。再循环集合B中的每个box，这个时候就有差别了，如果是传统的NMS操作，那么当B中的box bi和M的IOU值大于阈值Nt，那么就从B和S中去除该box；如果是Soft NMS，则对于B中的box bi也是先计算其和M的IOU，然后该IOU值作为函数f()的输入，最后和box bi的score si相乘作为最后该box bi的score。就是这么简单！

![这里写图片描述](https://img-blog.csdn.net/20180309200626763?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

接下来得重点就是如何确定函数f()了。
首先NMS算法可以用下面的式子表示：

![这里写图片描述](https://img-blog.csdn.net/20180309200655456?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

为了改变NMS这种hard threshold做法，并**遵循iou越大，得分越低的原则**（iou越大，越有可能是false positive），自然而然想到可以用下面这个公式来表示Soft NMS：

![这里写图片描述](https://img-blog.csdn.net/20180309200714754?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

但是上面这个公式是不连续的，这样会导致box集合中的score出现断层，因此就有了下面这个Soft NMS式子（也是大部分实验中采用的式子）：

![这里写图片描述](https://img-blog.csdn.net/20180309200749873?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这个式子满足了：A continuous penalty function should have no penalty when there is no overlap and very high penalty at a high overlap.(一个连续的处罚函数应该没有处罚当它没有重叠且有非常高的处罚在特别高的重叠上)，对的，当iou(M,bi)=0的时候，si=si;当iou(M,bi)特别大的时候e这个值就特别小，si*e这个值就越小。

实验结果：

![这里写图片描述](https://img-blog.csdn.net/20180309200815578?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Table1是在MS-COCO数据集上的结果对比。表中的G表示上面连续的Soft NMS公式，L表示上面不连续的Soft NMS公式，从实验对比可以看出二者之间的差别并不大。对于G类型的Soft NMS，参数a取0.5，对于L类型的Soft NMS，参数Nt取0.3（感觉这个参数取的有点偏低）。

Table2是在VOC 2007数据集上的结果对比。

![这里写图片描述](https://img-blog.csdn.net/20180309200832480?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Table3是一个很不错的对比图。左边一半是NMS，右边一半是Soft NMS。在NMS部分，相同Ot条件下（Ot较小的情况下），基本上Nt值越大，其AP值越小，这主要是因为有越多的重复框没有过滤掉。

![这里写图片描述](https://img-blog.csdn.net/20180309200850638?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

————————————————------------------------------------------------------------------------------------------------------
原文链接：https://blog.csdn.net/u014380165/article/details/79502197

NMS代码：

```python
# 传统的NMS代码
'''
NMS作业标答
作业内容：
1. 实现非极大值抑制(NMS)，并使用NMS对原始人脸框进行筛选；
2. 尝试调整NMS中使用的置信度，研究不同置信度对人脸框筛选会造成什么影响。
备注：作业所提供的人脸框(face_box)是从MTCNN的R-net与O-net中获得的。
'''
import numpy as np
import cv2

# 读入图片，录入原始人脸框（[x1, y1, x2, y2, score]）
image = cv2.imread('image.jpg')
face_boxs = np.array([[238, 82, 301, 166, 0.99995422], [239, 86, 300, 166, 0.99997818], [341, 26, 412, 112, 0.99781644],
                      [239, 83, 301, 166, 0.99990737], [85, 49, 152, 132, 0.99995887], [340, 25, 411, 112, 0.99890125],
                      [341, 26, 412, 111, 0.99748683], [85, 49, 151, 130, 0.99962735], [84, 48, 151, 130, 0.99987411],
                      [340, 28, 409, 112, 0.99846846], [341, 28, 410, 111, 0.99695492], [340, 26, 410, 110, 0.99970192],
                      [341, 27, 410, 111, 0.99794656], [238, 84, 299, 165, 0.99928051], [84, 49, 151, 131, 0.99978763],
                      [85, 49, 148, 131, 0.99988151], [238, 81, 305, 168, 0.99999976], [340, 26, 410, 112, 0.99981469],
                      [84, 52, 153, 134, 0.99992657], [336, 23, 411, 114, 0.99238223], [238, 83, 300, 164, 0.99994004],
                      [236, 83, 301, 164, 0.99982053], [340, 25, 411, 112, 0.9982546], [85, 50, 150, 139, 0.99916756],
                      [85, 49, 151, 131, 0.99978501], [232, 87, 317, 173, 0.99997389], [330, 26, 438, 134, 0.9898662],
                      [236, 96, 306, 166, 0.99976283], [359, 38, 431, 110, 0.98443735], [351, 31, 434, 115, 0.99634606],
                      [225, 75, 335, 185, 0.99919599], [311, 13, 454, 156, 0.92719758], [87, 59, 170, 142, 0.99837035],
                      [259, 100, 309, 150, 0.92693377], [241, 91, 316, 166, 0.99995005], [79, 60, 161, 141, 0.99849546],
                      [82, 53, 140, 111, 0.96095043], [72, 52, 183, 162, 0.96566218], [341, 38, 406, 104, 0.99826789],
                      [254, 101, 306, 153, 0.90867722], [319, 23, 402, 106, 0.99615687], [335, 30, 423, 119, 0.999345],
                      [117, 74, 161, 119, 0.92760825], [215, 78, 318, 181, 0.99981409], [101, 60, 169, 127, 0.99795973],
                      [238, 104, 287, 153, 0.96899307], [245, 115, 294, 164, 0.89920408],
                      [243, 88, 330, 176, 0.99885798],
                      [86, 67, 160, 141, 0.98279655], [234, 90, 299, 155, 0.99896216], [75, 59, 166, 150, 0.98545951],
                      [224, 80, 321, 177, 0.99998498], [87, 56, 149, 118, 0.99664032], [85, 72, 133, 120, 0.78204125],
                      [346, 25, 455, 134, 0.8496629], [334, 24, 434, 124, 0.99889356], [322, 35, 407, 120, 0.99624914]])

# 将原始人脸框绘制在人脸图像上
image_for_all_box = image.copy()
for box in face_boxs:
    x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
    image_for_all_box = cv2.rectangle(image_for_all_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('image_for_all_box', image_for_all_box)


# 定义一个nms函数
def nms(dets, thresh):
    '''
    input：
        dets: [x1, y1, x2, y2, score]
        thresh: float
    output：
        index
    '''
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep


# 使用nms对人脸框进行筛选
keep = nms(face_boxs, thresh=0.6)
nms_face_boxs = face_boxs[keep]

# 将筛选过后的人脸框绘制在人脸图像上
image_for_nms_box = image.copy()
for box in nms_face_boxs:
    x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
    image_for_nms_box = cv2.rectangle(image_for_nms_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('image_for_nms_box', image_for_nms_box)

cv2.waitKey()
cv2.destroyAllWindows()
```

```python
# soft-NMS代码



```





![1627866132626](D:\AppData\Roaming\Typora\typora-user-images\1627866132626.png)

### 3、ROI Pooling

```python
# 听课笔记
# RIO目的：resize成相同形状
比如我们的目标是变成2x2,我们先打格成2x2
1、ROI形状不同，可以将其resize成7x7,resize是自己决定的，但惯例是7x7
2、resize可能造成信息的丢失，那究竟是怎么弄丢失的呢？
答案：除以16，形成红色虚线框，就会有小数产生，pixel必须是整数，所以取整得到红色实线框。
所以对小物体的检测就不是很好
3、那么有没有什么方法让我们的pooling变好呢？
答案：ROI Align[2017,Kaiming]
里面运用到双线性插值方法，得到小粉点，然后在相邻的四个小粉点里面选择max值，得到2x2的方格
ROI Align的问题：
	黑框是怎么来的？3x3,N=4是人为添加来的，所以这个流程里面有人为因素参与
3、Precise ROI Pooling[2018,iou-Net(是为了解决NMS而来的)]    
ROI过程：
	首先，假如原图3x3,然后要除以16，框会变小，这样我们使用双线性插值方法就可以得到小数点组成的框，这每一个小数点框中的一个像素值是通过原图4个像素点双线性插值方法得来的，唉，这下明白红点怎么来了的吧。
    
```

```python
# 网上摘抄的ROI理解过程
```

> 前言
> RoI Pooling 是目标检测任务中的常见手段，最早在 Faster R-CNN 中提出，作用是将一系列大小不同的 RoI 投影至特征图上，然后通过池化操作将它们处理为一致大小，从而方便后面的网络层进行处理（历史原因，**以前的网络结构中最后几层往往是全连接层，因此需要固定的输入尺寸**），同时起到了加速计算的作用。
>
> 本文先对 RoI Pooling 进行介绍，该方法由于量化误差而带来了精度上的损失，后来有大神基于该方法提出了 RoI Align 和 Precise RoI Pooling，本文后半部分会让大伙瞧瞧这俩个家伙的玩法。
>
> 本文框架
> 01.RoI Pooling -- **将不同的尺寸变为一致**
>
> 02.RoI Align -- **没有量化误差**
>
> 03.Precise RoI Pooling -- **无需超参，每个像素点均有梯度贡献**
>
> 一、Rol Pooling——将不同的尺寸变为一致
> 先来概述下 RoI Pooling 的操作：
>
> i). RoI 的尺寸通常是对应输入图像的，特征图是输入图像经过一系列卷积层后的输出，因此，首先将 RoI 映射到特征图上的对应区域位置；
>
> ii). 最终需要将尺寸不一的 RoI 变为固定的 n x n 大小，于是将 RoI 平均划分为 n x n 个区域；
>
> iii). 取每个划分而来的区域的最大像素值，相当于对每个区域做 max pooling 操作，作为每个区域的“代表”，这样每个 RoI 经过操作后就变为 n x n 大小。
>
> 结合一个例子说明下 RoI Pooling 带来的量化误差：
>
> 如下图，假设输入图像经过一系列卷积层下采样32倍后输出的特征图大小为8x8，现有一 RoI 的左上角和右下角坐标（x, y 形式）分别为(0, 100) 和 (198, 224)，映射至特征图上后坐标变为（0, 100 / 32）和（198 / 32，224 / 32），由于像素点是离散的，因此向下取整后最终坐标为（0, 3）和（6, 7），这里产生了第一次量化误差。
>
> 假设最终需要将 RoI 变为固定的2x2大小，那么将 RoI 平均划分为2x2个区域，每个区域长宽分别为 (6 - 0 + 1) / 2 和 (7 - 3 + 1) / 2 即 3.5 和 2.5，同样，由于像素点是离散的，因此有些区域的长取3，另一些取4，而有些区域的宽取2，另一些取3，这里产生了第二次量化误差。
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\01.jpg)
>
> 二、RoI Align——没有量化误差 
> RoI Align 是在 Mask R-CNN 中提出来的，基本流程和 RoI Pooling 一致，但是没有量化误差，下面结合一个例子来说明：
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\02.jpg)
>
>
> 如上图，输入图像分辨率为800x800，其中一个 RoI 大小为 665x665，输入图像经过 VGG16 下采样32倍后输出分辨率为25x25的特征图。
>
> 1). 将 RoI 映射至特征图上，大小为 （665/32）x（ 665/32） 即 20.78x20.78，注意这里不进行取整；
>
> 2). 最终需要将 RoI 输出为7x7大小，因此将 20.78x20.78大小的 RoI 均分为7x7个区域，每个区域大小为2.97x2.97，注意这里也没有取整操作；
>
> 3). RoI Align 需要设置一个超参，代表每个区域的采样点数，即每个区域取几个点来计算“代表”这个区域的值，通常为4；
>
> 4). 对每个划分后的区域长宽各划分为一半，“十字交叉”变为4等份，取每份中心点位置作为其“代表”，中心点位置的像素值利用双线性插值计算获得，这样就得到4个中心点像素值，采样点数为4就是这个意思；
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\03.jpg)
>
> ​					2.97x2.97区域划分为4等份，每份利用双线性插值计算其中心点像素值
> 5).每个2.97x2.97的区域都有4个中心点像素值，它们分别取4个中心点像素值中的最大值作为其“代表”，这样7x7个区域就产生7x7个值，最终将 RoI 变为了7x7大小。
>
> 三、Precise RoI Pooling ——无需超参  
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\04.jpg)   
>
> RoI Align 虽然没有量化损失，但是却需要设置超参，对于不同大小的特征图和 RoI 而言这个超参的取值难以自适应，于是就有人提出 Precise RoI Pooling 来解决这一问题，真是人才辈出呐！
>
> Precise RoI Pooling 和 RoI Align 类似，将 RoI 映射到特征图以及划分区域时都没有量化操作，不同的是，Precise RoI Pooling 没有再次划分子区域，而是对每个区域计算积分后取均值来“代表”每个区域，因而不需要进行采样。
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\05.jpg)
>
>
> 另外，由上述公式可知，区域内的每点在反向传播中对梯度都是有贡献的，而对 RoI Align 和 RoI Pooling 来说，只有区域内最大值那点才对梯度有贡献，相当于“浪费”了大部分的点。
>
> 四、总结
> 以上操作的原理不难理解，很多人看完后或许都有这样的feel——咦，挺简单的嘛！但是如果试着从代码层面去实现的话或许就会发现不是那么容易了，特别是，要能应用到生活场景中。这里我手撸的 RoI Pooling 和 RoI Align 也仅是基于原理去实现的，用作学习参考，真正业界上应用的通常是用C或C++实现和编译的，纯 py 的版本通常由于性能原因难以落地到工程中。另外，这里附上的 Precise RoI Pooling 源码是原作者的版本，但是我使用时一直出现编译错误，各位大侠可以试试，如果有类似问题或者解决方案希望可以和我一起探讨，谢谢！ 
>
> 源码参考
> https://www.cnblogs.com/wangyong/p/8523814.html 

#### 1)、传统RIO

​	备注：因为取最大值的原因，只有一个loss传递，just one point  loss passed in each bin.

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\ROI.jpg)

#### 2)、RIO Align

粉点是在黑格中双线性插值来的，然后在小绿框中将四个小粉点通过双向性插值得到黄点。

黑框是怎么来的？

N=4，是人为添加试出来的，这样的话就有四个点传递给我们的loss

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\ROI_Align.jpg)

#### 3)、Precise RIO

我直接在绿框中进行插值，更加暴力。红点是通过蓝点双线性插值得到的。红点是相对于蓝点(整数点)带小数的点

红点就是蓝点的精细版的点。

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\Precise_ROI.jpg)

### 4、FAST R-CNN算法解析

```python
# 听课笔记：
1、ROI Pooling的目的：
将不同size的ROI变成相同的size
2、相比于RCNN,多了ROI Projection
3、FAST RCNN流程
B0.Region Proposal:same as RCNN
   2K per image.record location for each ROI
B1&B2.Convolution & Projection
	a.do conv for image,project location for each ROI
    b.3 basic structures provided,use Vgg16 as an E.g.
    c.4 max pooling,/16 # 就是4层，除以16
B3. ROI Pooling(是在feature map当中处理)
	a.grid each ROI in feature map to fixed size,and do max pooling within each grid
    b. so different size of feature maps can transfer into feature maps with same size
# 代码中表示物体：
			  左上角和右上角(x1,y1,x2,y2)
			  中心点坐标和框的宽高(x,y,w,h)
# VIP:
    region proposal是在原图当中，而ROI projection是在对的feature map当中，是有一一对应关系的        
B4. FC Layers
	fc layer cost a lot,can use SVD to accerate（这句话就是扯淡，原文作者在代码里面就根本没有用这个加速方法，面试可不提这一点）
B5. Multi-task Loss

```



### 5、Faster R-CNN算法解析

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_42.jpg)

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_44.jpg)

上图是test阶段对应的图

```python
# 听课笔记
Faster R-CNN[2015]:
结构如上图所示：整体结构有三块，
C1.Backbone
	a.Aim:extract feature integratedly
        -No need to extract feature per ROI;
        -Use generated feature to generate proposal
    b.structure:ZF/ResNet/VGG16(13conv+13ReLU+4Pooling)
    c.output:B x C x H/16 x W/16
        	 1 x 256 x 38 x 50
             [feature map]  # 这里backbone输出的就是feature map。
网络结构如上所示：
C2.RPN
网络上面部分是classification，下面一部分是regression，1x1操作是为了改变channel的个数，上面channel变成了18，下面regression变成了36.
@1.Aim:generate region proposal[real detected objects are coming from RPs]
 	%1.it's the reason where the name "two -stage" coming from:RPN+Bbox Regression
    %2.it's the reason why some argue two-stage detection has better results then one-stage methods
@2.output of RPN:
    %1.ROIs:128x5
        [0,x1,y1,x2,y2]->physical region proposal; # 备注：0就是占位
    %2.Label:128
        [0~20]-> RIOs' classification   # 备注：0对应的是background
    %3.bbx_target:128x84
        [(20+1)x4]->targets for bounding box regression # 4代表的是x1,y1,x2,y2四个参数
	%4.bbx_weight:128x84
        [0 or 1]->weights for bx_target when box regressing # 0就是我们的背景，1就是我们的前景。
   train和test最大的差距就是train多了回归loss和分类loss
C2.1:Classification branch
    18=9x2,2就是0/1,9个anchor
    anchor是如何产生的？
    答案：原始图像经过CNN网络得到很多Tensor-feature map，feature map中的每一个点point对应原始图像中的一个块(代表原始图像的一个区域)，原始的anchor我们选择16*16（2:1,1:1；1:2），分别会乘以8/16/32:  128*128,256*256,512*512，所以基本会覆盖掉大中小三种size的物体。每一个特征点代表着9个anchor。
    # structure：anchor****************************************************
    %1.represent an area in original image
    %2.have different scale & ratio to cover all kinds of objects
    %3. 9 anchors in total:3 scales x 3 ratios
	%4. 1x9x38x50=17100 anchors        
    %5. coordinates of anchors are of original images
备注：每一个feature map中的pixel点有9个anchor，每个anchor对应原图中的一个小区域
C2.1.1:classification reshape
    因为softmax结果是二分类，所以要把feature map搞成2个channel，所以38x9x2(将18拆成9x2，然后摞起来。)
C2.1.2:softmax
    2 branches:
        a.bp loss
        b.no loss/get score
	@1、1 foreground & 0 background.
    	get loss with anchor and pass back.
    	use IOU:iou<0.3标注为0，iou>0.7标注为1，如果是在0.3~0.7在代码中我们就ignore掉。
    @2、just do classification and get the score,
    	use the score to select proposal.
# anchor的压缩：
	第一次压缩：超出图像边界的anchor我们ignore了        
    第二次压缩：ignore
    第三次压缩：sofmax之后对iou进行一个排序，完了取前12000个
    第四次压缩：NMS
C2.1.3: cls reshape
    @1.back to original shape:reshape成38x50
    @2.each voxel(像素) represents the possibility of being a foreground or a background of an anchor.
C2.2:regression
    channel是36：每一个pixel有9个anchor，每一个anchor有4个坐标
    1x36(9x4,4个坐标(x0,y0,x1,y1))x38x50

# 回归问题
C2.2.1: Smooth L1 Loss
    @1. need to multiply the mask
    	%1.anchor=1,mask=1
        %2.anchor=0,-1,mask=0
    @2. smooth L1 Loss：L1 Loss在0处不可导,L2 Loss处处连续可导（图像与y=x^2类似）
    公式(代码中有定义好的接口)见课件：
# Regress（center）offset,Not coordinates
  这种方法是现在好多地方都在用的方法，所以必须得掌握
公式里使用log的原因：suppress greater bbox，能直接回归。
# 为啥不能直接回归坐标而是要用offset回归？
    因为直接回归对小物体是致命打击**************************************
    @)、why use exp
        tx & ty is used for scaling we need to ensure scale>0
    @)、why use log
        a、reversing procedure of exp
        b、suppress greater bbox
C2.3:get proposal
    @1.regard anchor as FG when IOU>0.7
    @2.regard anchor as BG when IOU<0.3
    @3.regardless other anchors
    @4.then just keep 128:0.25fg+0.75bg anchors
C3. After RPN
	same as Fast RCNN
C4. how to train
	@1.Classic:alternatively 4 steps
        %1. use imageNet finetune(微调) our RPN
        %2. use trained proposal from step1(RPN的输出) to train an Fast RCNN
        %3. use detected results to initialize RPN training where we freeze the backbone layers but to train pure RPN-related layers
        %4.finetune fast RCNN-related layers only.
    @2.other methods:
        we can also train Faster RCNN as a whole in just one step
# Faster RCNN用的不是非常多，主要用的是它提出的anchor，实际场景当中运用的还是很少。
下图右边图多了俩loss
```

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_49.jpg)



![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_59.jpg)



> 主要是关于 mid-level 的 cv 技法，包括但不限于： 
>
> 1. SIFT 特征点：（这个最重要） 
>
>    a. 图像平滑（高斯核卷积） 
>
>    b. 光照不变形（图像颜色变化） 
>
>    c. Image pyramid, DoG （高斯核卷积） 
>
>    d. Harris 角点 (泰勒展开) 
>
>    e. 极值点（线性插值） 
>
>    f. 特征向量的生成（旋转不变形） 
>
> 2. 其他相关特征点， 包括但不限于： 
>
>    HoG, SURF, ORB, FAST 等 
>
> 3. Haar 特征，Integral Image(积分图) 
>
> 4. 传统 ML 方法： 
>
>    SVM，Decision Tree, Logistic Regression, Linear Regression, Neural Network, 
>
> Adaboost 

## One-Stage检测算法

### 1、理论

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_00.jpg)

![week1-4 Detection-3 stages_2021_01](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_01.jpg)

![week1-4 Detection-3 stages_2021_02](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_02.jpg)

![week1-4 Detection-3 stages_2021_03](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_03.jpg)

![week1-4 Detection-3 stages_2021_04](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_04.jpg)

![week1-4 Detection-3 stages_2021_05](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_05.jpg)

![week1-4 Detection-3 stages_2021_06](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_06.jpg)

![week1-4 Detection-3 stages_2021_07](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_07.jpg)

![week1-4 Detection-3 stages_2021_08](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_08.jpg)

![week1-4 Detection-3 stages_2021_09](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_09.jpg)

![week1-4 Detection-3 stages_2021_10](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_10.jpg)

![week1-4 Detection-3 stages_2021_11](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_11.jpg)

![week1-4 Detection-3 stages_2021_12](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_12.jpg)

![week1-4 Detection-3 stages_2021_13](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_13.jpg)

![week1-4 Detection-3 stages_2021_14](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_14.jpg)

![week1-4 Detection-3 stages_2021_15](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_15.jpg)

![week1-4 Detection-3 stages_2021_16](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_16.jpg)

![week1-4 Detection-3 stages_2021_17](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_17.jpg)

![week1-4 Detection-3 stages_2021_18](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_18.jpg)

![week1-4 Detection-3 stages_2021_19](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_19.jpg)

![week1-4 Detection-3 stages_2021_20](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_20.jpg)

![week1-4 Detection-3 stages_2021_21](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_21.jpg)

![week1-4 Detection-3 stages_2021_22](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_22.jpg)

![week1-4 Detection-3 stages_2021_23](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_23.jpg)

![week1-4 Detection-3 stages_2021_24](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_24.jpg)

![week1-4 Detection-3 stages_2021_25](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_25.jpg)

![week1-4 Detection-3 stages_2021_26](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_26.jpg)

![week1-4 Detection-3 stages_2021_27](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_27.jpg)

![week1-4 Detection-3 stages_2021_28](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_28.jpg)

![week1-4 Detection-3 stages_2021_29](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_29.jpg)

![week1-4 Detection-3 stages_2021_30](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_30.jpg)

![week1-4 Detection-3 stages_2021_31](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_31.jpg)

![week1-4 Detection-3 stages_2021_32](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_32.jpg)

![week1-4 Detection-3 stages_2021_33](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_33.jpg)

**别信作者的鬼话**：SVD能不能加速网络作者自己都没试验

![week1-4 Detection-3 stages_2021_34](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_34.jpg)

![week1-4 Detection-3 stages_2021_35](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_35.jpg)

![week1-4 Detection-3 stages_2021_36](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_36.jpg)

![week1-4 Detection-3 stages_2021_37](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_37.jpg)

![week1-4 Detection-3 stages_2021_38](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_38.jpg)

![week1-4 Detection-3 stages_2021_39](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_39.jpg)

![week1-4 Detection-3 stages_2021_40](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_40.jpg)

![week1-4 Detection-3 stages_2021_41](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_41.jpg)

![week1-4 Detection-3 stages_2021_42](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_42.jpg)

![week1-4 Detection-3 stages_2021_43](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_43.jpg)

**要理解这句话哦，在RCNN中，feature map是在每个RP里面提取出来的，但在Faster RCNN中，就是先产生feature map，然后在feature map中产生proposals。**

**RCNN流程：**
input image-->extract region proposals(~2K大概就是2000个)-->compute CNN features-->classify regions(Classification+Detection) 

**由这个流程可以看出来，RCNN中的的region proposals是在原图上产生的，但在Faster RCNN中，先是在backbone部分提取feature，然后再feature里面进行产生region proposals。**

------

![week1-4 Detection-3 stages_2021_44](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_44.jpg)

------

![week1-4 Detection-3 stages_2021_45](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_45.jpg)

![week1-4 Detection-3 stages_2021_46](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_46.jpg)

上图得好好研究一下这个过程，具体是怎么产生的。

sliding window:

代码中其实做了conv2d()卷积操作

https://zhuanlan.zhihu.com/p/116022332

![week1-4 Detection-3 stages_2021_47](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_47.jpg)

![week1-4 Detection-3 stages_2021_48](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_48.jpg)

![week1-4 Detection-3 stages_2021_49](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_49.jpg)

![week1-4 Detection-3 stages_2021_50](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_50.jpg)

![week1-4 Detection-3 stages_2021_51](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_51.jpg)

![week1-4 Detection-3 stages_2021_52](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_52.jpg)

**备注**：anchor是在原图上产生的，anchor的坐标也是相对于原图的坐标。so important!!!!!!!!!!!!!

------

![week1-4 Detection-3 stages_2021_53](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_53.jpg)

这一步reshape的目的：为了softmax时获取前景和背景。

![week1-4 Detection-3 stages_2021_54](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_54.jpg)

![week1-4 Detection-3 stages_2021_55](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_55.jpg)

**reshape成原来的尺寸**：back to original shape

------

![week1-4 Detection-3 stages_2021_56](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_56.jpg)

![week1-4 Detection-3 stages_2021_57](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_57.jpg)

------

![week1-4 Detection-3 stages_2021_58](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_58.jpg)

![week1-4 Detection-3 stages_2021_59](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_59.jpg)

![week1-4 Detection-3 stages_2021_60](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_60.jpg)

![week1-4 Detection-3 stages_2021_61](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_61.jpg)

![week1-4 Detection-3 stages_2021_62](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_62.jpg)

最后loss回归完了通过anchor，最后生成的就是proposals共128个。

------

![week1-4 Detection-3 stages_2021_63](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_63.jpg)

![week1-4 Detection-3 stages_2021_64](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_64.jpg)

**开启YOLO检测算法**

![week1-4 Detection-3 stages_2021_65](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_65.jpg)

![week1-4 Detection-3 stages_2021_66](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_66.jpg)

![week1-4 Detection-3 stages_2021_67](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_67.jpg)

![week1-4 Detection-3 stages_2021_68](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_68.jpg)

![week1-4 Detection-3 stages_2021_69](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_69.jpg)

![week1-4 Detection-3 stages_2021_70](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_70.jpg)

![week1-4 Detection-3 stages_2021_71](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_71.jpg)

![week1-4 Detection-3 stages_2021_72](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_72.jpg)

------

YOLO独门秘籍------------**打格法**

![week1-4 Detection-3 stages_2021_73](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_73.jpg)

![week1-4 Detection-3 stages_2021_74](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_74.jpg)

![week1-4 Detection-3 stages_2021_75](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_75.jpg)

注意上面2 个方格图，有一个很奇怪的0，你注意到了吗？

那个就是在代码里通过anchor_threshold阈值来确定的，不让它参与计算

![week1-4 Detection-3 stages_2021_76](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_76.jpg)

![week1-4 Detection-3 stages_2021_77](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_77.jpg)

![week1-4 Detection-3 stages_2021_78](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_78.jpg)

each cell will only .....

![week1-4 Detection-3 stages_2021_79](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_79.jpg)

![week1-4 Detection-3 stages_2021_80](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_80.jpg)

![week1-4 Detection-3 stages_2021_81](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_81.jpg)

![week1-4 Detection-3 stages_2021_82](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_82.jpg)

![week1-4 Detection-3 stages_2021_83](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_83.jpg)

![week1-4 Detection-3 stages_2021_84](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_84.jpg)

![week1-4 Detection-3 stages_2021_85](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_85.jpg)

![week1-4 Detection-3 stages_2021_86](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_86.jpg)

**awori指的是原始anchor的宽**(原始的anchor是基于原图尺寸的)，然后再除以**图像的宽**，再乘以13，原图打格成13x13，最终我们的feature map是13x13的大小，所以我们要把anchor归一化到13x13的范围内。

上图anchor的公式目的就是把anchor的宽和高确定下来。

因为有5个anchor，所以共有10个数字。

------

![week1-4 Detection-3 stages_2021_87](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_87.jpg)

明白了。其中，f指的是final，i,j指的是cell的ID

你看。假如x是2.3(因为x的范围在0~13之间)，那么它对应的i就是2

假如x是4.8，那么它对应的i就是4，i和j对应的是x和y的整数部分。

这样的话最后两行公式就将cell有机的结合起来了。

理解：原图13x13个格，然后经过conv层输出feature map也是13x13个点，每一个点代表原图中的每一个cell或者可以说是图像块。大概意思如下图：

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\2021-08-11_165740.jpg)

![week1-4 Detection-3 stages_2021_88](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_88.jpg)

![week1-4 Detection-3 stages_2021_89](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_89.jpg)

开始理解：

蓝色框是预测框--predicted BBoxes

虚线框是anchor：参数值是pw和ph，值的范围都在0~13之间。

我们选择和ground truth最大IOU的那个anchor来回归，跟GT没有IOU或者iou小的anchor就不参与回归了。

同时不能忘了一个规则：就是物体落在了哪个cell，那个cell就去预测回归那个物体

其中，bx就是上面一直说的x坐标值，其值属于0~13的范围

左边公式中的tg一定在0~1之间。

网络给我们的是tp值，但是我们不能保证tp它一定就在0~1之间，而我们的target--tg却在0~1之间，所以我们加个sigmoid函数就将tp的范围限制在0~1之间了。

bx,by,bw,bh依然是我们想要的值。

------

面试题：

​	anchor应该怎么设计？

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422155816281.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMDg4NDc1,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422160010100.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMDg4NDc1,size_16,color_FFFFFF,t_70)

------

![week1-4 Detection-3 stages_2021_90](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_90.jpg)

![week1-4 Detection-3 stages_2021_91](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_91.jpg)

![week1-4 Detection-3 stages_2021_92](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_92.jpg)

![week1-4 Detection-3 stages_2021_93](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_93.jpg)

![week1-4 Detection-3 stages_2021_94](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_94.jpg)

这种图像金字塔对应的一般是传统CV，SIFT这个必须得会啊。

FPN: Feature Pyramid Network

右下角下面这张图，如果没有predict的话，那么它就是一个U-Net结构。

![week1-4 Detection-3 stages_2021_95](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_95.jpg)

![week1-4 Detection-3 stages_2021_96](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_96.jpg)

![week1-4 Detection-3 stages_2021_97](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_97.jpg)

![week1-4 Detection-3 stages_2021_98](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_98.jpg)

![week1-4 Detection-3 stages_2021_99](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_99.jpg)

![week1-4 Detection-3 stages_2021_100](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_100.jpg)

**上面这个后面在数据增强部分会讲到**

![week1-4 Detection-3 stages_2021_101](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_101.jpg)

![week1-4 Detection-3 stages_2021_102](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_102.jpg)

![week1-4 Detection-3 stages_2021_103](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_103.jpg)

![week1-4 Detection-3 stages_2021_104](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_104.jpg)

![week1-4 Detection-3 stages_2021_105](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_105.jpg)

![week1-4 Detection-3 stages_2021_106](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_106.jpg)

![week1-4 Detection-3 stages_2021_107](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_107.jpg)

![week1-4 Detection-3 stages_2021_108](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_108.jpg)

![week1-4 Detection-3 stages_2021_109](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_109.jpg)

![week1-4 Detection-3 stages_2021_110](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_110.jpg)

![week1-4 Detection-3 stages_2021_111](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_111.jpg)

![week1-4 Detection-3 stages_2021_112](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_112.jpg)

![week1-4 Detection-3 stages_2021_113](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_113.jpg)

![week1-4 Detection-3 stages_2021_114](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_114.jpg)

![week1-4 Detection-3 stages_2021_115](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_115.jpg)

![week1-4 Detection-3 stages_2021_116](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_116.jpg)

![week1-4 Detection-3 stages_2021_117](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_117.jpg)

![week1-4 Detection-3 stages_2021_118](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_118.jpg)

![week1-4 Detection-3 stages_2021_119](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_119.jpg)

![week1-4 Detection-3 stages_2021_120](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_120.jpg)

![week1-4 Detection-3 stages_2021_121](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_121.jpg)

![week1-4 Detection-3 stages_2021_122](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_122.jpg)

![week1-4 Detection-3 stages_2021_123](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_123.jpg)

![week1-4 Detection-3 stages_2021_124](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_124.jpg)

![week1-4 Detection-3 stages_2021_125](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_125.jpg)

![week1-4 Detection-3 stages_2021_126](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_126.jpg)

![week1-4 Detection-3 stages_2021_127](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_127.jpg)

![week1-4 Detection-3 stages_2021_128](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_128.jpg)

### 2、代码

```python
# 听课笔记：

```



### 3、作业:NMS

```python
# NMS作业
'''
NMS作业标答
作业内容：
1. 实现非极大值抑制(NMS)，并使用NMS对原始人脸框进行筛选；
2. 尝试调整NMS中使用的置信度，研究不同置信度对人脸框筛选会造成什么影响。
备注：作业所提供的人脸框(face_box)是从MTCNN的R-net与O-net中获得的。
'''
import numpy as np
import cv2

# 读入图片，录入原始人脸框（[x1, y1, x2, y2, score]）
image = cv2.imread('image.jpg')
face_boxs = np.array([[238, 82, 301, 166, 0.99995422], [239, 86, 300, 166, 0.99997818], [341, 26, 412, 112, 0.99781644],
                    [239, 83, 301, 166, 0.99990737], [ 85, 49, 152, 132, 0.99995887], [340, 25, 411, 112, 0.99890125],
                    [341, 26, 412, 111, 0.99748683], [ 85, 49, 151, 130, 0.99962735], [ 84, 48, 151, 130, 0.99987411],
                    [340, 28, 409, 112, 0.99846846], [341, 28, 410, 111, 0.99695492], [340, 26, 410, 110, 0.99970192],
                    [341, 27, 410, 111, 0.99794656], [238, 84, 299, 165, 0.99928051], [ 84, 49, 151, 131, 0.99978763],
                    [ 85, 49, 148, 131, 0.99988151], [238, 81, 305, 168, 0.99999976], [340, 26, 410, 112, 0.99981469],
                    [ 84, 52, 153, 134, 0.99992657], [336, 23, 411, 114, 0.99238223], [238, 83, 300, 164, 0.99994004],
                    [236, 83, 301, 164, 0.99982053], [340, 25, 411, 112, 0.9982546 ], [ 85, 50, 150, 139, 0.99916756],
                    [ 85, 49, 151, 131, 0.99978501], [232, 87, 317, 173, 0.99997389], [330, 26, 438, 134, 0.9898662 ],
                    [236, 96, 306, 166, 0.99976283], [359, 38, 431, 110, 0.98443735], [351, 31, 434, 115, 0.99634606],
                    [225, 75, 335, 185, 0.99919599], [311, 13, 454, 156, 0.92719758], [ 87, 59, 170, 142, 0.99837035],
                    [259,100, 309, 150, 0.92693377], [241, 91, 316, 166, 0.99995005], [ 79, 60, 161, 141, 0.99849546],
                    [ 82, 53, 140, 111, 0.96095043], [ 72, 52, 183, 162, 0.96566218], [341, 38, 406, 104, 0.99826789],
                    [254,101, 306, 153, 0.90867722], [319, 23, 402, 106, 0.99615687], [335, 30, 423, 119, 0.999345  ],
                    [117, 74, 161, 119, 0.92760825], [215, 78, 318, 181, 0.99981409], [101, 60, 169, 127, 0.99795973],
                    [238,104, 287, 153, 0.96899307], [245,115, 294, 164, 0.89920408], [243, 88, 330, 176, 0.99885798],
                    [ 86, 67, 160, 141, 0.98279655], [234, 90, 299, 155, 0.99896216], [ 75, 59, 166, 150, 0.98545951],
                    [224, 80, 321, 177, 0.99998498], [ 87, 56, 149, 118, 0.99664032], [ 85, 72, 133, 120, 0.78204125],
                    [346, 25, 455, 134, 0.8496629 ], [334, 24, 434, 124, 0.99889356], [322, 35, 407, 120, 0.99624914]])

# 将原始人脸框绘制在人脸图像上
image_for_all_box = image.copy()
for box in face_boxs:
    x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
    image_for_all_box = cv2.rectangle(image_for_all_box, (x1, y1), (x2, y2), (0,255,0), 2)
cv2.imshow('image_for_all_box', image_for_all_box)

# 定义一个nms函数
def nms(dets, thresh):
    '''
    input：
        dets: [x1, y1, x2, y2, score]
        thresh: float
    output：
        index
    '''
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep

# 使用nms对人脸框进行筛选
keep = nms(face_boxs, thresh=0.6)
nms_face_boxs = face_boxs[keep]

# 将筛选过后的人脸框绘制在人脸图像上
image_for_nms_box = image.copy()
for box in nms_face_boxs:
    x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
    image_for_nms_box = cv2.rectangle(image_for_nms_box, (x1, y1), (x2, y2), (0,255,0), 2)
cv2.imshow('image_for_nms_box', image_for_nms_box)

cv2.waitKey()
cv2.destroyAllWindows()
```





## One-Stage检测算法

### 1、理论

课件见day02理论部分

### 2、代码

> 1.YOLO V1解决不了两个物体的中心点落在同一个格子，就没法识别解决了。
>
> 2.

```python
# 听课笔记：
# 上周课回顾
1、coding test is the most important
2、数据结构+算法：BFS必会、动态规划、二分查找、排序算法，并查集可以了解一下就行；two point的问题，比如一个数组中，给你个target,find two items that target=add of two items;基础数据结构，红黑树实现不用必须会；数组，链表，栈，队列，树状结构，dict，set等一定要掌握的。比如拿两个队列实现一个栈等等这种复合的实现，链表的翻转，递归，树的遍历（前中后序，迭代及递归方法等等必须要掌握）；层序遍历（可以属于BFS），BP当中比较经典的题：正则表达式，从数组的最左端走到最上面等等；迷宫；deepcopy方法，拷贝类问题，merge成一个linklist等等。
3、one stage 和 two stage的区别及其概念：

4、NMS的算法一定自己掌握：
# NMS算法实现代码：

5、ROI必考题：

6、RPN+Anchor

# 本节课内容
# one stage detction
1、Yolo V1
really fast(18 faster than faster rcnn)
Procedure:
    1.grid an image into S X S cells[将448x448大小的图像-打格成->7x7]:
      one cell will be responsible for predicting an object as long as object's center locating in that cell.
大概意思就是如果哪个方格落到了哪个物体的中心点那么这个方格就去预测这个物体；
假如多个物体的中心点落到了同一个cell中，那么 Yolo V1算法没法解决这个问题，由此产生了后面一系列算法；
	2.each cell predicts B bounding box with a confidence(置信度--用model预测物体的准确度)
    Bounding Box:x,y,w,h(center)，这里B我们取2.
    confidence:P(object).IOU:P(生成物体的可信度)负责预测是不是这个物体，IOU(truth和predict的交并比)预测准不准；P是虚拟出来的一个值，不是网络预测出来的值。
    Final output tensor:S x S x(5*B+C):---->[7X7X(5*2+20)] # pascol vooc数据集中就有20个类别。   C指的是类别，要看你的项目最终要预测几类物体。我们的这个项目是C=2,不戴口罩和戴口罩两种情况。
    一维就是vector向量，二维就是matrix矩阵，三维就是Tensor张量。
    备注：这里(5*B+C)我们叫做Tensor的channel
    具体长什么样看下图，channel方向有2个bbox，每个bbox有5个值：x,y,w,h,c;还有20个类别。
    每一个点预测2个Bounding box,5代表每一个bounding box有5个参数,2代表B1和B2,5对应（x,y,w,h）和confidence。
# w,h=x2-x1+1的原因：不是连续域并非是连续的，数字图像都是点阵矩阵列，x2-x1只能算距离，而在数字图像里面我们要算像素点，所以还得+1就得到了这俩值中间的那个像素点的值。(这点理解很重要)
	20：指20个分类，Tensor的channel就是B1+B2+20，虽然两个bounding box是两个，但是预测的是物体中同一个cell或者图像块，因为他俩一起加了20。
	3. Loss Function
    	开根号的原因：suppress the effect for larger bbox，就是为了对小物体友好一点。
        1对应的是0和1的矩阵mask,乘以mask就是为了去除掉物体没落在cell中的格子，没必要去预测了。
        # 问题：mask矩阵咋来的？？？？？？？？？？？
        答案：
        # 问题：1object和1noobject是取反的关系
        # 看图中，有一个都是0的cell，这个又是什么原因呢？？？？
        答案：我们为了让我们的结果更加鲁棒，在某些cell将1置成0
	4. confidence loss
    	8D就是2个bbox的x,y,w,h
        我们的网络会生成x,y,w,h，用这个生成的去和标注的框进行计算IOU，即就是计算出了confidence
    5. mask只和label有关，和bounding box无关
    6. classification loss

# Yolo V1优缺点总结：------------------------这点要记住也要明白----------------------------
pros：one stage ,really fast
cons:
    1. bad for crowed objects[1 cell 1 object](一定要知道)，还有就是一共7x7=49个cell，所以只能预测49个物体。# 一定要记住
    2. bad for small objects(一定要知道)  # 一定要记住
    3. bad for objects with new width-height ratio
    4. No BN
    
```

```python
# Yolo v2    
1. add bn
2. high resolution classifier[foucusing in backbone]
	a. train on ImageNet(224x224)  # model trained on small images may not be good
    b. resize & Finetune in ImageNet(448x448)  # so we finetune the model on larger images.
    c. Finetune in dataset   # to let the model be used to larger images.
    d. we get 13x13 feature maps finally,代替了Yolo V1的7x7
3. we use anchors # (it is very important)
4. Fine-Grained Features（也可以叫做shortcut）
    浅层网络一般学习到的是物体的物理信息
    首先，对于纯检测问题而言，物理信息更重要，
    但对于正常的检测问题来说，物理信息和语义信息(分类)同等重要。
	a. lower features are concatenated directly to higher features
    b. a new layer is added for that purpose:reorg(如果不理解去看课件，就是将浅层较大的feature进行拆分，然后和深层的feature进行连接)----reorganization
   reorg:跳跃性的取值组合新的卷积/矩阵，比如一个4x4的矩阵通过reorg之后就变成了4个2x2的矩阵。
   但是目前人们已经不用reorganization结构了，基本都用upsample,transponse convolution.
   
5. Multi-Scale Training(可以接受不同尺度的图像---原因就是移除了全连接FC层)
	a. remove FC layers:can accept any size of inputs,enchance model robustness.
    b. size across320,....608.change per 10 epochs
6. Anchor in Yolo V2
	a. what is anchor 
    	1)、pre-defined virtual bboxes:是个已知数，预先人为设置好的虚拟框
        2)、final bboxes are generated from them.
        理解：铺满整张图像。
	b.   

# 训练阶段不需要score=0操作，预测阶段是可以的
每个grid cell只能预测1个物体，7x7=49也就是说只能预测49个物体，这也是YOLO_V1性能差的原因，因为小物体它可能预测不到。
每个grid cell预测2个bbox,那究竟又哪个bbox去预测物体呢？应该由和ground truth IOU比较大的bbox来预测它,也就是置信概率C来决定。
```

#### 问题：

> 1、YOLO V1中，如果多个物体落在了同一个cell中，这种情况应该怎么解决呢？
>
> 答案：
>
> 2、





![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_01.jpg)

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_02.jpg)

![YOLO_V1_03](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_03.jpg)

![YOLO_V1_04](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_04.jpg)

![YOLO_V1_05](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_05.jpg)

![YOLO_V1_06](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_06.jpg)

![YOLO_V1_07](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_07.jpg)

![YOLO_V1_08](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_08.jpg)

![YOLO_V1_09](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_09.jpg)

![YOLO_V1_010](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_010.jpg)

![YOLO_V1_011](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_011.jpg)

![YOLO_V1_012](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_012.jpg)

![YOLO_V1_013](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_013.jpg)

![YOLO_V1_014](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_014.jpg)

![YOLO_V1_015](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_015.jpg)

![YOLO_V1_016](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_016.jpg)

![YOLO_V1_017](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\YOLO_V1_017.jpg)

### 3、作业:Anchor

#### 作业要求

##### @1、思考内容

anchor的生成对于anchor-based的目标检测算法而言至关重要，通过今天的学习，我们应该对anchor的概念有了一个初步的了解，为了让同学加深印象，请同学们先回顾一下一下几个问题：

> 1. anchor是什么？anchor的作用是什么？
> 2. anchor如何参与到运算/训练过程中？
> 3. anchor如何进行回归？
> 4. anchor生成的标准是什么？

##### @2、代码编写

> 1. 请根据课程中所讲述的内容，手动实现实现基础anchor的生成，即使用3个比例、3个大小，生成9个基础anchor；
> 2. 根据原图大小以及下采样特征图的大小，将anchor绘制在原图上。

##### @3、提交要求

> 1. 请提交：你的作业代码（Python文件）
> 2. 请提交：你绘制anchor的图片（可用屏幕截图或通过cv2.write保存一张图片）

##### @4、相关文件

> 1. Anchor-Homework.py
> 2. kkb.jpg

```python
# 标准答案：
'''
Anchor作业标答
作业内容：
1. 实现基础anchor的生成（给定ratios=[0.5, 1, 2]，scales=[128, 256, 512]）；
2. 根据原图大小以及下采样特征图的大小，将anchor绘制在原图上（直接使用原图缩小32倍，来模拟32倍下采样特征图）。
'''
import numpy as np
import cv2

def generate_anchors(scales, aspect_ratios):
    '''
    基础anchor的生成
    input:
        scales: array([128, 256, 512])
        ratios: array([0.5, 1, 2])
    output:
        anchor: array()
    '''
    # TODO
    pass

def grid_anchors(grid_size, stride, cell_anchor):
    '''
    把基础anchor套在网格特征图上
    input:
        grid_size: tuple()
        stride: list()
        cell_anchor: array()
    output:
        anchor: list() or array()
    '''
    # TODO
    pass


if __name__ == '__main__':
    # 读入图片，缩放32倍，以此模拟下采样后的特征图
    image = cv2.imread('kkb.jpeg')
    feature_map = cv2.resize(image, dsize=(image.shape[1]//32, image.shape[0]//32))
    # 获取图片与特征图的shape，计算长、宽方向上的stride
    image_size = image.shape[:2]
    feature_map_size = feature_map.shape[:2]
    strides = [image_size[0] // feature_map_size[0], image_size[1] // feature_map_size[1]]

    # 给定ratios以及scales
    ratios = np.array([0.5, 1, 2])
    scales = np.array([128, 256, 512])
    # 生成基础anchor
    cell_anchors = generate_anchors(scales, ratios)

    # 将基础anchor匹配到下采样所得到的网格特征图上
    all_anchor = grid_anchors(feature_map_size, strides, cell_anchors)

    # 直接生成的anchor会有超出图像边界的地方，这将会导致绘图失败，因此可以先在原图外面进行一圈填充
    image = cv2.copyMakeBorder(image,400,400,400,400,cv2.BORDER_CONSTANT,value=[255,255,255])
    # 将anchor绘制在填充后的图片上
    for box in all_anchor[0]:
        x1, y1, x2, y2 = int(box[0])+400, int(box[1])+400, int(box[2])+400, int(box[3]+400)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

```

##### @5、作业补充

##### AnchorGenerator

总体来说AnchorsGenerator的核心作用有两点

- 生成anchor模板
- 将anchor模板映射到原图像中

##### 生成anchor模板

anchor模板指特征图中单个cell上套用的一组anchor，如图13所示，图中模拟了(小，大)两种尺寸，(1:1,1:2,2:1)(1:1,1:2,2:1), 3种纵横比比例，会有6个anchor组成的模板，其中蓝色为尺寸为小的anchor，橙色表示尺寸为大的anchor。6个anchor的中心点为同一个中心点(0,0)。

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\3、One-Stage检测算法\images\2021-07-11_143907.jpg)

> 为什么要有anchor模板？是因为对于特征图上的每个cell，套用的anchor都是一样的。创建一个anchor模板，将模板“复制”到各个cell当中，下面先看一下在代码的角度怎么实现创建一组anchor模板。
>
> 创建anchor模板的必要条件，也是需要传入的参数是anchor的面积(尺寸)以及横纵比。scales是anchor的面积，aspect_ratios是anchor的横纵比，由于一组anchor要满足不同的大小及横纵比，所以scales和aspect_ratios都是List[int]类型，而anchor模板中anchor的数量是len(scales) * len(aspect_ratios)

```python
import torch

def generate_anchors(scales, aspect_ratios, dtype=torch.float32, device="cpu"):
    # type: (List[int], List[float], int, Device) -> Tensor
    """
    compute anchor sizes
    Arguments:
        scales: sqrt(anchor_area)
        aspect_ratios: h/w ratios
        dtype: float32
        device: cpu/gpu
    """
    # 将scales与aspect_ratios转成tensor，以便后面跟tensor进行计算
    scales = torch.as_tensor(scales, dtype=dtype, device=device)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
    # 求平方根
    # scales表示anchor面积的开平方，面积公式 s = width * height = scales**2
    # scales = sqrt(width * height)
    # aspect_ratio = height/width
    # 将width=height/aspect_ratio带入 得到 height=sqrt(aspect_ratio) * scales
    # 将height=width * aspect_ratio带入  得到width=scales/sqrt(aspect_ratio)
    # 所以h_ratios = sqrt(aspect_ratio)   w_ratios=1/sqrt(aspect_ratio)
    # scales = [64, 128, 256]
	# aspect_ratios = [0.5, 1, 2]
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    # [r1, r2, r3]' * [s1, s2, s3]
    # number of elements is len(ratios)*len(scales)

    # 前面得到了宽和高对于面积scales的比例系数，下面相乘就能得到宽和高
    ## 矩阵乘法运算，得到宽度比例/高度比例 * anchor比例的结果
    ## 举例 w_ratios.shape=[3] w_ratios[:, None].shape=[3, 1]
    ## 举例 h_ratios.shape=[1] h_ratios[:, None].shape=[1, 1]
    ## 这里是矩阵相乘，根据公式生成Tensor(shape=[3,1])的tensor
    ## 后面的view(-1)相当于把结果展平，也就是将Tensor(shape=[3,1])变成Tensor(shape=[3])
    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)

    # 上一步得到了传入参数需要计算的所有anchor的宽和高，这创建的是anchor模板，模板的意思是原点是(0, 0)点
    # 知道了中心点和宽高，就可以计算出x_min、y_nmin、x_max、y_max 具体计算如下
    ## 注意后面的除2，前面计算出来得到的是anchor的宽和高，现在创建的是一个模板，所有的anchor中心点在原点
    ## 中心点在原点所以要除以2
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    # round 四舍五入  坐标必须是整数
    return base_anchors.round() 

scales = [64, 128, 256]
aspect_ratios = [0.5, 1, 2]

base_anchors = generate_anchors(scales, aspect_ratios)
print(base_anchors)
# 打印结果：
tensor([[ -45.,  -23.,   45.,   23.],
        [ -91.,  -45.,   91.,   45.],
        [-181.,  -91.,  181.,   91.],
        [ -32.,  -32.,   32.,   32.],
        [ -64.,  -64.,   64.,   64.],
        [-128., -128.,  128.,  128.],
        [ -23.,  -45.,   23.,   45.],
        [ -45.,  -91.,   45.,   91.],
        [ -91., -181.,   91.,  181.]])

```

> 在代码中，设置了三种尺寸与三种横纵比，所以我们得到的anchor模板中有9个anchor，每个anchor中的4个值分别表示𝑥𝑚𝑖𝑛,𝑦𝑚𝑖𝑛,𝑥𝑚𝑎𝑥,𝑦𝑚𝑎𝑥xmin,ymin,xmax,ymax。为什么会出现负值呢？并且负值都出现在𝑥𝑚𝑖𝑛xmin与𝑦𝑚𝑖𝑛ymin中。这是因为设置的anchor模板的中心点是在(0, 0)点，所以左上角在负值区，而右下角在正值区。不用担心负值，其他步骤的处理中，会对坐标为负值的anchor进行处理。
>
> 下面用代码将这组anchor呈现出来试一下

```python
import cv2
from PIL import Image
import numpy as np
.
image = cv2.imread('./timg.jpeg')
print('原图像尺寸：', image.shape)
anchor_numpy = base_anchors.numpy().reshape((-1))

'''
由于坐标存在负值，展示需要将负值转换成非负，所以这里取得最小的x和最小的y，在图像的左和上方向填充白色，再将坐标值进行变换
'''
# 获取最小的x坐标与y坐标
x_minimum = 0
y_minimum = 0
for i in range(0, len(anchor_numpy), 4):
    if anchor_numpy[i] < x_minimum:
        x_minimum = anchor_numpy[i]
    if anchor_numpy[i + 1] < y_minimum:
        y_minimum = anchor_numpy[i + 1]
# 将图像进行填充，填充颜色为白色
image = cv2.copyMakeBorder(image, -y_minimum, 0, -x_minimum, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

anchor_numpy = anchor_numpy.reshape((-1, 4))
for anchor in anchor_numpy:
    image = cv2.rectangle(image, (anchor[0] - x_minimum, anchor[1] - y_minimum), (anchor[2] - x_minimum, anchor[3] - y_minimum), (0, 255, 0))
display(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype('uint8')))
```

```python
# 打印结果：
原图像尺寸： (446, 650, 3)
```

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\3、One-Stage检测算法\images\下载.png)

## Anchor Free检测算法

### 1、理论

```python
# 听课笔记

```



### 2、代码

- ##### Yolo V2

```python
# 听课笔记：
1. Anchor in Yolo V2
	a. what is anchor (尽可能的去初始化贴合物体的bounding box形状)
    	1)、pre-defined virtual bboxes:预先设定好的虚拟边框，是个已知数
        2)、final bboxes are generated from them
        大resolution到小resolution直接可以卷积
        理解：铺满整张图像，框是怎么回归的呢？假如有预先设定好的边框，两个不同尺度的红色边框进行回归，看哪个更能拟合蓝色框；在图中进行地毯式的铺满白色×，预先设置好虚拟的边框，我们都不知道哪个尺度最适合物体形状，我们将所有可能的形状都排列出来，进行地毯式的排列，总有一个矮胖型或者瘦长型的边框拟合物体形状进行回归，这样我们就能很好地找到一个初始边框，回归就比较容易了。
	但是呢，还有个问题，如果图中有斜着的物体该怎么办呢？如果仅仅是用矮胖型或者瘦长型就不能满足了，为了回归斜着的物体，为什么不能设置成一个斜着的anchor呢？答案是当然可以的，我们可以设置成有角度的初始化anchor。
    但是如果有特别小的物体呢？那就再设置一个小尺度的anchor，所以可以设置成多尺度的anchor，即大中小不同角度的anchor，是为了拟合不同尺度的物体形状。
    依据物体的中心点，进行设置不同的anchor，
问题：
	假如很大的物体边上有一个很小的物体，这种情况应该怎么设置anchor呢？就是设置不同尺寸的anchor。
	b.anchor,ground truth ,truth bboxes prediction
    这部分理解看课件图片，牢记三个颜色的框，a,p,g
    t代表的是offset
    we hope a->g,but actually a generates p;
    so as long as p->g,we get a good result;
    so as long as the offset of p-a->g-a,we get a good result;
    so here we hope tp-->tg，t代表的是offset(p和a,g和a)
    以前都是回归坐标，但当我们引入了anchor之后，我们回归的都是偏差offset;
    在图中式子当中，哪些是未知的？哪些是已知的？只有绿色是已知的，红色也是已知的，所以蓝色predict是未知的；
    anchor完全可以聚类得到，anchor跟物体g在哪没有关系，跟我们设置的规则有关。
    anchor的生成只跟人设置的规则有关,predict的数量只跟ground truth有关,
    除以wa的原因就是，想要归一化；
  # 思考：###############################非常重要########################################
        @1.哪些数是网络算出来的？
		答案：tp是网络给的，预测出来的，但是注意：p对应的4个坐标值x,y,w,h不是已知的，是我们需要计算的。
        @2.哪些数是我们要手动算出来的？
		答案：tg需要我们通过ground truth和anchor的坐标计算出来
        @3.哪些数是已知的？
		答案：g的x,y,w,h是已知的
        @4.哪些数是我们想要得到的？
    	答案：p的x,y,w,h是我们想要计算得到的。
	这里，就开始看一下课件
2. anchor size and number
	a.Faster RCNN:9 by hands   # 注意区别
    b.Yolo v2: 5 by K-Means    # 注意区别     # YOLO V3是9个
3.anchor,truth bbox&predicted bboxes是如何结合的？
@1.这儿没听懂    # 第三遍过来听
10个数字（5个anchor，有俩w和h）
@2.Truth Bboxes:
    大写的W和H是原图的尺寸，f是final，(i,j)--grid ID对应都是x和y的整数部分，比如x=4.8,y=3.2,则i=4,j=3,此时的0-1是相对于cell来说的
    而第一步的0-1是相对于图像的
    减去i的原因没听懂（需要补课）
4.怎样数学化？
左边式子是引入cell后的式子，这页PPT有点难理解啊，啊啊啊啊啊要死了！！！！！！！！！！# 第三遍
这儿不行再去看一下之前看的那个博客看能理解不
Yolo还有一个规则就是：那个物体落到了哪个cell，那个cell就预测回归那个物体
使用sigmod的目的就是将tp压到0-1之间 
Cx和Cy是cell的id，是已知的
5.Problems
	@1.better for small & crowede objects

```

> 下图中，我们最想知道的是蓝色的p系列的值，g是人为标记好的位置，a是人为设置好的，所以a,p,g位置空间没有任何关系。
>
> a的生成只跟人有关系，a的尺寸大小只跟人有关系。
>
> two stage :通过神经网络生成一大堆region proposal
>
> one stage: 人为设定好生成的anchor(密集型生成).
>
> 公式中，都除以Wa，ha是为了抑制大物体的影响。
>
> 使用log函数也是为了抑制大物体对损失函数的影响。
>
> 公式中t代表的就是offset。
>
> encode的过程就是计算tg的过程，decode的过程就是计算p对应四个坐标值x,y,w,h的过程，深刻理解公式。
>
>

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_59.jpg)

![week1-4 Detection-3 stages_2021_60](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_60.jpg)

> ***对anchors,Truth BBoxes & Predicted BBoxes的理解***
>
> ------

> 1、Anchors:
>
> anchors:0.57273,0.677385,...,9.77025,9.16828[10 numbers]
>
> ​		aw0,       ah0,     .....,  aw4,ah4						# 因为有5个anchor
>
> 其中，anchors[0]=awi=(awori/w)*13  [not strict,just aiming to say how we get those numbers]--这里乘以13是因为最终的feature map是13x13，原始anchor的宽除以原始图像的宽。
>
> 2、Truth Bboxes:
>
> original bbox:[x0,y0,w0,h0]∈[0,W|H]
>
> 其中图中公式中的i和j代表的cell的ID也就是cell的行列数，还有就是i和j其实对应的是x和y的整数部分，所以这样理解的话，那么xf和yf就是x和y的小数点部分，那么进一步理解就是ground truth就是相对于每一个cell而言进行了归一化。这样就将cell有机的结合了一起。
>
> 具体见下图。
>
> 3、Predicted BBoxes:
>
> 详见下图。
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_86.jpg)
>
> ![week1-4 Detection-3 stages_2021_87](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_87.jpg)
>
> **将如上表述如何给数学化？**
>
> 如下图所示：
>
> 第二章图中，左边式子是引入cell之后的公式，右边是没引入cell之前的公式。
>
> 最下面的式子是对公式里面相关范围的说明。
>
> 左边式子中的tgx和tgy一定是在(0,1)之间的。
>
> 其中使用的σ（sigmod函数）是为了将其值压缩到0~1之间。
>
> 超出去无所谓，负数可以将其设置为0代码里面。
>
> code test is so important.
>
> YOLO系列算法一个cell一直是预测不了多个物体的。
>
> YOLO V2面试中一定要提的是anchor，so important.
>
>
>
>
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_88.jpg)
>
> ![week1-4 Detection-3 stages_2021_89](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_89.jpg)





- ##### Yolo V3

```python
# Yolo V3
1.improvements
	@1.change classification:  # important
        a.80 classes,from softmax--->logistic  # 这个是YOLO V3改良的，也是独有的。
# 逻辑理解：
分为三种：13x13,26x26,52x52,其中的逻辑要理解：就是格子打的越稀松就预测大物体，格子打的越密集就预测小物体，那意思就是13x13就是要预测大物体，26x26预测型物体，52x52预测小物体，意思明白了吧。

# softmax:
# 数值不稳定如何让它稳定，你得会求导，作业要好好做，你得知道怎么跟我们的高斯cross entropy结合,及之后的求导等等这些都得会。important
2.summary
	@1.output
    multi scale(指的是网络最后输出feature的大小)、label
3.FPN Net (feature pyramid network)    
面试四个方面：
	@1.传统machine learning
    @2.code test
    @3.CNN
    @4.传统的convelution

```

> **YOLO V3 采用的New Structure**:
>
> 如下图所示：改良版的resnet残差网络

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_91.jpg)

> **FPN** **Net**  [Feature Pyramid Network]----其实是个经典的U-Net网络
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_94.jpg)

### 3、作业:softmax---important，面试必备

#### 1、Coding

1. 【必做】使用numpy实现Softmax（假设2个样本，给定[[1, 2, 3], [2，1，3]]);
2. 【必做】使用torch.nn.functional.softmax() 验证numpy实现Softmax是否一致；
3. 【选做】了解目标检测xml的标注文件，提取目标框和图片长宽，把图像上的物体框出来。

使用浏览器打开文件【test_00000335.xml】

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701104808779.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)

#### 2、自行学习

++++++++++++++++ **以下不是作业，是额外的一些好东西，不用提交，自行学习** ++++++++++++++++ 

##### 一.PyTorch 中的易混 loss

Pytorch 中提供了很多类型的 loss (Criterion)，如 l1, l2, crossentropy 等。同时，同一种 criterion 可 能被包含在不同的功能模块下，如 torch.nn，torch.nn.functional 等。

这里，希望同学们能够区分和辨别其中的差异，以及他们之前相互的关联，不至以后感觉模糊。

1. 明确 F.xx 和 nn.xx (xx 表示某 criterion 如 cross_entropy 等)在使用时的区别

2. 明确各常用 F.xx 以及 nn.xx 的具体内容。可以从传入参数、数学形式、作用目的等不同方

   面总结。
    常用 criterion 需包括(但不限于[欢迎更多的总结]): F.softmax, F.cross_entropy, F.log_softmax, F.nll_loss, nn.CrossEntropyLoss, nn.NLLLoss, nn.LogSoftmax 等

##### 二.一些 criterion 的数学

 为进一步强化某些格外重要的 criterion 的理解，现补充一些数学求导，对面试很有帮助:

1. (数学意义上的)softmax 的求导。

2. (数学意义上的)cross_entropy+softmax 的求导。请注意，需包含 cross_entropy+softmax 的

   原始形式，以及求导后的结果。另外，请注意，下标的正确对应以及分类讨论, 以及分类 讨论后的整合。

```python
# 作业比较重要，下节课会讲，面试也会问到，认真完成
作业内容：

# 代码

def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)
 	
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
 	
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s
```



## Yolo V4讲解

### 1、理论

![]()

### 2、代码

```python 
# 听课笔记：
继续讲Yolo V3:上节课遗留内容
FPN Net:
    @1.featurized image pyramid:这里的SIFT特征点检测必须得掌握其原理
    @2.single feature map
    @3.pyramid feature hierarchy
    @4.feature pyramid network
其结构见课件
    
```

- soft-max

  ```python
  公式推导过程：
  见笔记本手写版
  
  ```

  #### Yolo V4

```python
# Improvements:
@1.data augmentation
	mosaic
@2.data augmentation:
    self-adversal
@3.CBN(了解)
@4.Modified SAM(这个比较重要--spatial(空间的) attention module):
    
@5.Modified PAN:
简单来讲，就是将两个FPN每层对应拼接到一起，之前是直接加到一起，而现在是concatenation到一起。
# Attention Module:
大概意思就是让图像注意到图像中哪部分最重要
# @1.spatial attention module(平面上的):
    对应的物体的具体位置；记住有个切片的过程，最后会有个weight matrix(0~1之间的权重)重在理解，通过sigmod函数然后会产生0和1的一个权重矩阵，然后在乘回去，这样我们就能知道哪些空间信息是重要的，哪些信息是不重要的。
# 理解：
	maxpool以前是就是将每一维度上的矩阵抽出来进行卷积运算得到一个值(或者说是一个点)，结果会变成一个向量，但是在这里的maxpool：它是在竖着的方向进行分层成每一片，然后去做maxpool。
# @2.channel attention(channel维度的，竖着的)：
	最后会有个weight vector(0~1之间的权重),通过sigmod函数然后会产生0和1的一个权重矩阵，然后再乘回去，这样就能知道哪个channel更加重要
图中MLP:multiple layer perception------多层感知机
    
    
# Modified SAM:
直接在tensor上sigmod一下
```

> **attention module**:
> 主要意思就是它想告诉我们图像中哪部分更重要
>
>
>
> ![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_98.jpg)

代码讲解

```python
adamw了解下
burn_in=warm up=1000
就是从0增加到1000就会达到0.001的学习率
shortcut层linear:在这里起到占位符的作用
9个anchor：52x52;26x26;13x13
ignore_thresh=0.7 will be reset to 0.5 in models.py
[route]
layers=-1,36意思就是将上面的一层和我这层连起来就是我这层的输出
layers=-4的理解，往上数四层网络
YOLO上面的网络没有Batch norm

x = torch.sigmoid(prediction[..., 0]) # 85维的第0维 # Center, tx   # (b,3,13,13)            # 1 +
y = torch.sigmoid(prediction[..., 1]) # 85维的第1维 # Center, ty   # (b,3,13,13)            # 1 +
w = prediction[..., 2]  # Width，tw   # (b,3,13,13)            # 1 +
h = prediction[..., 3]  # Height,th       # (b,3,13,13)            # 1 +
pred_conf = torch.sigmoid(prediction[..., 4])   # confidence(两个概率的乘积在0~1之间)      # Conf (b,3,13,13)            # 1 + = 5 +
pred_cls = torch.sigmoid(prediction[..., 5:]) 


pred_boxes[..., 0] = x.data + self.grid_x  # bx
pred_boxes[..., 1] = y.data + self.grid_y  # by
pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  # bw
pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h	# bh
具体去看公式，通过已知公式反推出来的

anchor为什么要view下，因为形状对不上没法计算
```

YOLO V3网络结构请参考这个： https://blog.csdn.net/dz4543/article/details/90049377

```python
# nn.ModuleList() VS nn.Sequential()
@1、nn.ModuleList(): 就是Module的list，并没有实现forward函数(并没有实际执行的函数)，所以只是module的list，并不需要module之间的顺序关系
@2、nn.Sequential(): module的顺序执行。是实现了forward函数的，即会顺序执行其中的module，所以每个module的size必须匹配
说的不错的链接：https://blog.csdn.net/watermelon1123/article/details/89954224
             https://zhuanlan.zhihu.com/p/64990232
```



### 3、作业

```python
# 跑通YOLO V3代码

```





## 检测算法技巧(一)+Yolo V3代码讲解

### 1、理论

![]()

### 2、代码

```python
# 听课笔记：20210724
>>RetinaNet
@1.structure
	Resnet+FPN+FCN
    结构图见课件
>>Focal Loss: ########################重点#####################################
    Q.why one-stage performs worse than tow stage?
    A.1.Beause net/pos samples are extremely unbalanced
      2.Gradient is dominated by easy samples.	# 想那张图，狗自行车那张，负样本占多(背景)，所以学的多了负样本背景。
    S.We can use FOCAL LOSS to solve it.
>>公式见课件（本业PPT(见下图)及其important，必须得背下来------#老师说要背下来#------------）    
	αt和γ伽玛作为超参数
>>SSD(Single shot detection) --参考学习了解   
Faceboxes:该页PPT对应的文章可以参考看一下,对应的链接可以点进去了解学习一下
    
```

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_104.jpg)

```python
# ANCHOR FREE的学习
>>Trend
	@1.lots of hyperparametres:sizes,aspect-ratios,number of anchors,....
    @2.hard to generalize:different datasets have different data shape,need to redesign
    @3. difficult to train:numbalanced positive/negative samples
    @4. complex calculation
    @5.myriads of redundancy
>>anchor free net is a trend
	@1.cornerNet(了解)
    @2.FoveaBox(了解)
    @3.CenterNet(学习)
    @4.FCOS(学习)
>> CenterNet 
	>features
        @1.more accurate & faster
        @2.detect "object as points"[only detect center points]
        @3.multiple functions in one structure
        @4.no need for post-processing
 进行了高斯模糊
	>structure:whole
        该架构的普适性特别强
        backbone->neck->head
        结构见课件
任何检测网络：先localization,然后classification

```

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_108.jpg)

![week1-4 Detection-3 stages_2021_109](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_109.jpg)

![week1-4 Detection-3 stages_2021_110](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_110.jpg)

![week1-4 Detection-3 stages_2021_111](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_111.jpg)

![week1-4 Detection-3 stages_2021_112](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_112.jpg)

![week1-4 Detection-3 stages_2021_113](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_113.jpg)

![week1-4 Detection-3 stages_2021_114](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_114.jpg)

> one class one channel,图中80就是80个class或者80个channel，最上面的公式其实是在做一个高斯模糊，
>
> 对每一个pixel进行了focal loss，作reference的时候，我们只来个3x3的maxpool找到最大的点，而不作NMS。

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_115.jpg)

> YOLO是在小feature上面做检测的，如果乘回去(这里是4倍)到原图，必然是有偏差的，为了缓解计算机硬件量化有偏差的问题，所以这里直接进行了预测offset。
>
> label怎么来的？P/R就是小数，而P~是整数。

![week1-4 Detection-3 stages_2021_116](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_116.jpg)

> 上面size这一步就是预测物体的size，这里乘以2是x和y的。

![week1-4 Detection-3 stages_2021_117](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_117.jpg)

![week1-4 Detection-3 stages_2021_118](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_118.jpg)

![week1-4 Detection-3 stages_2021_119](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_119.jpg)

> CenterNet的缺点就是会产生很多channel，



![week1-4 Detection-3 stages_2021_120](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_120.jpg)

> FCOS不是特别重要

![week1-4 Detection-3 stages_2021_121](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_121.jpg)

> head就是要实现不同的功能

![week1-4 Detection-3 stages_2021_122](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_122.jpg)

> pi就是预测哪一层，决定物体落在哪一层，决定物体预测的功能。
>
> m是层数或者说是阈值都可以。m后面的数字是ID，第几个。
>
> 从2x2反推回去4x4，对应的点应该在哪。

![week1-4 Detection-3 stages_2021_123](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_123.jpg)

> 采用了独特的IOU Loss.

![week1-4 Detection-3 stages_2021_124](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_124.jpg)

![week1-4 Detection-3 stages_2021_125](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_125.jpg)

![week1-4 Detection-3 stages_2021_126](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_126.jpg)

![week1-4 Detection-3 stages_2021_127](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_127.jpg)

> center-ness，希望中心真就是在中心，可以理解为中心性。
>
> 为什么用这种方式就能计算这个呢？如果在中心点的话，center-ness的值最大值为1。

![week1-4 Detection-3 stages_2021_128](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\2、项目介绍及One-Stage检测算法\week1-4 Detection-3 stages_2021\week1-4 Detection-3 stages_2021_128.jpg)

> BCE:binary cross entropy

### 3、作业

> **YOLO V3网络结构的学习**
>
> 参考如下链接：
>
> https://blog.csdn.net/dz4543/article/details/90049377
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/2019050919344358.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6NDU0Mw==,size_16,color_FFFFFF,t_70)
>
> **网络输出**：
>
> ```python
> layer     filters    size              input                output
>    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32 0.299 BF
>    1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64 1.595 BF
>    2 conv     32  1 x 1 / 1   208 x 208 x  64   ->   208 x 208 x  32 0.177 BF
>    3 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64 1.595 BF
>    4 Shortcut Layer: 1
>    5 conv    128  3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x 128 1.595 BF
>    6 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64 0.177 BF
>    7 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128 1.595 BF
>    8 Shortcut Layer: 5
>    9 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64 0.177 BF
>   10 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128 1.595 BF
>   11 Shortcut Layer: 8
>   12 conv    256  3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 256 1.595 BF
>   13 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>   14 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>   15 Shortcut Layer: 12
>   16 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>   17 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>   18 Shortcut Layer: 15
>   19 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>   20 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>   21 Shortcut Layer: 18
>   22 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>   23 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>   24 Shortcut Layer: 21
>   25 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>   26 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>   27 Shortcut Layer: 24
>   28 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>   29 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>   30 Shortcut Layer: 27
>   31 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>   32 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>   33 Shortcut Layer: 30
>   34 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>   35 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>   36 Shortcut Layer: 33
>   37 conv    512  3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 512 1.595 BF
>   38 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   39 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   40 Shortcut Layer: 37
>   41 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   42 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   43 Shortcut Layer: 40
>   44 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   45 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   46 Shortcut Layer: 43
>   47 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   48 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   49 Shortcut Layer: 46
>   50 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   51 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   52 Shortcut Layer: 49
>   53 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   54 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   55 Shortcut Layer: 52
>   56 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   57 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   58 Shortcut Layer: 55
>   59 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   60 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   61 Shortcut Layer: 58
>   62 conv   1024  3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x1024 1.595 BF
>   63 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
>   64 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
>   65 Shortcut Layer: 62
>   66 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
>   67 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
>   68 Shortcut Layer: 65
>   69 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
>   70 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
>   71 Shortcut Layer: 68
>   72 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
>   73 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
>   74 Shortcut Layer: 71
>   75 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
>   76 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
>   77 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
>   78 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
>   79 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
>   80 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
>   81 conv     18  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x  18 0.006 BF
>   82 yolo
>   83 route  79
>   84 conv    256  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 256 0.044 BF
>   85 upsample            2x    13 x  13 x 256   ->    26 x  26 x 256
>   86 route  85 61
>   87 conv    256  1 x 1 / 1    26 x  26 x 768   ->    26 x  26 x 256 0.266 BF
>   88 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   89 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   90 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   91 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
>   92 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
>   93 conv     18  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x  18 0.012 BF
>   94 yolo
>   95 route  91
>   96 conv    128  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 128 0.044 BF
>   97 upsample            2x    26 x  26 x 128   ->    52 x  52 x 128
>   98 route  97 36
>   99 conv    128  1 x 1 / 1    52 x  52 x 384   ->    52 x  52 x 128 0.266 BF
>  100 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>  101 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>  102 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>  103 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
>  104 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
>  105 conv     18  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x  18 0.025 BF
>  106 yolo
> 
> ```
>
>
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/2019050922171990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6NDU0Mw==,size_16,color_FFFFFF,t_70)
>
> 在前文网络的基础上，用红色做了注释。residual使用残差结构。什么是残差结构？举个例子在第一层残差结构（其输出为208208128），其输入为20820864，经过3211和6433的卷积后，其生成的特征图与输入叠加起来。其结构如下：
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190509222449333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6NDU0Mw==,size_16,color_FFFFFF,t_70)
>
> 其叠加后的特征图作为新的输入输入下一层。YOLO主体是由许多这种残差模块组成，减小了梯度爆炸的风险，加强了网络的学习能力。
>
> 可以看到YOLO有3个尺度的输出，分别在52×52，26×26，13×13。嗯，都是奇数，使得网格会有个中心位置。同时YOLO输出为3个尺度，每个尺度之间还有联系。比如说，13×13这个尺度输出用于检测大型目标，对应的26×26为中型的，52×52用于检测小型目标。上一张图，我觉得很详细看得懂。
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190509224534499.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R6NDU0Mw==,size_16,color_FFFFFF,t_70)





## 检测算法技巧(二)

### 1、理论

![week6-7 Advanced Detection Tricks_01](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_01.jpg)

检测算法的小技巧：---面试官可能会问：有什么技巧可以让我们的检测算法变的更好呢？这特么**我面试的时候还真的问了**！！！！！！！

考虑的角度/思路：**思路很重要**

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\思路.jpg)

> 1、input输入部分--->数据增强data augmentation

> 2、network中间网络部分--->regularization正则化+activitation function激活函数；引入attention module ；layer->module->net

> 3、output输出部分--->loss函数，label---disturbed label（指鹿为马）

> 4、train网络训练相关部分--->比如优化器
>
> 防止网络过拟合的时候也可以从上面4部分来考虑

![week6-7 Advanced Detection Tricks_02](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_02.jpg)

![week6-7 Advanced Detection Tricks_03](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_03.jpg)

- disturbed label:指鹿为马故意label错，进行打乱。

![week6-7 Advanced Detection Tricks_04](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_04.jpg)

将两张图混合揉到一起。2张效果是最好的。MixUp:分类有效。专门针对分类的

gradient descent

一次只处理一张图的话，我们叫SGD--stochastic gradient descent

顺便回顾下GD知识：

> 1）Batch gradient descent(批量梯度下降)
>   在整个数据集上
>   每更新一次权重，要遍历所有的样本，由于样本集过大，无法保存在内存中，无法线上更新模型。对于损失函数的凸曲面，可以收敛到全局最小值，对于非凸曲面，收敛到局部最小值。
>   随机梯度下降（SGD）和批量梯度下降（BGD）的区别。SGD 从数据集中拿出一个样本，并计算相关的误差梯度，而批量梯度下降使用所有样本的整体误差：「关键是，在更新中没有随机或扩散性的行为。」
>
> 2）stochastic gradient descent(SGD,随机梯度下降)
>   可以在线学习，收敛的更快，可以收敛到更精确的最小值。但是梯度更新太快，而且会产生梯度震荡，使收敛不稳定。
>   随机梯度下降（SGD）和批量梯度下降（BGD）的区别。SGD 从数据集中拿出一个样本，并计算相关的误差梯度，而批量梯度下降使用所有样本的整体误差：「关键是，在更新中没有随机或扩散性的行为。」
>
> 3）Mini-batch gradient descent(MBGD,小批量梯度下降)----**目前最常用的**
>   批量梯度下降算法和随机梯度下降算法的结合体。两方面好处：一是减少更新的次数，使得收敛更稳定；二是利用矩阵优化方法更有效。
>   当训练集有很多冗余时（类似的样本出现多次），batch方法收敛更快。
>   以一个极端情况为例，若训练集前一半和后一半梯度相同。那么如果前一半作为一个batch，后一半作为另一个batch，那么在一次遍历训练集时，batch的方法向最优解前进两个step，而整体的方法只前进一个step。
> ————————————————
> 原文链接：https://blog.csdn.net/wydbyxr/article/details/84822806

那我们用这个mixup是将所有的数据打乱再去SGD呢？还是我们用一次SGD打乱一次呢？

作者说：我们用第二种方法--用一次打乱一次。就是图中说得Mixed within one batch is enough

将不同的标签混合到一起效果要好一点。比如讲兔子和哈士奇揉到一起等等，实验得知：相同的label里面揉的话对于网络的性能没什么提升。

真正实现上有2种方法---代码里会讲。



![week6-7 Advanced Detection Tricks_05](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_05.jpg)

问题：

1. 揉2张图效果是最好的，其他更多的图揉进行去的话产生的效果都差不多；
2. 在每一个batchsize里面用的时候打乱，没有必要将所有的数据集进行打乱mixup操作；
3. 一个是类间揉，一个是类内揉，实验证明肯定是类间揉方法比较好；类内揉没有什么提升
4. mixup实现方式有2种；具体看代码
5. 这种方法主要是针对于分类任务的。

问题：

​	那揉完标签怎么办？

​	代码部分来讲







![week6-7 Advanced Detection Tricks_06](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_06.jpg)

cutout既可以用在classification也可以用在detection当中。

normalize:将图像变灰

应该怎么去挖它呢？什么样的形状会更好呢？

实验结果表明：挖的形状(shape)不重要，但是挖多少(size)却比较重要

以前的方法：rectangle:random erasing

现在的方法：**GridMask**(improvment)

具体实现方法看代码。

![week6-7 Advanced Detection Tricks_07](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_07.jpg)

CutmMix:   ---针对于检测+分类任务





![week6-7 Advanced Detection Tricks_08](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_08.jpg)

减的时候不能小于0，所以min=0

加的时候不能加到图像外面去，所以max=W/max=H

target=lambda x target+(1-lambda) x target_s：就是个混合型label

**这里target就是label的意思**

这里lambda是个比例，其实是做了一个weight权重

代码讲解部分具体见代码资源。

permutation:排列问题

就是个weighted loss

uniform distribution

![week6-7 Advanced Detection Tricks_09](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_09.jpg)

YOLOV4里面使用了这种方法

这种操作相当于增加了batchsize，无形中增加了物体的形状多样性

实现方式较多代码里我们只看其中一种方法；看的话结合下面这个图来看

![week6-7 Advanced Detection Tricks_10](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_10.jpg)

![week6-7 Advanced Detection Tricks_11](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_11.jpg)

![week6-7 Advanced Detection Tricks_12](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_12.jpg)

![week6-7 Advanced Detection Tricks_13](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_13.jpg)

> Derivation这部分自己来做
>
> 为什么label smoothing效果会更好一些？
>
> 答案：通过公式来看，yi是真正label的值，真正做的事就是把我们的label值进行改变，0也不是0了会变成0.0几，1也不是1了，而是0.9几。

![week6-7 Advanced Detection Tricks_14](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_14.jpg)

> 总有公式中最后一个值：∈H(u,p)这个小值将它拉到左边，没有那么容易接近于0,从而使得曲边变得smooth一点。

![week6-7 Advanced Detection Tricks_15](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_15.jpg)

> 以前传统意义上的卷积层最后两层一般为FC层，但是人们发现FC层的参数太多了，此时人们就想到了一个办法：dropout，就是我们真正做connection的时候我们不要做全连接，我们随机的去掉一部分连接，这种方法是这种线性的vector是有效的，但是对于这种tensor的结构你随便删除就不行，就是你在上图中随机的删除一些点没用，没啥效果，后来人们发现，对于convolution的这种操作(因为convloution的这种操作是划窗的方法)，他们会有shared local info这种操作，所以这样的话就没有作用。local information sharing
>
> 那么人们就想到了一种方法：DropBlock,就是我们要消灭周围所有的点，本来是要删除一个点的，这样我们就把这个点周围所有的点都给删了。
>
> 万一给消灭没了咋整？所以消灭的量还是很重要的
>
> 看下图：
>

![week6-7 Advanced Detection Tricks_16](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_16.jpg)

> Mij矩阵符合伯努利分布
>
> M里面有些点是设置成0的
>
> 这里的伽玛值 very important

![week6-7 Advanced Detection Tricks_17](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_17.jpg)

> +1的操作就是为了处理离散的问题。不是连续域
>
> (feature_size-block_size)的原因就是防止取的点的极值出界出了黄色区域
>
> (feature_size-block_size+1)就是绿色的区域
>
> block_size^2就是蓝框的范围
>
> 伽玛值是我们真正要设置的threshold值
>
> **代码中block_size的值一般为5/7**，为什么？
>
> 答案：block_size=(kernel_size/2)x2+kernel_size
>
> d x (f^2)=γ x (b^2) x (g^2)  --->d x(f^2/b^2)=γ x g^2
>
> 我们的目的就是去除掉有关于这个点的所有的点，只要你卷积卷积到的，为了删除掉蓝色的点，从x点就已经有点了。
>
> 理解：
>
> ​	drop_rate:相对于整体的drop率，比如说10%=0.1
>
> ​	绿框：就是我们的中点可以取的范围
>
> ​	伽玛值是***[中点]()*(center points)的drop率**
>
> ​	伽玛值到时候代码要算出来的,伽玛值只跟中点相关。
>
> 已知的值：drop_rate或者keep_prob
>
> 伽玛值会被求出来。

![week6-7 Advanced Detection Tricks_18](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_18.jpg)

![week6-7 Advanced Detection Tricks_19](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_19.jpg)

> 0处不可导，信息有损失<0的时候
>
> None-zero centered指的是数值没有跨越正负半轴
>
> 在BP的时候回溯的路径会走之字形，这样的话路径就变长了，这样就会比较慢了。
>
> 这样可能会陷入到局部最优解而不是全局最优解
>
> sigmod函数也有这种情况
>
> 为什么会出现zigzag这种情况呢？
>
> 看下图：
>
> 激励之后所有的xi都是大于0的
>
> 要理解下面这个公式
>
> 一个很好地面试题。

![week6-7 Advanced Detection Tricks_20](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_20.jpg)

relu(wx+b)，所以x只能大于0

cons: 缺点

pros：优点

。x=0处不可导的缺点就是：

​	很可能会陷入一个局部最优解

。Non-zero centered:主要是我们的函数没有跨越正负半轴，**如果正负半轴都有值就是zero-centered**

   Non-zero centered的缺点就是BP的时候路径更长，更容易陷入局部最优解，不容易走到全局最优解。

。为什么Non-zero centered会出现zigzag?

   具体详见上面公式推导。只能每次通过走zigzag形可以使得Wi的值保持是同正或者同负

。负半轴导数为0，信息有损失

sigmod函数也是Non-zero centered

由于最终我的任何一个权重它所要更新的方向只取决于我的loss，关于这个f的方向，所以导致所有的weight更新的方向是一致的，或者同正或者同负，这样就会导致zigzag，导致我们的训练周期变长，还有就是路程多了，这样就有更大的概率掉入局部最优解。再配合relu本身，走着走着不小心就“掉坑里了”。【思考与理解】

建议：

自己在工程中采集的数据集的话，严肃的去审核标注的数据，尽量不要有标注错误的图



sigmod(none-zero centered)和tanh(zero-centered)函数的缺点：就是真正有值的时候函数曲线基本接近线性，所以非线性拟合能力较差;

https://zhuanlan.zhihu.com/p/276186049这个博客有参考价值



relu函数为什么叫非线性激活函数？

本身负半轴为0，正半轴为线性函数，但是组合起来整体就是非线性函数，因为不满足f(ax1)+f(ax2)!=af(x1)+bf(x2)

![week6-7 Advanced Detection Tricks_21](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_21.jpg)

右边这张图就是LSTM的结构图，这儿如果不懂了可以去看核心课高老师讲的这个算法及课件比较详细

LSTM:如何在时域获得更好的一个结果，NLP里面运用的广泛

Alternative:改良



![week6-7 Advanced Detection Tricks_22](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_22.jpg)

求导不难，sigmod的求导必须得很清楚啊

。大于0的时候没有界限：unbounded above -->avoid saturation

。bounded below:非线性会非常好

。No monotonicity：非单调性

。smoothness: 不容易陷入局部最优解

saturation：饱和

monotonicity：单调性

![week6-7 Advanced Detection Tricks_23](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_23.jpg)

Mish:是由三个函数耦合的

swish：是由x和sigmod函数组成的。

![week6-7 Advanced Detection Tricks_24](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_24.jpg)

![week6-7 Advanced Detection Tricks_25](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_25.jpg)

![week6-7 Advanced Detection Tricks_26](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_26.jpg)

下面总结那块，就是在检测模型当中，多个bbox可能和GT有相同的loss，但是位置不一样，比如对称啥的，最后计算总的loss就都一样，二者相等。使用下面IOU loss可以解决。

![week6-7 Advanced Detection Tricks_27](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_27.jpg)

TSD:task-aware spati

![week6-7 Advanced Detection Tricks_28](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_28.jpg)

![week6-7 Advanced Detection Tricks_29](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_29.jpg)

1、IOU为0了，如上图左下角图

2、交集形状不一样但是面积却一样，比如右上角的图像真实框和预测框形状更加类似，但是下面的就不太一样了

所以为了解决这种情况，人们想到了另一种方法：GIOU-G指的是global

![week6-7 Advanced Detection Tricks_30](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_30.jpg)

merge inteval:一道经典算法题

**inter-area计算**：简便记忆---**左大右小**

。左上角xy坐标都取max

。右下角xy坐标都取min

符合条件的计算，不符合条件的直接给0

蓝色框是引入蓝色的虚拟的框。

整个计算过程都要理解。

GIOU都是正的。

**要明确每种loss解决了哪种问题**

![week6-7 Advanced Detection Tricks_31](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_31.jpg)

G:generate

当两个物体相互包含的时候GIOU也是无能为力。

由图可知，显然DIOU效果是最好的。

![week6-7 Advanced Detection Tricks_32](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_32.jpg)

计算了一个中心点的loss

Diou=1-IOU+(d^2)/(c^2)

平方只是为了保证我们的d是正数

目的就是计算比值

如果比值越趋近于0说明两个物体越近

d: distance

![week6-7 Advanced Detection Tricks_33](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_33.jpg)

![week6-7 Advanced Detection Tricks_34](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_34.jpg)

DIOU loss也是有缺陷的：

​	如果像上图这种情况，DIOU也是解决不了的。

​	但是第二个图更好一点，因为长得最像：长宽比能解决这个问题 aspect ratio。

长宽比如何解决，请看下面公式：

​	

![week6-7 Advanced Detection Tricks_35](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_35.jpg)

C: complete

the more IOU is ,the more the α will be little,the more important the v will be.

gt就是ground truth

**v是aspect ratio的loss**

α阿发是权重程度值，跟IOU有关系，离得近(IOU越大)则阿发越大，V越重要。

YOLOV3的paper当中loss是smooth L1 LOSS

![week6-7 Advanced Detection Tricks_36](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_36.jpg)

![week6-7 Advanced Detection Tricks_37](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\6、检测算法技巧(一)\week6-7 Advanced Detection Tricks\week6-7 Advanced Detection Tricks_37.jpg)





### 2、代码

```python
# 听课笔记：2021-08-07
# 请问有什么方法可以让检测变得更好？
disturbed label(相当于在input里面引入噪声)
# 本节课pipeline
1、Data Augmentation
	@1、mixed up
    @2、coutout
    @3、cutmix
    @4、mosaic
2、Regularization---让算法更加鲁棒
	@1、Label Smoothing
    @2、BropBlock
3、Activation Function
	@1、Relu
    @2、Swith
    @3、Mish
4、Loss
	@1、
    @2、
    @3、

# 1、Mixup
Features:
    a、mixed up to 2 images is good enough
    [3 or more,similar effect,more time]
    b、mixed within one batch is enough
    [no need to mix within the whole dataset:same effect]
    c、better to mix within different labeled images
    [no promotion within same labels]
    所以这种情况标签应该怎么标呢？？？
    d、two types of implementation
    [lets see code later]
    e、Mainly for classification

# 2、cutout
Feature:
    a、Normalize first,then cutout
    [reduce the potential effect]
    b、(size>shape)of the cutout patch
    cut square patch;
    rectangle:random erasing
    c、too simple,might erase useful info
经典的面试题(类似这种code_test必须得会的)：
vector/list=[1,2,3,4,5],all permutations?[[2,1,3,4,5]...]
set:
 sigmoid也是None-zero centered
 

```



### 3、作业



### 4、补充学习--MobileNet

![](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210624_233834_com.kaikeba.android.jpg)

![Screenshot_20210706_080611_com.kaikeba.android](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210706_080611_com.kaikeba.android.jpg)

![Screenshot_20210717_162018_com.kaikeba.android](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210717_162018_com.kaikeba.android.jpg)

![Screenshot_20210724_094455_com.kaikeba.android](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210724_094455_com.kaikeba.android.jpg)

![Screenshot_20210806_084823_com.kaikeba.android](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210806_084823_com.kaikeba.android.jpg)

![Screenshot_20210806_124712_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210806_124712_tv.danmaku.bili.jpg)

![Screenshot_20210814_105307_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210814_105307_tv.danmaku.bili.jpg)

![Screenshot_20210814_110851_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210814_110851_tv.danmaku.bili.jpg)

![Screenshot_20210815_000720_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_000720_tv.danmaku.bili.jpg)

![Screenshot_20210815_000921_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_000921_tv.danmaku.bili.jpg)

![Screenshot_20210815_001021_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_001021_tv.danmaku.bili.jpg)

![Screenshot_20210815_001223_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_001223_tv.danmaku.bili.jpg)

![Screenshot_20210815_001523_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_001523_tv.danmaku.bili.jpg)

![Screenshot_20210815_001610_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_001610_tv.danmaku.bili.jpg)

![Screenshot_20210815_001639_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_001639_tv.danmaku.bili.jpg)

![Screenshot_20210815_001846_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_001846_tv.danmaku.bili.jpg)

![Screenshot_20210815_001937_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_001937_tv.danmaku.bili.jpg)

![Screenshot_20210815_002144_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_002144_tv.danmaku.bili.jpg)

![Screenshot_20210815_002415_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_002415_tv.danmaku.bili.jpg)

![Screenshot_20210815_002517_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_002517_tv.danmaku.bili.jpg)

![Screenshot_20210815_002604_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_002604_tv.danmaku.bili.jpg)

![Screenshot_20210815_002635_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_002635_tv.danmaku.bili.jpg)

![Screenshot_20210815_002849_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_002849_tv.danmaku.bili.jpg)

![Screenshot_20210815_002938_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_002938_tv.danmaku.bili.jpg)

![Screenshot_20210815_003127_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_003127_tv.danmaku.bili.jpg)

![Screenshot_20210815_003250_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_003250_tv.danmaku.bili.jpg)

![Screenshot_20210815_003524_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_003524_tv.danmaku.bili.jpg)

![Screenshot_20210815_003541_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_003541_tv.danmaku.bili.jpg)

![Screenshot_20210815_003758_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_003758_tv.danmaku.bili.jpg)

![Screenshot_20210815_003834_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_003834_tv.danmaku.bili.jpg)

![Screenshot_20210815_003948_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_003948_tv.danmaku.bili.jpg)

![Screenshot_20210815_004030_tv.danmaku.bili](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\images\MobileNet\Screenshots\Screenshot_20210815_004030_tv.danmaku.bili.jpg)



## 算法加速

### 1、理论

![week7-8 Acceleration_01](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_01.jpg)

![week7-8 Acceleration_02](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_02.jpg)

**Network Design必须会**

![week7-8 Acceleration_03](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_03.jpg)

#### 1、Network Design

![week7-8 Acceleration_04](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_04.jpg)

训练之前有什么方法可以让我们的训练速度变快一点；

训练当中有什么方法可以让我们的训练速度变快一点；

训练之后有什么方法可以让我们的训练速度变快一点；

![week7-8 Acceleration_05](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_05.jpg)

训练之前加速方法

![week7-8 Acceleration_06](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_06.jpg)

训练当中的加速方法

![week7-8 Acceleration_07](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_07.jpg)

很重要的网络设计方法：

**MobileNet Series**网络必须得会

![week7-8 Acceleration_08](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_08.jpg)

红色部分必须掌握；

ASPP(隐身自SPP)有时间看下，属于neck部分

##### MobileNet V1

![week7-8 Acceleration_09](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_09.jpg)

N是卷积核的个数

单独的，没有串联起来信息

1/N可以忽略不计，因为N太大了了，但是1/Dk^2不能忽略

左边是原始计算方法，右边是加速方法

**优缺点：**

从图中可以看出，速度明显加快了8~9倍但是同时性能也会变弱，因为参数变少了函数的拟合能力变弱了。

Step2的输入：Df x Df x M是Step1的输出。

标准卷积：

![77ef87952eb9b445ba6bcad2457ebdbf.png](https://img-blog.csdnimg.cn/img_convert/77ef87952eb9b445ba6bcad2457ebdbf.png)

1x1卷积好像理解了：参考下文好好理解

https://blog.csdn.net/weixin_39589394/article/details/111382263

**三图看懂深度可分离卷积：**

![4da9a43f11f777bdc2fac0e53bbbde82.png](https://img-blog.csdnimg.cn/img_convert/4da9a43f11f777bdc2fac0e53bbbde82.png)

![d17ff8f87d37f59dfe38251057150b75.png](https://img-blog.csdnimg.cn/img_convert/d17ff8f87d37f59dfe38251057150b75.png)

![85c96e9b8bbf8f30e5f8fa298cb4a4e0.png](https://img-blog.csdnimg.cn/img_convert/85c96e9b8bbf8f30e5f8fa298cb4a4e0.png)

![week7-8 Acceleration_10](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_10.jpg)

##### **MobileNet V2**

![week7-8 Acceleration_11](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_11.jpg)

![week7-8 Acceleration_12](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_12.jpg)

**叫Linear的原因**：最后没有使用 Relu激活函数(为了增加非线性拟合能力)

![week7-8 Acceleration_13](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_13.jpg)

哈哈上面这个人话说得真人话。通俗易懂

![week7-8 Acceleration_14](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_14.jpg)

##### MobileNet V3

![week7-8 Acceleration_15](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_15.jpg)

SENet: Squeeze-Excitation Block

IOUNet:

GIOUNet:

![week7-8 Acceleration_16](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_16.jpg)

1、Fsq：是通过global average pooling得到的

2、Fex:   具体操作见下图-----Excitation：激励激发

3、SENet其实就是个attention module。





![week7-8 Acceleration_17](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_17.jpg)

scale: 就是做乘法的过程，权重vector乘以tensor

sigmod好处就是每一个输出都是一个分类，但是如果是softmax的话就是输入100个，这个100个输出结果和为1，只有一个分类，，，意思懂了吧，这儿只是大概记录一下，不太清楚可能。就像YOLO V3里面用sigmod的目的也是这。

![week7-8 Acceleration_18](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_18.jpg)

![week7-8 Acceleration_19](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_19.jpg)

![week7-8 Acceleration_20](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_20.jpg)

h:hard

**为什么要使用Relu6呢？目的是为了加速但是效果没有实现加速**

因为为了加速我们还要考虑其他因素。

答案：留个悬念后面讲

![week7-8 Acceleration_21](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_21.jpg)

##### **ShuffleNet V1**

![week7-8 Acceleration_22](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_22.jpg)

![week7-8 Acceleration_23](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_23.jpg)

分组卷积的好处：计算量减少，在每一个分组内进行convolution,但是有个缺点，每部分只能预测到原图一部分区域，损失了原图各像素之间的联系，所以作者又想到了一个办法：那就是channel shuffle，这样的话，这样每一组就能获得tensor的完整的信息，每一组的输出就是一个“小象”--具有完整信息

fractional: adj. 部分的；[数] 分数的，小数的

![week7-8 Acceleration_24](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_24.jpg)

fractional:部分的

![week7-8 Acceleration_25](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_25.jpg)

##### ShuffleNet V2

![week7-8 Acceleration_26](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_26.jpg)

![week7-8 Acceleration_27](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_27.jpg)

输入和输出channel相等的时候速度最快，所以要尽量减少channel数的变化。

fragmentation：碎片

parallelism:平行，类似，对应

eltwise:element wise很吃资源的

negligible:微不足道的可以忽略的。

![week7-8 Acceleration_28](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_28.jpg)

![week7-8 Acceleration_29](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_29.jpg)

element wise操作：点对点操作，比如add，或者乘法，都是非常吃资源的。

![week7-8 Acceleration_30](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_30.jpg)

1. concat:维度上进行拼接，消耗资源相对少一点。
2. add(element wise):resnet里面也是这种操作，由图中可知，消耗资源极大。

![week7-8 Acceleration_31](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_31.jpg)

使用了channel split操作

![week7-8 Acceleration_32](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_32.jpg)

##### EfficientNet

前言：并不非人为设计的网络，用计算机搜索来做的网络，简单了解即可。

![week7-8 Acceleration_33](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_33.jpg)

了解一下这个EfficientNet

![week7-8 Acceleration_34](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_34.jpg)

![week7-8 Acceleration_35](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_35.jpg)

![week7-8 Acceleration_36](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_36.jpg)

##### GhostNet

前言：华为提出的网络

![week7-8 Acceleration_37](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_37.jpg)

![week7-8 Acceleration_38](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_38.jpg)

n是输出channel，s是分几组，可以加速s倍。

![week7-8 Acceleration_39](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_39.jpg)

![week7-8 Acceleration_40](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_40.jpg)

#### Model/Parameter prune

前言：模型剪枝，**这部分是针对于已经训练好的网络进行的剪枝压缩等操作。**

![week7-8 Acceleration_41](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_41.jpg)

这些上面讲过的网络有人专门对比过，MobileNet V3效果是最好的。

remove掉哪些layer呢？

哪些连接可以定义为重要？哪些连接可以定义为不重要呢？

#### Model Quantization

前言：模型的量化就是将模型的参数类型进行修改，比如可以将浮点型改为整形，32位改为8位等等。

![week7-8 Acceleration_42](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_42.jpg)

![week7-8 Acceleration_43](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_43.jpg)

Method1:https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html;https://github.com/onnx/onnx-tensorrt

Method2:https://github.com/NVIDIA-AI-IOT/torch2trt

LIKN1:https://zhuanlan.zhihu.com/p/157610269

LINK2:https://www.jianshu.com/p/ec7baf54ac7d

Relu6的作用：目的就是人为限制正数的范围，到6那就不要往上走了，因为量化的时候超过6然后再量化就有精度损失了。

![week7-8 Acceleration_44](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_44.jpg)

这部分有代码实战。

#### 2、Network Slimming

![week7-8 Acceleration_45](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_45.jpg)

这些层的权重(channel scaling factors)是由BN层来的伽玛γ值来决定

prune:  vi. 删除；减少；vt. 修剪；删除；剪去

##### Pipeline:

- ##### train original model

- ##### load trained model +prune layers

- ##### Finetune pruned model

![week7-8 Acceleration_46](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_46.jpg)

illustration: **说明**，理解，插图。



![week7-8 Acceleration_47](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_47.jpg)

**去掉权重不重要的层，那这个权重怎么来的呢？**

**来源于BN层，伽玛值就是权重值，代表这一层的重要程度**

**select：通过select方法从BN层出来挑出有用的layer层**, **得到一个w权值的列表，然后进行sort排序。**

**select具体方法是咋样的呢？**

**通过weight值进行排序，然后按比例截取前面分数高的部分。**截取50%效果不错，建议这样。

![week7-8 Acceleration_48](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_48.jpg)

伽玛γ值会实现scale，β值会实现shift

![week7-8 Acceleration_49](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_49.jpg)

select方法：

​	@a.Sort bn’s γ(weight)

​	@b.Set prune ratio 

​	@c.Create a mask where

![week7-8 Acceleration_50](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_50.jpg)

**代码框架：**

训练完的大model，经过剪枝操作，得到一个lighted model，然后进行finetune得到最后的小model。

![week7-8 Acceleration_51](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_51.jpg)

![week7-8 Acceleration_52](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_52.jpg)

![week7-8 Acceleration_53](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_53.jpg)

#### 3、Knowledge Distillation

前言：知识蒸馏，要从网络结构入手

![week7-8 Acceleration_54](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_54.jpg)

通过大model产生一个小model，然后这样可以让小model产生的效果跟大model一样。

能够学习到以前被我们压抑或者抑制掉的知识

老师教一部分(CE Loss--lambda)，学生自己学习一部分(CE Loss--1-lambda)

**这个方法主要针对于分类问题。**

![week7-8 Acceleration_55](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_55.jpg)

T是超参数，一般是10或者20

那这个知识蒸馏在代码里面具体怎么实现的呢？

![week7-8 Acceleration_56](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_56.jpg)

![week7-8 Acceleration_57](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_57.jpg)

![week7-8 Acceleration_58](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_58.jpg)

![week7-8 Acceleration_59](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_59.jpg)

参考代码：https://github.com/clovaai/overhaul-distillation

![week7-8 Acceleration_60](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_60.jpg)

![week7-8 Acceleration_61](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_61.jpg)

![week7-8 Acceleration_62](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_62.jpg)

![week7-8 Acceleration_63](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_63.jpg)

![week7-8 Acceleration_64](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_64.jpg)

![week7-8 Acceleration_65](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_65.jpg)

​									**本节课的知识架构总结**

![week7-8 Acceleration_66](D:\Desktop\8期CV课\项目二(安防监控之实时口罩人脸检测)\8、算法加速\week7-8 Acceleration\week7-8 Acceleration_66.jpg)

deploy：配置，部署

不能dropout，它只是为了防止过拟合。

YOLO V3 Network Slimming 代码网上可以找到。



### 2、代码

```python
# 听课笔记：

```



### 3、作业

```python
# 
```

如何跑通YOLO V3代码：

1、环境安装及配置

百度搜索anaconda 清华下载

安装包下载地址：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

安装第三方库命令：`conda install numpy`

​			   或者`pip install numpy`

https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

更换成清华源的话就会快很多

使用命令：

​	`conda activate base`直接进行的是base环境

anaconda创建新环境命令：

```python
conda create -n 文件夹 python=3.6（指定python的版本）
```

当弹出 ：Proceed ([y]/n)? 输入y回车

完成后就可以进入此环境，查看anaconda所有的环境：

`conda env list`

 激活环境：

```python
conda activate 文件夹
```

参考链接：https://blog.csdn.net/baidu_34638825/article/details/105253587



2、训练部分

train.txt文件说明：

类别ID cx cy width height

COCO数据：L t w h 都是绝对的也就是相对于原图的来说的；左上角的点和w以及h

VOC：l t r b；我们的口罩数据集是这样的。VOC是绝对的，左上角和右下角

YOLO：cx cy w h坐标都是相对的，都是经过除以原图的w和h的

而且都在坐标第四象限



如果是口罩数据集的话：

config.cfg文件里面需要修改的是：

[YOLO]里面的filters=21=3x(4+1+2)

还有classes=2

查看可视化：`tensorboard --logdir log`



### 补充：跑通YOLOV3

1、如何跑通YOLO V3项目

在终端跑代码

进入终端解压【PyTorch-YOLOv3-code.zip*】

```python
$ cd PyTorch-YOLOv3-code
$ cd coco   
$ mkdir images   
$ cd images  
```

cd .. 出来到data路径下, 把图片 val2014.zip、train2014.zip 放在 PyTorch-YOLOv3-code/coco/images 下：

```python
$ unzip train2014.zip    
$ unzip val2014.zip    
```

调到路径：PyTorch-YOLOv3-code/data/coco 下：

此处应该有【5k.part】【 trainvalno5k.txt】

Set Up Image Lists，设置图片的路径 $PWD 改为 ./data/coco paste <(awk "{print "./data/coco"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt paste <(awk "{print "./data/coco"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

路径：PyTorch-YOLOv3-code/data/coco 下，应该有【labels.tgz】

```
$ tar xzf labels.tgz
```

**使用coco数据集+Darknet53+在ImageNet上训练的与训练权重，训练模型**

```
$ python train.py --data_config config/coco.data  --pretrained_weights weights/darknet53.conv.74
```

若遇见 No module named 'tensorflow' ==> pip install tensorflow==1.15.0 若遇见 No module named 'terminaltables' ==> pip install terminaltables==3.1.0

若遇见 /pytorch/aten/src/ATen/native/IndexingUtils.h:20: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.

在【model.py】上修改，convert int8 to bool

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709131928771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)

训练起来的效果，此处使用的是CPU

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709135950192.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)

百度平台的常用指令

```shell
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```

```shell
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```

```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
Looking in indexes: https://mirror.baidu.com/pypi/simple/
Collecting beautifulsoup4
  Downloading https://mirror.baidu.com/pypi/packages/d1/41/e6495bd7d3781cee623ce23ea6ac73282a373088fcd0ddc809a047b18eae/beautifulsoup4-4.9.3-py3-none-any.whl (115kB)
     |████████████████████████████████| 122kB 26.1MB/s eta 0:00:01
Collecting soupsieve>1.2; python_version >= "3.0" (from beautifulsoup4)
  Downloading https://mirror.baidu.com/pypi/packages/36/69/d82d04022f02733bf9a72bc3b96332d360c0c5307096d76f6bb7489f7e57/soupsieve-2.2.1-py3-none-any.whl
```

```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

2、口罩遮挡项目项目环境搭建

使用 face data 跑通YOLOv3代码

解压缩数据集

> 建议在终端，路径跳转到data94512文件夹下,解压【PyTorch-YOLOv3-code.zip】【face_mask.zip】

```
# $ 意味在终端输入
$ cd data/data94512/ 
$ unzip PyTorch-YOLOv3-code.zip
$ ls 
```

- 解压之后文件夹有【PyTorch-YOLOv3-code】
- 图片放在【JPEGImages】，标签放在【Annotations】，代码放在【PyTorch-YOLOv3-code】

> 本次任务是使用 face_mask.zip 数据集，跑通此【PyTorch-YOLOv3-code】 代码！

提示：
data97412 文件夹的【PyTorch-YOLOv3-code】，是把数据集解压之后，放在 【data/data97412/PyTorch-YOLOv3-code/data/custom/images】下
先不管这个文件夹，先自己根据文档操作。

查看 [Readme_cn.md] 中文文档

> **data/data94512/PyTorch-YOLOv3-code/Readme_cn.md** 重点看【5 在自定义数据集上训练】

5.1 自定义模型

输入自定义数据集中的类别数（把`<num-classes>`替换为你的类型数），运行下列代码，生成自定义模型的定义文件。

```
$ cd PyTorch-YOLOv3-code/config
$ bash create_custom_model.sh <num-classes>
```

运行之后应该会在config文件夹中生成一个'yolov3-custom.cfg'文件。
**注：这个文件可能会生成在根目录下，需要将其移动回config文件夹。**

> 电脑配置不一样，报错也会不一样，特别是Linux，若出现以下错误，需要修改[data/data94512/PyTorch-YOLOv3-code/config/yolov3-custom.cfg]

```shell
create_custom_model.sh: line 2: $'\r': command not found
create_custom_model.sh: line 4: $'\r': command not found
expr: non-integer argument
expr: syntax error
expr: non-integer argument
expr: syntax error
expr: non-integer argument
expr: syntax error
```

> vim 查看 create_custom_model.sh，看见anchors 长串数字，停下来，看见计算公式(expr 3 * $(expr $NUM_CLASSES + 5))。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210628174003661.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)

> 把公式(expr 3 * $(expr $NUM_CLASSES + 5))放在终端里运行获得数字 21
>
> 就是3*（num_class+5）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210628174143569.png)

> 将filtersfiltersfilters 填写上 = 21

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210628174438217.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)

5.2 自定义类别

在`data/custom/classes.names`文件中填入自定义数据集的标签（类别）名称，每一行填一个标签（类别）名称。

> 一般情况下，制作数据集会知道类别多少个，本次数据集是2类，face、face_mask

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210628180646848.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)

> **注：填写完成后需多按一个回车，保证文件的最后一行是空行，也就代表“什么物体都不是”，否则会造成使用index在list中取值时报错。**

```sh
$ vim classes.names
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210628181459526.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)

5.3 自定义图片

将自定义数据集的所有图片存入`data/custom/images/`文件夹。

```sh
$ cd /
$ cp -r home/aistudio/JPEGImages home/aistudio/data/data94512/PyTorch-YOLOv3-code/data/custom
```

5.4 自定义标签

将自定义数据集的所有标签文件存入`data/custom/labels/`文件夹。

修改标签文件 ****.xml

**你需要准备（自行编写）：**

1. annotations解析文件。即原始的annotations文件一般为json文件或者xml文件，需要将原始数据解析出来，并按照本项目要求的格式重新存储。可参考https://www.runoob.com/python/python-xml.html进行解析；
2. 训练/测试集划分文件。即需要从整的数据集中划分一部分为训练集，另一部分为验证集。可使用`sklearn.model_selection.train_test_split`进行划分。

**你需要确保：**

1. **每一个图片文件对应一个标签文件**，例如一张图片：`data/custom/images/train.jpg`，对应一个标签文件`data/custom/labels/train.txt`，即图片——标签一一对应；
2. 标签文件内的每一个box（一张图上的每一个box），都应被定义为`标签的index box中心点的x坐标 box中心点的y坐标 box的宽 box的高`。 需要注意的是，
   ①这5项内容之间应以“ ”空格连接；
   ②标签index应与自定义类别时填写的`data/custom/classes.names`文件对应，即标签在哪行它的index就是几（从0开始算）；
   ③中心点坐标以及宽高需归一化到[0,1]之间，即使用原始的中心点坐标和宽高的值除以对应的图片总宽、图片总高。

参考代码

```python
import xml.dom.minidom as xmldom
import os

input_annotations_dir = 'Annotations'
output_labels_dir = 'work/PyTorch-YOLOv3-code/data/custom/labels'
```

```python
anno_list = os.listdir(input_annotations_dir)

for anno in anno_list:
	#print(anno)
	# 获取地址
	annodir = os.path.join(input_annotations_dir, anno)
	# 初始化
	domobj = xmldom.parse(annodir)
	elementobj = domobj.documentElement

	# 读入图片文件名与宽高
	filename = elementobj.getElementsByTagName("filename")[0].firstChild.data
	img_W = int(elementobj.getElementsByTagName("width")[0].firstChild.data)
	img_H = int(elementobj.getElementsByTagName("height")[0].firstChild.data)

	# 打开一个txt
	with open(os.path.join(output_labels_dir, filename.split('.')[0]+'.txt'), 'w') as f:
		# 读取物体信息（label与box）
		object_n = elementobj.getElementsByTagName("object")
		for one_object in object_n:
			label_name = one_object.getElementsByTagName("name")[0].firstChild.data
			bnbox = one_object.getElementsByTagName("bndbox")[0]
			xmin = int(bnbox.getElementsByTagName("xmin")[0].firstChild.data)
			ymin = int(bnbox.getElementsByTagName("ymin")[0].firstChild.data)
			xmax = int(bnbox.getElementsByTagName("xmax")[0].firstChild.data)
			ymax = int(bnbox.getElementsByTagName("ymax")[0].firstChild.data)

			# 获取一个box的信息
			label_idx = '0' if label_name == 'face' else '1'
			x_center = str((xmin + xmax) / 2 / img_W)
			y_center = str((ymin + ymax) / 2 / img_H)
			width = str((xmax - xmin) / img_W)
			height = str((ymax - ymin) / img_H)

			# 将信息写入txt
			data = label_idx+' '+x_center+' '+y_center+' '+width+' '+height+'\n'
			f.write(data)
```

**你需要生成**

1. `data/custom/train.txt`文件，并在里面保存所有训练图片的路径；
2. `data/custom/valid.txt`文件，并在里面保存所有验证图片的路径。

**注：以上内容在原始代码中都已给出简单的参考文件**

参考代码

```sh
! ls
2059138.ipynb  Annotations  data  work
```

```python
from sklearn.model_selection import train_test_split
import os

image_list = os.listdir('work/PyTorch-YOLOv3-code/data/custom/images')
X_train, X_test = train_test_split(image_list, test_size=0.2)

with open('work/PyTorch-YOLOv3-code/data/custom/train.txt', 'a') as f:
	for tra in X_train:
		data = os.path.join('data/custom/images', tra)
		f.write(data+'\n')

with open('work/PyTorch-YOLOv3-code/data/custom/valid.txt', 'a') as f:
	for tes in X_test:
		data = os.path.join('data/custom/images', tes)
		f.write(data+'\n')
```

5.5 训练

运行下列代码，训练自定义模型。

```shell
$ python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

注：若需使用在ImageNet上训练的预训练骨干网络，请在上述命令行中添加`--pretrained_weights weights/darknet53.conv.74`.

报错：AttributeError: module 'tensorflow._api.v1.summary' has no attribute 'file_writer'解决方法：

```python
将这句代码self.writer = tf.summary.create_file_writer(log_dir)
修改成这句代码完美解决self.writer = tf.summary.FileWriter(log_dir)
```

完美解决 UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool

[参考方案，本人用过](https://blog.csdn.net/c2250645962/article/details/104869015)

RuntimeError: DataLoader worker (pid) is killed by signal: Killed.

原因是机器内存不够。
可通过减少dataloader的num_worker或增加虚拟内存解决。
本人设置num_worker = 2

torch.view()的作用：

```python
 Example::
        
            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])
        
            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False
```

- unsqueeze(n)函数  VS  squeeze(n)函数

```python
https://blog.csdn.net/flysky_jay/article/details/81607289
unsqueeze(n)函数是在第n维度上增加一个维度
squeeze(n)函数是在第n维度上减去一个维度
```

![img](https://pic4.zhimg.com/80/v2-54d50cce5e9539495ab4beed88b5522b_720w.jpg)