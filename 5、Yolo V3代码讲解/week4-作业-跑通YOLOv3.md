### week4

从本周开始，我们的课程也开始逐步迈向深水区。需要有些与真实项目挂钩的内容了。

本周的内容是我们真正实战的开始，意味着我们将逐步完成(戴口罩)人脸的检测。通常情况下，

具体实施步骤大致规划如下:

 	A. Follow 别人代码跑通
	 B. 在别人代码基础上跑通自己的数据
	 C. 根据实际需求/算法进展进行改良
	 D. 如需工程化，还需进行模型加速/转换等

因而，本周作业侧重**A、B**上，如下:

**A、使用COCO2014跑通 Yolo v3 PyTorch 版本（<font color=red>非作业</font>）**

**B、使用真实的数据跑通 Yolo v3 PyTorch 版本（<font color=red>作业</font>）**

**<font color=red>从重要的的B项开始</font>~**

#### 1. 使用真实的数据（Face Data） 跑通 Yolo v3 PyTorch 版本

(**<font color=red>此部分检查，是“作业”</font>**)

**1.1 **工作中，数据筹备通常也是自己完成，包括:数据采/收集，标注，格式，清洗等等。这

里，为了节省大家时间，我们的工程人员已经帮大家干了这些事情。数据已上传。

Face Data百度云盘链接:https://pan.baidu.com/s/1iozVgZGaHCDC9hPvSGnF2g  密码:riof

**1.2 ** 利用 readme 中的提示，完成训练:请注意，哪里需要改，比如 cfg 文件，代码等。

<font color=red>请提交</font>训练 log 或截图。

**1.3 ** Inference，利用自己训练好的模型进行人脸检测，并<font color=red>提交 </font>inference 的结果图(几张即

可)。如果万事正确，应能看到这样的成果:

<img src="https://img-blog.csdnimg.cn/20210710104344915.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70" style="zoom:50%;" />



​	我们采用 [YoloV3 (PyTorch 版)](https://github.com/eriklindernoren/PyTorch-YOLOv3)的代码为我们的基础代码。打开链接应能看到如下界面:

<img src="https://img-blog.csdnimg.cn/20210707151202550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:30%;" />



其 ReadMe(就是“介绍”)非常详尽。请同学们按照此“ReadMe”, 看不懂的同学，我们还提供了<font color=red>“ReadMe”的翻译版本</font>/PyTorch-YOLOv3-code/Readme_cn.md，在百度云盘下载，链接:https://pan.baidu.com/s/1RPIIc5UWU5F5a90Sy2DUJw  密码:p2o8

<font color=red>百度平台入口：</font>



![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709151843868.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)



#### 2. 使用COCO2014 跑通 Yolo v3 PyTorch 版本

>  (【**跑通 Yolo v3 PyTorch 版本】这部分不检查，也可以认为此部分“不算作作业”。** 目的是让同学们有个循序渐进的过程。
>  对于程度较好的同学，可以忽略此部分。 对于训练流程还不熟悉的同学，可以通过此部分熟悉。)

<font color=red>百度平台入口:</font>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709143228435.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70)

- 先按要求下载 pretrained weight / data等(COCO2014，百度平台有)，想下载的同学跳转到**2.1 .认识 COCO 数据集**)。 
- 用 pretrained weight “Inference”，确保整体代码 work。

- “Train” COCO 数据。因数据量较大，不需要训完，在 5 个 epoch 之内即可，但需要确保 训练流程通畅正确，如 loss 在正常下降等。

##### 2.1 .认识 COCO 数据集

任何真实项目，最为关键的，在目前 AI 发展工程化阶段，就是数据，没有之一。所以，认识数据，将是 我们至关重要的任务。

在所有通行网络所利用到的数据当中，COCO 数据集无疑是应用较广，比较有影响力的准标准化 数据集。换句话说，就是较多的公开框架会使用 COCO 数据集 (但绝不意味着只能使用 COCO 数据 及其格式)。所以，熟悉 COCO 数据集及其数据格式，有一定必要性。

我们应用的 YoloV3 (PyTorch 版)的官方 demo 同样应用了 COCO 数据集。因而

- 我们可以了解 COCO 数据的格式，可以去百度平台上了解数据集COCO2014（自己的电脑内存够，可以下载相关内容）。下载链接及界面如下:
  下载链接：https://cocodataset.org/#download  or  COCO2014数据集百度云盘链接: https://pan.baidu.com/s/1gNieo0Cmf1xmG8gBDF_N2w  密码:wmo4

<img src="https://img-blog.csdnimg.cn/20210706153525989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:30%;" />

​				其中，图像部分就是“Images”下面的，而相应的标注信息就在“Annotations”下面。



- 下载数据集

  任选某一年(2014 或 2017)的 Train 或者 Val images 以及相对应的 annotations (2014 Train/Val annotations 或 2017 Train/Val annotations) 【注意，Train 图像非常大】

- 认识图像与标注关系:

  友情提示: 

  ​	a. 图像很多，所以下载并打开可能很慢

  ​	b. 标注文件不小，并且格式奇葩，不建议大家打开 train.json。打开的话，请使用 val.json

  请大家整理.json 中的内容，充分理解:

  ​	 Json 格式 (体会 dict 如何在里面起作用)

  ​	整理出一份秀珍版的 json 。举例如下，这里我用的是 2017 年的 val .json:

<img src="https://img-blog.csdnimg.cn/20210706163657962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:50%;" />

所有绿色的圆圈:应该是 json 标注文件中的 root keys, 如果各位整理正确，应该有 5 个。 所有红色的框:应该是 key 下某个 value，此 value 就是某个 sample 的标注信息。 应当注意，一个 key 下的 sample 是非常多的，我们只需要保留 1-2 个 sample 帮助我们 理解就可以了

为确保我们理解，利用整理好的信息，选取一张图像，画出标注结果(bounding box + class，其他的不要)，如:

<img src="https://img-blog.csdnimg.cn/20210710105637942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA3NTE5NzQ=,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:30%;" />

​																(本图有两个 instances，这里只标出了人)

如果仍有困难，比如不知道 id/image_id 是如何对应的，centernet 的代码能够狠好的帮 助我们，请参考:

https://github.com/zzzxxxttt/pytorch_simple_CenterNet_45/blob/master/datasets/coco.py

line 89-94 的直接提示。

+++++++++++++++++ **以下不是作业，是一些简单的说明** +++++++++++++++++++ 

**关于COCO格式标注文件**

参考资料：[COCO数据标注介绍](https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.0/docs/tutorials/PrepareDataSet.md#coco%E6%95%B0%E6%8D%AE%E6%A0%87%E6%B3%A8%E4%BB%8B%E7%BB%8D)

COCO数据标注是将所有训练图像的标注都存放到一个json文件中。数据以字典嵌套的形式存放。

json文件中包含以下key：

- info，表示标注文件info。
  
- licenses，表示标注文件licenses。
  
- images，表示标注文件中图像信息列表，每个元素是一张图像的信息。如下为其中一张图像的信息：
  

```python
    {
     'license': 3,                       # license
     'file_name': '000000391895.jpg',    # file_name
     'height': 360,                      # image height
     'width': 640,                       # image width
     'id': 391895                        # image id
    }
```

- annotations，表示标注文件中目标物体的标注信息列表，每个元素是一个目标物体的标注信息。如下为其中一个目标物体的标注信息：

```python
    {
     'segmentation':             # 物体的分割标注
     'area': 2765.1486500000005, # 物体的区域面积
     'iscrowd': 0,               # iscrowd
     'image_id': 558840,         # image id
     'bbox': [199.84, 200.46, 77.71, 70.88], # bbox [x1,y1,w,h]
     'category_id': 58,          # category_id
     'id': 156                   # image id
    }
```

查看COCO标注文件

```python
    import json
    coco_anno = json.load(open('./annotations/instances_train2017.json'))
    # coco_anno.keys
    print('\nkeys:', coco_anno.keys())
    # 查看类别信息
    print('\n物体类别:', coco_anno['categories'])
    # 查看一共多少张图
    print('\\n图像数量：', len(coco_anno['images']))
    # 查看一共多少个目标物体
    print('\n标注物体数量：', len(coco_anno['annotations']))
    # 查看一条目标物体标注信息
    print('\n查看一条目标物体标注信息：', coco_anno['annotations'][0])
```



