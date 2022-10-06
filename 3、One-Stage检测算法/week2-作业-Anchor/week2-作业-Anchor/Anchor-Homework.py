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
    print(image.shape)  # (720, 1080, 3)
    feature_map = cv2.resize(image, dsize=(image.shape[1]//32, image.shape[0]//32))
    # 获取图片与特征图的shape，计算长、宽方向上的stride
    image_size = image.shape[:2]
    print(image_size) # (720, 1080)
    feature_map_size = feature_map.shape[:2]
    print(feature_map_size) # (22, 33)
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




