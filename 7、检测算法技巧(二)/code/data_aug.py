import numpy as np
import random
import cv2

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import argparse as args


''' Mix up '''
for i,(images,target) in enumerate(train_loader):
    # 1.input output
    images = images.cuda()
    target = torch.from_numpy(np.array(target)).float().cuda()

    # 2.mixup
    # beta distribution: change shape by setting different shape:
    # https://www.sciencedirect.com/topics/computer-science/beta-distribution
    alpha=config.alpha      # by default we could set to 1.0 (uniform dist.) 设置为1就是平均分布
    ratio = np.random.beta(alpha, alpha)           # E.g. lam = 0.432
    # randomly return a permutation. E.g torch.randperm(5)=> tensor([2, 3, 0, 4, 1])
    # permutation面试题:
    # vecotr/list=[1,2,3,4,5],all permutation?   [[2,1,3,4,5],[3,1,2,4,5],.......]
    #                                            [1,2,3,4,5,4,4,4,3],all permutations?-> dfs  Time complexity
    index = torch.randperm(images.size(0)).cuda()  # 随机排列一下，打乱之后的index
    inputs = ratio * images + (1 - ratio) * images[index]
    # image mixed up, so do the targets
    # 2 种实现：还一种：targets = ratio * target  + (1 - ratio) * target[index]
    # 后面只用一个criterion就行了
    targets_a, targets_b = target, target[index]
    outputs = model(inputs)
    # do not forget the compound/mixed up criterion
    loss = ratio * criterion(outputs, targets_a) + (1 - ratio) * criterion(outputs, targets_b)

    # 3.backward
    optimizer.zero_grad()   # reset gradient
    loss.backward()
    optimizer.step()        # update parameters of net

# 解析：
# index = torch.randperm(3)
# images = torch.randint(0, 256, (3, 2, 4))
# lam = np.random.beta(1.0, 1.0)
# inputs = ratio * images + (1 - ratio) * images[index]

''' Cutout '''
# 原始用的是cifar10 (size=32, padding=4)
# 怕补0影响结果，所以先要normalize， 再cutout
# [125.3, 123.0, 113.9]代表RGB各个通道的均值
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],       # (rgb order)
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.ToTensor())    # to tensor: [0.0, 1.0]
train_transform.transforms.append(normalize)
# 注意了！
train_transform.transforms.append(Cutout(n_holes=1, length=16))     # by default

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)    # np.clip此处：将范围限定在[0, h]
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


''' cutmix '''
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# generate mixed sample
lam = np.random.beta(args.beta, args.beta)      # 1.0, we use uniform dist.
rand_index = torch.randperm(input.size()[0]).cuda()
target_a = target
target_b = target[rand_index]
bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
# adjust lambda to exactly match pixel ratio
lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))  # 原图保留原始部分的图形比上整张图像的权重
# compute output
output = model(input)
loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)  # (1. - lam)添加新的图像部分占比

''' All used for classification above '''
''' USed for regression below '''

''' Mosaic: Promoted in Yolo v4 '''
# https://github.com/ultralytics/yolov3/blob/master/utils/datasets.py
def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            # 把新图像先设置成原来的4倍，到时候再resize回去，114是gray
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (new/large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (original/small image)
            # 回看ppt讲解
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b    # 有时边上还是灰的
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # 此时x是0-1，同时，label是[bbox_xc, bbox_yc, bbox_w, bbox_c]
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        # a = np.array([[1, 2], [3, 4]])
        # c = np.concatenate(a, axis=0)
        # c: [1, 2, 3, 4]
        labels4 = np.concatenate(labels4, 0)    # 0是dimension
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove

    return img4, labels4






