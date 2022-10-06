# 代码实现NMS和soft-NMS
import numpy as np

boxes = np.array([[100, 100, 210, 210, 0.72],
                  [250, 250, 420, 420, 0.8],
                  [220, 220, 320, 330, 0.92],
                  [100, 100, 210, 210, 0.72],
                  [230, 240, 325, 330, 0.81],
                  [220, 230, 315, 340, 0.9]])  # (x1,y1,x2,y2,score)


def nms(boxes, threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []
    idxs = scores.argsort()[::-1]  # 从大到小排序,argsort函数返回从小到大的索引,[::-1]反序变成从大到小
    while (idxs.size > 0):
        i = idxs[0]
        keep.append(i)
        x11 = np.maximum(x1[i], x1[idxs[1:]])
        y11 = np.maximum(y1[i], y1[idxs[1:]])
        x22 = np.minimum(x2[i], y2[idxs[1:]])
        y22 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[idxs[1:]] - overlaps)
        idxs2 = np.where(ious < threshold)[0]  # np.where函数
        idxs = idxs[idxs2 + 1]  # 注意这个+1

    return keep




# soft_nms操作，这里假设boxes是无序(未按score做降序)的，所以每轮soft_nms迭代都需要类似冒泡排序操作，选择当前top-1 bbox做NMS
# Nt：计算IoU的阈值，IoU > Nt，对应bbox的score权重就要降低
# threshold：降权后通过threshold进一步剔除低权重bbox
def soft_nms(boxes, sigma=0.5, threshold1=0.7, threshold2=0.1, method=1):
    '''
    paper:Improving Object Detection With One Line of Code
    '''
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1  # 下面操作就很常规了，找到当前index i之后所有bboxes中，
                     # score最大的bbox，并将之赋值给maxscore、maxpos

        # 得到评分最高的box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1
        # 通过上面这while循环即可找到maxpos,会将前面的maxpos更新

        # 下面操作更简单，想想我们最开始学C语言，a、b两变量如何交换
        # add max box as a detection
        # 交换第i个box和评分最高的box,将评分最高的box放到第i个位置
        boxes[i, 0] = boxes[maxpos, 0]  # maxpos内的信息，放到index i处，也是当前需要处理的bbox
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1   # 别忘了tx1中可是保存了boxes[i,0]备份的
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]   # 此时tx1就保存的maxpos位置的bbox信息了
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # soft-NMS迭代
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    uinon = float((tx2 - tx1 + 1) *
                                  (ty2 - ty1 + 1) + area - iw * ih)
                    iou = iw * ih / uinon  # 计算iou
                    if method == 1:  # 线性更新分数
                        if iou > threshold1:
                            weight = 1 - iou   #
                        else:
                            weight = 1
                    elif method == 2:  # 高斯权重
                        weight = np.exp(-(iou * iou) / sigma)
                    else:  # 传统 NMS
                        if iou > threshold1:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]  # 根据和最高分数box的iou来更新分数
                    # if box score falls below threshold, discard the box by swapping with last box，update N
                    # 如果bbox调整后的权重，已经小于阈值threshold，那么这个bbox就可以忽略了，
                    # 操作方式是直接用最后一个有效的bbox替换当前pos上的bbox

                    # 如果box分数太低，舍弃(把他放到最后，同时N-1)
                    if boxes[pos, 4] < threshold2:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1           # N-1位置上的bbox已经赋值到前面了，该bbox就可以忽略了；
                        pos = pos - 1       # pos位置上引入了新的有效bbox(N-1)，就需要再计算一遍了

            pos = pos + 1

    keep = [i for i in range(N)]  # 返回这个列表有啥意思？？？？？
    return keep


keep1 = nms(boxes, 0.8)
keep2 = soft_nms(boxes)
print("end")




