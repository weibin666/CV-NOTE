'''
Softmax作业
作业内容：
1. 使用numpy实现Softmax（假设2个样本，给定[[1, 2, 3], [2，1，3]])；
2. 使用torch.nn.functional.softmax() 验证结果是否一致。
'''
import numpy as np
import torch
import torch.nn.functional as F


def softmax_my(predict):
    '''
    使用numpy实现Softmax
    input:
        numpy.ndarray
        [[1, 2, 3],
        [2，1，3]]
    output:
        softmax value: numpy.ndarray
    '''

    row_size, col_size = predict.shape
    x = np.array(predict)
    # y = np.exp(x) / sum(np.exp(x))
    # print("上溢：", y)
    x = x - np.max(x)  # 减去最大值
    y = np.exp(x) / sum(np.exp(x))
    print("上溢处理为：", y)

    # 会使用max()，其中参数 axis = 1 表示二维数组中沿着横轴取最大值

    # 每一行减去本行最大的数字，提示：reshape

    # 计算e的指数次幂

    # 对每一行进行求和操作

    # 每一行 predict_exp / predict_exp_row_sum
    return y


if __name__ == '__main__':
    '''
    假设两个样本
    [[1, 2, 3],
    [2，1，3]]
    '''
    # 如果想尝试随机数
    # np.random.seed(0)
    # predict = np.random.randn(2, 3)
    # print(predict)
    predict = np.array([[1, 2, 3], [2, 1, 3]])
    softmax_value = softmax_my(predict)
    print('softmax结果：', softmax_value)

    # 验证softmax是否每行和为1
    print(softmax_value.sum(axis=1))

    # torch.nn.functional.softmax(input, dim)
    # 参数：dim:指明维度，dim=0表示按列计算；dim=1表示按行计算
    predict_tensor = torch.Tensor([[1, 2, 3], [2, 1, 3]])
    softmax_torch = F.softmax(predict_tensor, dim=1)
    print('torch 验证：', softmax_torch)

