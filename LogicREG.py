import numpy
import math
import random
def cal_gradient(x, y, w):
    """
    :param x: numpy vector, 增广特征向量
    :param y: int, x对应的结果
    :param w: numpy vector, 增广权向量
    :return: 一行数据的梯度
    """
    x = numpy.array(x)
    w = numpy.array(w)
    exp_xw = math.e**(numpy.dot(x, w))
    return (exp_xw/(1+exp_xw)-y)*x


def split_dataset(data, num):
    data_list = []
    val_size = len(data) // num
    for k in range(num - 1):
        t_data = []
        for i in range(val_size):
            t_data.append(data.pop(random.randint(0, len(data) - 1)))
        data_list.append(t_data)
    data_list.append(data)
    return data_list
