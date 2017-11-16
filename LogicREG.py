import numpy
import math

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



