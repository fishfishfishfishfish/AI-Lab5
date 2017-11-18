import numpy
import math
import random
import logging


# 配置日志输出，方便debug
def get_logger():
    log_file = "./nomal_logger.log"
    log_level = logging.DEBUG

    logger = logging.getLogger("loggingmodule.NomalLogger")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("[%(levelname)s][%(funcName)s][%(asctime)s]%(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


# 从文件读取训练数据
def get_train_data(filename):
    """
    :param filename: string, 训练文件名
    :return: list[numpy.array[float]], 训练数据[x,x,x,...,y]
    """
    data = []
    f = open(filename, 'r')
    for line in f.readlines():
        row = line.split(',')
        for i in range(len(row)):
            row[i] = float(row[i])
        row = numpy.array(row)
        data.append(row)
    return data


# 将训练数据随机分成平均的几份
def split_dataset(data, num):
    """
    :param data: list[numpy.array[float]], 原始的训练集
    :param num: int, 要分成的份数
    :return: list[list[array[float]], 分解后的num个数据集数组
    """
    random.seed(1)
    data_list = []
    val_size = len(data) // num
    for k in range(num - 1):
        t_data = []
        for i in range(val_size):
            t_data.append(data.pop(random.randint(0, len(data) - 1)))
        data_list.append(t_data)
    data_list.append(data)
    return data_list


# 获取训练集和验证集
def get_train_and_val(dataset, n):
    """
    :param dataset: list[list[array[float]], 分割后的数据集
    :param n: int, 第n份作为验证集
    :return: list[array[float]], list[array[flost]], 训练集和验证集
    """
    traindata = []
    for i in range(len(dataset)):
        if i != n:
            traindata += DataSet[i]
    valdata = dataset[n]  # 剩下的一份作为验证集
    return traindata, valdata


# 计算一行x和当前w计算得到的梯度值
def cal_gradient(x, y, w):
    """
    :param x: numpy vector, 增广特征向量
    :param y: int, x对应的结果
    :param w: numpy vector, 增广权向量
    :return: 一行数据的梯度
    """
    logger = logging.getLogger("loggingmodule.NomalLogger")
    x = numpy.array(x)
    w = numpy.array(w)
    # logger.debug(numpy.dot(x, w))
    exp_xw = math.e**(-numpy.dot(x, w))
    return (1/(1+exp_xw)-y)*x


def train(traindata, eta):
    logger = logging.getLogger("loggingmodule.NomalLogger")
    w = numpy.zeros(len(traindata[0])-1)
    cnt = 0
    while cnt < 1000:
        grad_sum = numpy.zeros(len(traindata[0])-1)
        for d in traindata:
            grad_sum += cal_gradient(d[0:len(d)-1], d[len(d)-1], w)
        w = w + eta*grad_sum
        logger.debug(grad_sum)
        logger.info(w)
        cnt += 1
    return w


# 获取日志输出器
Logger = get_logger()
# 读取训练数据，根据S折交叉验证切割
Data = get_train_data('train.csv')
Sfold = 10
DataSet = split_dataset(Data, Sfold)
Logger.debug("DataSet is : \n")
Logger.debug(numpy.array(DataSet))
# S份中取出一份作为验证集，其余构成训练集
for k in range(Sfold):
    TrainData, ValData = get_train_and_val(DataSet, k)
    train(TrainData, 0.1)



