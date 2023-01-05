# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os
import re
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim


def snr_(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    计算信噪比
    :param arr1: 去噪数据
    :param arr2: 干净数据
    :return: 信噪比
    """
    snr = 10*np.log10(np.sum(arr2**2)/np.sum((arr2-arr1)**2))
    return snr


def ssim_(imageA, imageB):
    grayScore = compare_ssim(imageA, imageB, win_size=15)
    return grayScore


def log(*args, **kwargs):
    # 例：2022-02-18 14:34:23 + 内容
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def findLastCheckpoint(save_dir):  # 获取模型文件路径
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))  # 路径+模型名字的拼接
    if file_list:  # 模型存在
        epochs_exist = []  # 创建空list
        for file_ in file_list:  # 遍历所有模型file_
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))  # 取第一个位置 例子：001变成1，str转成int类型
        initial_epoch = max(epochs_exist)  # 返回最大的迭代数组下标
    else:  # 模型不存在
        initial_epoch = 0  # 初始化迭代次数为0
    return initial_epoch  # 返回迭代次数的值


def produce_csv(file_name):
    df = pd.DataFrame(columns=['time', 'step', 'train_Loss', 'val_loss','time'])
    df.to_csv(file_name, index=False)


def save_csv(file_name,epoch, train_loss, val_loss,time):

    time = "%s" % datetime.now()  # 获取当前时间
    step = "Step[%d]" % epoch
    train_l = "%f" % train_loss
    val_l = "%g" % val_loss

    list = [time, step, train_l, val_l]
    data = pd.DataFrame([list])
    data.to_csv(file_name, mode='TestData11_picture', header=False, index=False)


def produce_csv1(file_name):
    df = pd.DataFrame(columns=[ 'step', 'pre_snr', 'snr',  'pre_ssmi', 'ssmi','time'])
    df.to_csv(file_name, index=False)


def save_csv1(file_name, epoch, pre_snr, snr,  pre_ssmi, ssmi, time):

  #  time1 = "%s" % datetime.now()  # 获取当前时间
    step = "Step[%d]" % epoch

    pre_snr_l = "%f" % pre_snr
    snr_l = "%f" % snr

    pre_ssmi_l = "%f" % pre_ssmi
    ssmi_l = "%f" % ssmi
    time = "%s" % time

    list = [ step, pre_snr_l, snr_l, pre_ssmi_l,ssmi_l,time]
    data = pd.DataFrame([list])
    data.to_csv(file_name, mode='TestData11_picture', header=False, index=False)

def show_csv(file_path, name_1, name_2, epoches):
    data = pd.read_csv(file_path)
    tain_loss = data[[name_1]]
    val_loss = data[[name_2]]
    x = np.arange(0, epoches)
    y1 =np.array(tain_loss)#将DataFrame类型转化为numpy数组
    y2 = np.array(val_loss)
    #绘图
    plt.plot(x, y1, label="train_loss")
    plt.plot(x, y2, label="val_loss")
    plt.title("loss")
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()   #显示标签
    plt.show()


def show_loss_(epoch, train_loss, val_loss):
    plt.plot(epoch, train_loss, label='训练集损失值')
    plt.plot(epoch, val_loss, label='验证集损失值')
    plt.legend(loc='best')
    plt.ylabel("损失值")
    plt.xlabel("迭代次数")
    plt.title("训练集和验证集的损失值对比图")
    plt.show()


def show_snr_(train_snr, val_snr):
    plt.plot(train_snr, label='训练集信噪比')
    plt.plot(val_snr, label='验证集信噪比')
    plt.legend(loc='best')
    plt.ylabel("信噪比")
    plt.xlabel("迭代次数")
    plt.title("训练集和验证集的信噪比对比图")
    plt.show()

def show_ssmi_(train_ssmi, val_ssmi):
    plt.plot(train_ssmi, label='训练集结构相似度')
    plt.plot(val_ssmi, label='验证集结构相似度')
    plt.legend(loc='best')
    plt.ylabel("结构相似度")
    plt.xlabel("迭代次数")
    plt.title("训练集和验证集的结构相似度对比图")
    plt.show()

def show_x_y(x, y):

    """
    :param x: x是干净数据
    :param y: y是噪声数据
    :return:
     """
    plt.figure(figsize=(12, 5))
    # 一行两列第一个位置
    plt.subplot(121)
    plt.imshow(x, cmap=plt.cm.seismic, interpolation='nearest', aspect='auto', vmin=-0.5, vmax=0.5)
    plt.subplot(122)
    plt.imshow(y, cmap=plt.cm.seismic, interpolation='nearest', aspect='auto', vmin=-0.5, vmax=0.5)
    plt.show()


def show_x1_n(x1, y1):

    """
   :param x1: 去噪后的干净数据
   :param y1: 去噪的噪声
   :return:
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(x1, cmap=plt.cm.seismic, interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(shrink=0.5)

    plt.subplot(122)
    plt.imshow(y1, cmap=plt.cm.seismic, interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(shrink=0.5)
    plt.show()







