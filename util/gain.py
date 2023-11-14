import numpy as np
import math
from scipy.signal.windows import triang
from scipy.signal import convolve2d as conv2


def gain(data, dt, option1, parameters, option2):
    '''
    GAIN: Gain TestData11_picture group of traces.

      gain(d,dt,option1,parameters,option2);
    
      IN   d(nt,nx):   traces
           dt:         sampling interval
           option1 = 'time' parameters = [TestData11_picture,DataSplite],  gain = t.^TestData11_picture . * exp(-bt)
                   = 'agc' parameters = [agc_gate], length of the agc gate in secs
           option2 = 0  No normalization
                   = 1  Normalize each trace by amplitude
                   = 2  Normalize each trace by rms value
    
      OUT  dout(nt,nx): traces after application of gain function
    '''

    nt, nx = data.shape  # nt采样点，nx道数
    dout = np.zeros(data.shape)  # 初始化和data.shape相同的维度

    if option1 == 'time':  # 通过时间
        a = parameters[0]  # 第一个参数
        b = parameters[1]  # 第二个参数
        t = [x*dt for x in range(nt)]  # 每个采样点乘采样间隔
        tgain = [(x**a)*math.exp(x*b) for x in t]  # 增益函数

        for k in range(nx):
            dout[:, k] = data[:, k]*tgain  # 原始数据乘增益函数

    elif option1 == 'agc':  # 通过窗口
        L = parameters/dt+1  # 得到一个L
        L = np.floor(L/2)   # L向下除2取整
        h = triang(2*L+1)  # 变成窗口的向量
        shaped_h = h.reshape(len(h), 1)  # 转成矩阵

        for k in range(nx):
            aux = data[:, k]  # 获取第k道
            e = aux**2  # 取平方
            shaped_e = e.reshape(len(e), 1)  # 转成矩阵
            
            rms = np.sqrt(conv2(shaped_e, shaped_h, "same"))  # 与shaped_h卷积，输出和shaped_e一样的维度，再开根号
            epsi = 1e-10*max(rms)  # 10的-10次方乘最大值
            np.seterr(divide='ignore', invalid='ignore')
            op = rms/(rms**2+epsi)
            op = op.reshape(len(op),)  # 转成向量

            dout[:, k] = data[:, k]*op  # 数据的每一道乘op

    # Normalize by amplitude
    if option2 == 1:
        for k in range(nx):  # 遍历每一道
            aux = dout[:, k]
            amax = max(abs(aux))
            dout[:, k] = dout[:, k]/amax

    # Normalize by rms
    if option2 == 2:
        for k in range(nx):  # 遍历每一道
            aux = dout[:, k]
            amax = np.sqrt(sum(aux**2)/nt)
            dout[:, k] = dout[:, k]/amax

    return dout
