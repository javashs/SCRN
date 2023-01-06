# -*- coding: utf-8 -*-
import random
import os
import pickle
import torch
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms


class MyDatasets(data.Dataset):
    def __init__(self,Input_root_dir,Target_root_dir,start_point_y=0,data_size=[128,128],transform=None):

        self.Input_root_dir = Input_root_dir
        self.Target_root_dir = Target_root_dir
        self.transform = transform
        self.Inputdata = os.listdir(self.Input_root_dir)
        self.Targetdata = os.listdir(self.Target_root_dir)

        self.start_point_y = start_point_y
        self.Data_size = data_size

    def __len__(self):
        return len(self.Targetdata)

    def __getitem__(self, index):
        data_index = self.Targetdata[index]
        # 确定矩阵初始截取y点
        start_point_y = self.start_point_y
        #确定训练数据大小
        data_size = self.Data_size

        data_target = (np.load(os.path.join(self.Target_root_dir,data_index),allow_pickle=True))
        #rate = (random.sample([0.02, 0.08, 0.11,  0.15, 0.18], 1))[0]
       
        rate = (random.sample([0.02, 0.08, 0.18,  0.28, 0.38], 1))[0]
        mask = irregular_mask(data_target, rate)
        data_input = mask*data_target
        
        SNR = (random.sample([-2,-1,1,5,10], 1))[0]
        noise = np.random.randn(data_target.shape[0],data_target.shape[1]) 	#产生N(0,1)噪声数据
        noise = noise - np.mean(noise) 								#均值为0
        signal_power = np.linalg.norm(data_target - data_target.mean())**2 / data_target.size	#此处是信号的std**2
        noise_variance = signal_power/np.power(10, (SNR/10))         #此处是噪声的std**2
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise    ##此处是噪声的std**2
        
        data_input = noise + data_input

        (l, w) = data_input.shape
        data_input = data_input.reshape((l,w,1))


        (l,w) = data_target.shape
        data_target = data_target.reshape((l,w,1))
        
        if self.transform:
            data_input = self.transform(data_input)
            data_target = self.transform(data_target)

        return data_input, data_target


        # #根据索引item获取该输入数据
        # data_index = self.Targetdata[index]
        # data_target = (np.load(os.path.join(self.Target_root_dir,data_index),allow_pickle=True))
        # SNR = (random.sample([-2,0,2,4,6,8,10], 1))[0]
        # noise = np.random.randn(data_target.shape[0],data_target.shape[1]) 	#产生N(0,1)噪声数据
        # noise = noise - np.mean(noise) 								#均值为0
        # signal_power = np.linalg.norm(data_target - data_target.mean())**2 / data_target.size	#此处是信号的std**2
        # noise_variance = signal_power/np.power(10, (SNR/10))         #此处是噪声的std**2
        # noise = (np.sqrt(noise_variance) / np.std(noise)) * noise    ##此处是噪声的std**2
        # data_input = noise + data_target
        # (l, w) = data_input.shape
        # data_input = data_input.reshape((l,w,1))
        #
        #
        # (l,w) = data_target.shape
        # data_target = data_target.reshape((l,w,1))
        #
        # if self.transform:
        #     data_input = self.transform(data_input)
        #     data_target = self.transform(data_target)
        #
        #
        # return data_input, data_target


def irregular_mask(data, rate):
    """the mask matrix of random sampling
    Args:
        data: original data patches
        rate: sampling rate,range(0,1)
    """
    mask = np.ones(data.shape[1])
    mask[:int(rate*data.shape[1])]=0
    np.random.shuffle(mask)
    return mask

def regular_mask(data, a):
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(), dtype=torch.float64)
    for i in range(n):
        if (i + 1) % a == 1:
            mask[:, :, i] = 1
        else:
            mask[:, :, i] = 0
    return mask

if __name__ == '__main__':

    Input_root_dir = r'train'
    Target_root_dir = r'train'
    data = MyDatasets(Input_root_dir=Input_root_dir, Target_root_dir=Target_root_dir, transform=None)
    dataloader = torch.utils.data.DataLoader(data,batch_size=32,shuffle=True)
    input_example, target_example= next(iter(dataloader))


