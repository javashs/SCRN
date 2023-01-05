# -*- coding: utf-8 -*-

from util.My_tool1 import *
import time
from model.SCRN import SCRN
import torch
if __name__ == '__main__':

    model = torch.load('train_model/model.pth')

    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()

    x = np.load('test_data\\clear.npy')
    x = x.astype(np.float64)

    y = np.load('test_data\\miss.npy')
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
    imgplot1 = plt.imshow(y, cmap=plt.cm.seismic, aspect='auto', vmin=-1, vmax=1)
    plt.show()

    torch.cuda.synchronize()
    start_time = time.time()
    y_ = y_.type(torch.float32)
    y_ = y_.cuda()

    x_ = model(y_)  # inferences
    x_ = x_.view(y.shape[0], y.shape[1])
    x_ = x_.cpu()
    x_ = x_.detach().numpy().astype(np.float64)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    pre_snr = snr_(y, x)
    print("去噪前：snr" + str(pre_snr))
    snr = snr_(y - x_, x)
    print("去噪后：snr"+str(snr))

    pre_ssim = ssim_(y, x)
    print("去噪前：ssim" + str(pre_ssim))
    ssim = ssim_(y - x_, x)
    print("去噪后：ssim" + str(ssim))

    imgplot1 = plt.imshow(x, cmap=plt.cm.seismic,  aspect='auto', vmin=-1, vmax=1)
    plt.show()

    imgplot1 = plt.imshow(y, cmap=plt.cm.seismic,  aspect='auto', vmin=-1, vmax=1)
    plt.show()

    imgplot1 = plt.imshow(x_, cmap=plt.cm.seismic, aspect='auto', vmin=-1.3, vmax=1.3)
    plt.show()

    imgplot1 = plt.imshow(y - x_, cmap=plt.cm.seismic, aspect='auto', vmin=-1, vmax=1)
    plt.show()












