# -*- coding: utf-8 -*-
from util.My_tool1 import *
import time
import torch
if __name__ == '__main__':

    model = torch.load('trained_model\\model.pth')

    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()

    x = np.load('test_data/clear.npy')
    x = x.astype(np.float64)

    y = np.load('test_data/noise_and_miss.npy')
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])


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
    print("before：snr" + str(pre_snr))
    snr = snr_(x_, x)
    print("After：snr"+str(snr))

    pre_ssim = ssim_(y, x)
    print("before：ssim" + str(pre_ssim))
    ssim = ssim_(x_, x)
    print("After：ssim" + str(ssim))

    imgplot1 = plt.imshow(x, cmap=plt.cm.seismic,  aspect='auto', vmin=-1, vmax=1)
    plt.show()

    imgplot1 = plt.imshow(y, cmap=plt.cm.seismic,  aspect='auto', vmin=-1, vmax=1)
    plt.show()

    imgplot1 = plt.imshow(x_, cmap=plt.cm.seismic, aspect='auto', vmin=-1, vmax=1)
    plt.xlabel('Trace')
    plt.ylabel('Samples')
    plt.show()













