# -*- coding: utf-8 -*
import argparse
from torch.autograd import Variable
from torchvision import transforms
from util.Datasets2 import MyDatasets
import time
import torch
import torch.nn as nn
from model.SCRN import SCRN
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from util.My_tool1 import save_csv, produce_csv

# 添加参数
parser = argparse.ArgumentParser(description='PyTorch SCRN')
parser.add_argument('--model', default='SCRN', type=str, help='choose a type of model')
parser.add_argument('--train_data_dir', default=r'/workspace/Paper2/TrainData11', type=str,
                    help='path of train original-test_data')
parser.add_argument('--epoch', default=80, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--patch_size', default=(128, 128), type=int, help='patch size')
args = parser.parse_args()


if __name__ == '__main__':

    # parameter
    Input_root_dir = args.train_data_dir
    patch_size = args.patch_size
    batch_size = args.batch_size

    # model
    print('====> Building  SCRN model')
    model = SCRN()

    # cuda
    cuda = torch.cuda.is_available()
    device = torch.device('cuda')
    criterion = nn.MSELoss(reduction='sum').to(device)
    if cuda:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 动态调整学习速率
    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    # milestones为一个数组，如 [50,70]. gamma为倍数。如果learning rate开始为0.01 ，则当epoch为50时变为0.001，epoch 为70 时变为0.0001。
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.2)  # learning rates

    # dataloader
    data_trans = transforms.Compose([transforms.ToTensor()])
    data = MyDatasets(Input_root_dir=Input_root_dir, Target_root_dir=Input_root_dir, transform=data_trans)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    step = 0
    train_n = args.epoch
    time_open = time.time()
    produce_csv('SCRN_loss.cvs')
    for epoch in range(train_n):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for (Input, Target) in dataloader:
            # 将数据放在GPU上训练
            X, Y = Variable(Input).to(device), Variable(Target).to(device)
            X = X.type(torch.float32)
            Y = Y.type(torch.float32)

            y_pred = model(X)
            optimizer.zero_grad()
            loss = criterion(y_pred, Y)
            loss.backward()
            optimizer.step()

            # 损失和
            running_loss += loss.data.item()
            epoch_loss = running_loss * batch_size / len(data)
            step += batch_size
        elapsed_time = time.time() - start_time
        print('\repoch:{} Loss:{:.4f} step:{} time:{:.4f} '.format(epoch+1,epoch_loss, step, elapsed_time), end=' ',flush=True)
        scheduler.step()
        save_csv("/workspace/Paper2/SCRN/SCRN_loss.cvs", epoch+1, epoch_loss, 0)
        torch.save(model, '/workspace/Paper2/SCRN/train_model/model_%03d.pth' % (epoch + 1))

    #     #########################################
    #     # TODO：开启测试
    #     model.eval()
    #
    #     data_path = r'..\Data\Numpy_DATA\test\theoretical_1NoiseData1.npy'
    #     original_data_path = r'..\Data\Numpy_DATA\Original\theoreticalData1.npy'
    #     # 读取数据
    #     data_input_z = np.load()
    #     # print(np.min(data_input_z))
    #     # 数据转换为tensor
    #     data_input_z_tensor = data_trans(data_input_z)
    #     data_input_z_tensor = data_input_z_tensor.unsqueeze(0)
    #     # 将数据放入GPU
    #     data_input_z_tensor = (Variable(data_input_z_tensor).to(device)).type(torch.float32)
    #     data_input_s = np.load()
    #
    #     # 模型预测概率
    #     y_pred = model(data_input_z_tensor)
    #     # 将数据降维
    #     out = y_pred.squeeze(0)
    #     # 将数据从gpu放入cpu并再次降维
    #     imag_narry = ((out.squeeze(0)).cuda().test_data.cpu()).detach().numpy()
    #
    #     out_data = imag_narry
    #
    #     print()
    #     print('去噪前, 峰值信噪比为:{}'.format(tool.__mtx_similar2__(data_input_z, data_input_s)))
    #     print('去噪后, 峰值信噪比为:{}'.format(tool.__mtx_similar2__(out_data, data_input_s)))
    #     print('去噪前, 信噪比为:{}'.format(tool.__SNR__(data_input_z, data_input_s)))
    #     print('去噪后, 信噪比为:{}'.format(tool.__SNR__(out_data, data_input_s)))
    #     print('去噪前, 结构相似性：{}'.format(sk_cpt_ssim(data_input_z, data_input_s)))
    #     print('去噪后, 结构相似性：{}'.format(sk_cpt_ssim(out_data, data_input_s)))
    #
    # time_end = time.time() - time_open
