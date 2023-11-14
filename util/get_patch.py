import cv2
import glob
import segyio
from torch.utils.data import Dataset
import torch
import numpy as np
from gain import *
from download_data import *
import matplotlib.pyplot as plt
import random
import numbers

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DownsamplingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): data_445 test_data patches
        rate: test_data sampling rate when regular=False, e.g. 0.3
              test_data sampling interval when regular=True
    """

    def __init__(self, xs, rate, regular=False):
        super(DownsamplingDataset, self).__init__()
        self.xs = xs
        self.rate = rate
        self.regular = regular

    def __getitem__(self, index):
        batch_x = self.xs[index]
        # the type of the test_data must be tensor
        if self.regular:
            mask = regular_mask(batch_x, self.rate)
        else:
            mask = irregular_mask(batch_x, self.rate)
        batch_y = mask.mul(batch_x)

        return batch_y, batch_x, mask

    def __len__(self):
        return self.xs.size(0)


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): data_445 test_data patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, xs):
        super(DenoisingDataset, self).__init__()
        self.xs = xs

    def __getitem__(self, index):
        sigma = (random.sample([10, 30, 50, 70, 90, 110], 1))[0]
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(sigma / 255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def irregular_mask(data, rate):
    """the mask matrix of random sampling
    Args:
        data: original test_data patches
        rate: sampling rate,range(0,1)
    """
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(), dtype=torch.float64)  # tensor
    v = round(n * rate)
    TM = random.sample(range(n), v)
    mask[:, :, TM] = 1  # missing by column
    return mask


def regular_mask(data, a):
    """the mask matrix of regular sampling
    Args:
        data: original test_data patches
        a(int): sampling interval, e.g: TestData11_picture = 5, sampling like : 100001000010000
    """
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(), dtype=torch.float64)
    for i in range(n):
        if (i + 1) % a == 1:
            mask[:, :, i] = 1
        else:
            mask[:, :, i] = 0
    return mask


def patch_show(train_data, save=False, root=''):
    '''
    show some sampels of train test_data
    save: save or not save the showed sample
    root(path)：if save=True, the test_data will be saved to this path(as TestData11_picture .png picture)
    '''
    samples = 8
    idxs = np.random.choice(len(train_data), samples, replace=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i, idx in enumerate(idxs):
        plt_idx = i + 1
        data = train_data[idx]
        y, x = np.reshape(data[0], (data[0].shape[1], data[0].shape[2])), np.reshape(data[1], (
        data[1].shape[1], data[1].shape[2]))

        plt.subplot(2, samples, plt_idx)
        plt.imshow(x, cmap=plt.cm.seismic,
                   interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
        plt.axis('off')
        plt.subplot(2, samples, plt_idx + samples)
        plt.imshow(y, cmap=plt.cm.seismic,
                   interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
        plt.axis('off')
    plt.show()


def data_aug(img, mode=None):
    # test_data augmentation
    if mode == 0:
        # original
        return img
    if mode == 1:
        # flip up and down
        return np.flipud(img)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(img)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(img))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(img, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(img, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(img, k=3))


def progress_bar(temp_size, total_size, patch_num, file, file_list):
    done = int(50 * temp_size / total_size)
    sys.stdout.write("\r[%s/%s][%s%s] %d%% %s" % (
    file + 1, file_list, '#' * done, ' ' * (50 - done), 100 * temp_size / total_size, patch_num))
    sys.stdout.flush()


def _compute_n_patches(i_h, i_w, p_h, p_w, s_h, s_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.
    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image width
    p_h : int
        The height of TestData11_picture patch
    p_w : int
        The width of TestData11_picture patch
    s_h : int
        the moving step in the image height
    s_w: int
        the moving step in the image width
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is TestData11_picture float
        between 0 and 1, it is taken to be TestData11_picture proportion of the total number
        of patches.
    extraction_step：moving step
    """
    n_h = np.floor((i_h - p_h) / s_h) + 1
    n_w = np.floor((i_w - p_w) / s_w) + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Integral))
              and max_patches >= all_patches):
            return all_patches
        elif (isinstance(max_patches, (numbers.Real))
              and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def _compute_total_patches(h, w, p_h, p_w, s_h, s_w, aug_times=[], scales=[], max_patches=None):
    num = 0
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        num += _compute_n_patches(h_scaled, w_scaled, p_h, p_w, s_h, s_w, max_patches=None) * (aug_times + 1)
    return num


def gen_patches(data, patch_size=(64, 64), stride=(32, 32), file=1, file_list=2, total_patches_num=None,
                train_data_num=float('inf'), patch_num=None, aug_times=[], scales=[], q=None, single_patches_num=None,
                verbose=None):
    '''
    Args:
        aug_time(list): Corresponding function data_aug, if aug_time=[],mean don`t use the aug
        scales(list): test_data scaling; default scales = [],mean that the test_data don`t perform scaling,
                      if perform scaling, you can set scales=[0.9,0.8,...]
    '''
    # read test_data
    h, w = data.shape
    p_h, p_w = patch_size
    s_h, s_w = stride

    patches = []
    num = q * single_patches_num

    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        data_scaled = cv2.resize(data, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        for i in range(0, h_scaled - p_h + 1, s_h):
            for j in range(0, w_scaled - p_w + 1, s_w):
                x = data_scaled[i:i + p_h, j:j + p_w]

                if sum(sum(x)) != 0 and x.std() > 1e-3 and x.shape == patch_size:
                    num += 1
                    patch_num += 1
                    patches.append(x)
                    if verbose:
                        progress_bar(num, total_patches_num, patch_num, file, file_list)

                    if patch_num >= train_data_num:
                        return patches, patch_num

                    for k in range(0, aug_times):
                        x_aug = data_aug(x, mode=np.random.randint(0, 8))
                        num += 1
                        patch_num += 1
                        patches.append(x_aug)
                        if verbose:
                            progress_bar(num, total_patches_num, patch_num, file, file_list)

                        if patch_num >= train_data_num:
                            return patches, patch_num
                elif verbose:
                    num = num + 1 + aug_times
                    progress_bar(num, total_patches_num, patch_num, file, file_list)
    return patches, patch_num


def datagenerator(data_dir, patch_size=(128, 128), stride=(32, 32), train_data_num=float('inf'), download=True,
                  datasets="Hess_VTI", aug_times=0, scales=[1], verbose=True, jump=1, agc=False):
    '''
    Args:
        data_dir: the path of the .segy file exit
        patch_size: the size the of patch
        stride: when get patches, the step size to slide on the test_data

        train_data_num: int or float('inf'),default=float('inf'),mean all the test_data will be used to Generate patches,
                        if you just need 3000 patches, you can set train_data_num=3000;
        download: bool; if you will download the dataset from the internet
        datasets: the num of the datasets will be download,if download = True
        aug_times: int, the time of the aug you will perform,used to increase the diversity of the samples,in each time,
                    Choose one operation at TestData11_picture time,eg:flip up and down、rotate 90 degree and flip up and down
        scales: list,The ratio of the test_data being scaled . default = [1],Not scaled by default.
        verbose: bool, Whether to output the generate situation of the patches

        jump: default=1, mean that read every shot test_data; when jump>=2, mean that don`t read the shot one by one
                instead of with TestData11_picture certain interval

        agc: if use the agc of the test_data
    '''

    if download:  # download=False
        if datasets > 0:  
            Download_data(data_dir, datasets=datasets) 
        else:
            print("=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>")
            print("Please input the num of the dataset to download ")
            print("=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>")

    print('=> Generating patch samples')
    file_list = glob.glob(data_dir + '/*.segy') + glob.glob(data_dir + '/*.sgy')  # 获取segy文件或者sgy文件
    print(file_list)
    all_patches = []
    patch_num = 0
    for i in range(len(file_list)):

        with segyio.open(file_list[i], 'r', ignore_geometry=True) as f:
            f.mmap()  
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)  
            shot_num = len(set(sourceX))  

            if (trace_num == shot_num ):
                shot_num = 1
                len_shot = trace_num
            elif(shot_num==1):
                shot_num = 1
                len_shot = trace_num
            else :
                len_shot = trace_num // shot_num  
            '''
            The test_data of each shot is read separately
            The default is that the test_data dimensions collected by all shots in the file are the same.
            Jump=1, which means that the test_data of all shots in the file is read by default. 
            When jump=2, it means that every other shot reads test_data.
            '''
            q = -1
            for j in range(0, shot_num, jump):
                data = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
                q += 1
                if agc:  
                    print("agc")
                    data = gain(data, 0.004, 'agc', 0.05, 1)
                else:
                    data = data / np.max(abs(data))

                # Number of shots used to generate the patch
                select_shot_num = len(list(range(0, shot_num, jump)))  
                h, w = data.shape
                p_h, p_w = patch_size
                s_h, s_w = stride
                single_patches_num = int(
                    _compute_total_patches(h, w, p_h, p_w, s_h, s_w, aug_times, scales, max_patches=None))

                if verbose:
                    total_patches_num = single_patches_num * select_shot_num
                    # Break into small pieces.
                    patches, patch_num = gen_patches(data, patch_size, stride, i, len(file_list), total_patches_num,
                                                     train_data_num, patch_num, aug_times, scales, q,
                                                     single_patches_num, verbose)
                else:
                    patches, patch_num = gen_patches(data, patch_size, stride, i, len(file_list),
                                                     train_data_num=train_data_num, patch_num=patch_num,
                                                     aug_times=aug_times, scales=scales, q=q,
                                                     single_patches_num=single_patches_num)

                for patch in patches:
                    all_patches.append(patch)
                    if len(all_patches) >= train_data_num:
                        f.close()
                        if verbose:
                            print(' ')

                        all_patches = np.expand_dims(all_patches, axis=3)
                        print(str(len(all_patches)) + ' ' + 'training test_data finished')
                        return all_patches

            if verbose:
                print(' ')
            f.close()

    all_patches = np.expand_dims(all_patches, axis=3)

    #   When the number of generated patches is an integer multiple of batch, run the following two lines of code.
    #   discard_n = len(all_patches)-len(all_patches)//batch_size*batch_size
    #   all_patches = np.delete(all_patches, range(discard_n), axis=0)
    print(str(len(all_patches)) + ' ' + 'training test_data finished')

    return all_patches


if __name__ == '__main__':
    '''
    root (string): the .segy file exists or will be saved to if download is set to True.
    train patch_size=(128, 128), stride=(48, 48)，aug_times=4, i+1
    """

    root = 'data/train/' #The path to load the dataset.
    train_data = datagenerator(data_dir=root, patch_size=(128, 128), stride=(48, 48), train_data_num=10,
                               download=False, datasets=0, aug_times=0, scales=[1], verbose=True, jump=1, agc=False)
    train_data = train_data.astype(np.float32)
    print(train_data.shape)
    xs = train_data.transpose((0, 3, 1, 2))
    print(xs.shape)
    for i in range(len(xs)):
        np.save('TestData11/test_data_%d' % (i + 1), xs[i][0]) #The path to save.

    # root = 'test_data/real_noise/'
    # real_noise_data = datagenerator(data_dir=root, patch_size=(128, 128), stride=(1, 1), train_data_num=2000,
    #                            download=False, datasets=0, aug_times=5, scales=[1], verbose=True, jump=1, agc=True)
    # real_noise_data = real_noise_data.astype(np.float32)
    # xs = real_noise_data.transpose((0, 3, 1, 2))
    # for i in range(len(xs)):
    #     np.save('../Real_noise/real_noise_data_%d' % (i+1), xs[i][0])





