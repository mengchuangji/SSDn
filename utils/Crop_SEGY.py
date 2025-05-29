import os
import cv2
import glob
import numpy as np
import scipy.io as sio

# cropSize = 512
# num = 50 #100
pch_size=128
stride=64


# Crop SIDD
# GT = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\盘客数据\\test', '*_dn.s*gy'))
# Noisy = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\盘客数据\\test', '*443.s*gy'))

# GT = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\BGP', '*600_dn.s*gy'))
# Noisy = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\BGP', '*600.s*gy'))

GT = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\\2021.3.27数据', '*857_dn.s*gy'))
Noisy = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\\2021.3.27数据', '*857.s*gy'))

# from utils.seis_utils.readsegy import readsegy
# clean_patches = np.array(readsegy(data_dir=clean_mat_file_path, file='00-Y.sgy'))
# noisy_patches = np.array(readsegy(data_dir=noisy_mat_file_path, file='00.sgy'))

GT.sort()
Noisy.sort()

# out_dir = "E:\博士期间资料\田亚军\田博给的数据\盘客数据\\test\\PK_test_s256_o128"
# out_dir = "D:\datasets\seismic\BGP\mhs\\mhs_s256_o128"
out_dir = "D:\datasets\seismic\hty\hty_test_s128_o64"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(os.path.join(out_dir, 'CL')): #CL GT
    os.mkdir(os.path.join(out_dir, 'CL'))
if not os.path.exists(os.path.join(out_dir, 'RN')): #RN Noisy
    os.mkdir(os.path.join(out_dir, 'RN'))

from utils.seis_utils.readsegy import readsegy
num_patch = 0
for ii in range(len(GT)):
    if (ii + 1) % 10 == 0:
        print('    The {:d} original images'.format(ii + 1))
    # im_noisy = sio.loadmat(Noisy[ii])['data']  # [:, :, ::-1]
    im_noisy = np.array(readsegy(file=Noisy[ii]))
    noisy_max = abs(im_noisy).max()
    H, W= im_noisy.shape
    # im_gt = sio.loadmat(GT[ii])['data']  # [:, :, ::-1]
    im_gt = np.array(readsegy(file=GT[ii]))
    ind_H = list(range(0, H - pch_size + 1, stride))
    if ind_H[-1] < H - pch_size:
        ind_H.append(H - pch_size)
    ind_W = list(range(0, W - pch_size + 1, stride))
    if ind_W[-1] < W - pch_size:
        ind_W.append(W - pch_size)
    kk=0
    for start_H in ind_H:
        for start_W in ind_W:
            pch_noisy = im_noisy[start_H:start_H + pch_size, start_W:start_W + pch_size,]/noisy_max
            pch_gt = im_gt[start_H:start_H + pch_size, start_W:start_W + pch_size, ]/noisy_max
            sio.savemat(os.path.join(out_dir, 'CL', '%d_%d.mat' % (ii, kk)), {'data': np.expand_dims(pch_gt,axis=0)})
            sio.savemat(os.path.join(out_dir, 'RN', '%d_%d.mat' % (ii, kk)), {'data': np.expand_dims(pch_noisy,axis=0)})
            kk+=1
            num_patch += 1
print('Total {:d} small images pairs'.format(num_patch))
print('Finish!\n')