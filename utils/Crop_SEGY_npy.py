import os
import cv2
import glob
import numpy as np
import scipy.io as sio

# cropSize = 512
# num = 50 #100
pch_size=256
stride=128


# Crop SIDD
# GT = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\盘客数据\\test', '*_dn.s*gy'))
# Noisy = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\盘客数据\\test', '*443.s*gy'))

# GT = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\BGP', '*600_dn.s*gy'))
# Noisy = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\BGP', '*600.s*gy'))

# GT = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\\2021.3.27数据', '*857_dn.s*gy'))
# Noisy = glob.glob(os.path.join('E:\博士期间资料\田亚军\田博给的数据\\2021.3.27数据', '*857.s*gy'))

GT = glob.glob(os.path.join('/home/shendi_mcj/datasets/seismic/test', '120-Y.s*gy'))
Noisy = glob.glob(os.path.join('/home/shendi_mcj/datasets/seismic/test', '120.s*gy'))

# from utils.seis_utils.readsegy import readsegy
# clean_patches = np.array(readsegy(data_dir=clean_mat_file_path, file='00-Y.sgy'))
# noisy_patches = np.array(readsegy(data_dir=noisy_mat_file_path, file='00.sgy'))

GT.sort()
Noisy.sort()

# out_dir = "E:\博士期间资料\田亚军\田博给的数据\盘客数据\\test\\PK_test_s256_o128"
# out_dir = "D:\datasets\seismic\BGP\mhs\\mhs_s256_o128"
# out_dir = "D:\datasets\seismic\hty\hty_test_s128_o64"

out_dir = '/home/shendi_mcj/datasets/seismic/fielddata/train/prep/SEGY_s256_o128_120_npy'

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
        print(f'    The {ii + 1} original images')

    # 读入图像
    im_gt = np.array(readsegy(file=GT[ii]))
    im_noisy = np.array(readsegy(file=Noisy[ii]))
    noisy_max = abs(im_noisy).max()

    H, W = im_gt.shape
    num_trace = 500  # 一段 section 中包含的 trace 数量
    num_section = W // num_trace  # 总共可分为多少个 section（strip）

    kk = 0
    for ss in range(num_section):
        if (ss + 1) % 1 == 0:
            print(f'    The {ss + 1} sections')

        start_W_section = ss * num_trace
        end_W_section = start_W_section + num_trace

        # 裁剪 section
        gt_section = im_gt[:, start_W_section:end_W_section]
        noisy_section = im_noisy[:, start_W_section:end_W_section]

        # 二级 patch 划分
        section_H, section_W = gt_section.shape

        ind_H = list(range(0, section_H - pch_size + 1, stride))
        if ind_H[-1] < section_H - pch_size:
            ind_H.append(section_H - pch_size)
        ind_W = list(range(0, section_W - pch_size + 1, stride))
        if ind_W[-1] < section_W - pch_size:
            ind_W.append(section_W - pch_size)

        for start_H in ind_H:
            for start_W in ind_W:
                patch_gt = gt_section[start_H:start_H + pch_size, start_W:start_W + pch_size] / noisy_max
                patch_noisy = noisy_section[start_H:start_H + pch_size, start_W:start_W + pch_size] / noisy_max

                np.save(os.path.join(out_dir, 'CL', f'{ii}_{kk}.npy'), np.expand_dims(patch_gt, axis=0))
                np.save(os.path.join(out_dir, 'RN', f'{ii}_{kk}.npy'), np.expand_dims(patch_noisy, axis=0))
                kk += 1
                num_patch += 1

print(f'Total {num_patch} small images pairs')
print('Finish!\n')