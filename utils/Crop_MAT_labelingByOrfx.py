import os
import cv2
import glob
import numpy as np
import scipy.io as sio
import time
import matlab.engine

def show(x,y,method):
    import matplotlib.pyplot as plt
    plt.figure(dpi=300,figsize=(6,10)) #(12,9)
    plt.subplot(131)
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray') #'gray' #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    # plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(132)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    # plt.title('denoised')
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('../../noise/shot_' + method + '_dn.mat'), {'data': y[:, :, np.newaxis]})

    plt.subplot(133)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    # io.savemat(('../../noise/shot_' + method + '_n.mat'), {'data': noise[:, :, np.newaxis]})
    # plt.title('removed noise')
    plt.tight_layout()
    plt.show()

# cropSize = 512
# num = 50 #100
pch_size=256
stride=128


# Crop SIDD
# GT = glob.glob(os.path.join('G:\datasets\seismic\marmousi', '*35.mat'))
Noisy = glob.glob(os.path.join('G:\datasets\seismic\marmousi', '*35*gn*.mat'))

# GT.sort()
Noisy.sort()

out_dir = "D:\datasets\seismic\marmousi\\f35_s256_o128_orfx"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(os.path.join(out_dir, 'CL')): #CL GT
    os.mkdir(os.path.join(out_dir, 'CL'))
if not os.path.exists(os.path.join(out_dir, 'RN')): #RN Noisy
    os.mkdir(os.path.join(out_dir, 'RN'))

num_patch = 0
for ii in range(len(Noisy)):
    if (ii + 1) % 10 == 0:
        print('    The {:d} original images'.format(ii + 1))
    im_noisy = sio.loadmat(Noisy[ii])['data']  # [:, :, ::-1]
    H, W= im_noisy.shape
    im_gt = sio.loadmat(Noisy[ii])['data']  # [:, :, ::-1]
    ind_H = list(range(0, H - pch_size + 1, stride))
    if ind_H[-1] < H - pch_size:
        ind_H.append(H - pch_size)
    ind_W = list(range(0, W - pch_size + 1, stride))
    if ind_W[-1] < W - pch_size:
        ind_W.append(W - pch_size)
    kk=0
    import matlab
    eng = matlab.engine.start_matlab()
    ii=0
    start_time = time.time()
    for start_H in ind_H:
        for start_W in ind_W:
            if (ii + 1) % 10 == 0:
                print('    The {:d} original images'.format(ii + 1))
            pch_noisy = im_noisy[start_H:start_H + pch_size, start_W:start_W + pch_size,]
            pch_dn1 = eng.fx_decon(matlab.double(pch_noisy.tolist()), matlab.double([0.001]), matlab.double([15]),
                                   matlab.double([0.01]), matlab.double([1]),
                                   matlab.double([124]));  # XJ 0.004 PK 0.002
            noise_data1 = pch_noisy - np.array(pch_dn1)
            pch_dn2, noise_data2, low = eng.localortho(matlab.double(pch_dn1),
                                                       matlab.double(noise_data1.tolist()),
                                                       matlab.double([[20, 20, 1]]),
                                                       100,
                                                       0.0,
                                                       1, nargout=3)  # 20,0,1
            pch_dn = np.array(pch_dn2)
            if ii == 0:
                show(pch_noisy, pch_dn, method='orthofxdecon')
            # pch_gt = im_gt[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
            sio.savemat(os.path.join(out_dir, 'CL', '%d_%d.mat' % (ii, kk)), {'data': np.expand_dims(pch_dn,axis=0)})
            sio.savemat(os.path.join(out_dir, 'RN', '%d_%d.mat' % (ii, kk)), {'data': np.expand_dims(pch_noisy,axis=0)})
            kk+=1
            num_patch += 1
            ii=ii+1
print('running time is:', time.time()-start_time)
print('Total {:d} small images pairs'.format(num_patch))
print('Finish!\n')