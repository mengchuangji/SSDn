import sys
sys.path.append('..')
import cv2
import argparse
from dataset.seis_mat import SeisMatValidationDataset
import numpy as np
import os
import scipy.io as sio

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.option import parse, recursive_print
from utils import util
import scipy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from  validate_.validate_seis import compare_SNR
import time
from model.three_stage import generate_alpha
import matplotlib.pyplot as plt
import scipy.io as sio
def plot_cmap(img,dpi,figsize,data_range,cmap,cbar=False):

    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=data_range[0], vmax=data_range[1], cmap=cmap)
    # plt.title('fake_noisy')
    if cbar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()
def noise_space_cor(noise):
    def correlation_coefficient(T1, T2):
        numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
        denominator = T1.std() * T2.std()
        if denominator == 0:
            return 0
        else:
            result = numerator / denominator
            return result

    spa_cor = np.ones((9, 9))
    H, W = noise.shape[0], noise.shape[1]
    for i in range(-4, 5, 1):
        for j in range(-4, 5, 1):
            if i >= 0 and j >= 0:
                spa_cor[i + 4, j + 4] = correlation_coefficient(noise[i:, j:], noise[:H - i, :W - j])
    A = np.asanyarray(spa_cor[4:, 4:])
    B = np.flipud(A)
    C = np.concatenate((B, A[1:]), axis=0)
    D = C[:, 1:]
    F = np.fliplr(C)
    E = np.concatenate((F, D), axis=1)
    return E

def plot_noise_space_cor(space_cor,dpi,figsize,data_range,cmap,cbar=False):
    fontsize = 14
    # Set font
    fpath = '/home/shendi_mcj/fonts/times.ttf'
    # fpath = 'E:\\times.ttf'
    from matplotlib import font_manager as fm
    prop = fm.FontProperties(fname=fpath, size=fontsize)
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    im=ax.imshow(space_cor, vmin=data_range[0], vmax=data_range[1], cmap=cmap)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04, shrink=1.0)
    # ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.text(0, -1, 'Correlation coefficient', fontsize=fontsize, fontproperties=prop)
    # ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_xticklabels(np.arange(-4, 5, 1), fontdict={'size': fontsize}, fontproperties=prop)
    ax.set_yticks(np.arange(0, 9, 1))  # np.arange(0, 3504/1000, 0.5)
    ax.set_yticklabels(np.arange(-4, 5, 1), fontdict={'size': fontsize}, fontproperties=prop)
    ax.set_xlabel("Horizontal", fontsize=fontsize, fontproperties=prop)
    ax.set_ylabel("Vertical", fontsize=fontsize, fontproperties=prop)
    # plt.title('fake_noisy')
    plt.tight_layout()
    plt.show()

def main(opt):
    test_path= opt['validation_datasets'][0]['args']['dataset_path']
    test_set = SeisMatValidationDataset(path=test_path,patch_size=256, pin_memory= True)
    test_loader = DataLoader(test_set, batch_size=1)

    # if os.path.exists(opt['mat_path']):
    #     os.remove(opt['mat_path'])

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])


    count = 0
    # load info
    # data_dir = 'D:\datasets\seismic\marmousi\\f35_s256_o128\\'
    # original = sio.loadmat(data_dir+'RN\\0_1550.mat')['data'].squeeze()
    # x_max=max(abs(original.max()),abs(original.min()))
    # GT = sio.loadmat(data_dir+'CL\\0_1550.mat')['data'].squeeze()

    data_dir = '/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128/'#_sgm02
    original = sio.loadmat(data_dir + 'RN/0_1550.mat')['data'].squeeze() #0_1550 1250
    x_max = max(abs(original.max()), abs(original.min()))
    GT = sio.loadmat(data_dir + 'CL/0_1550.mat')['data'].squeeze()

    noise=original-GT
    from scipy import signal, ndimage
    # 对原始高斯噪声进行均值滤波
    def generate_correlated_gaussian_noise_2d(original_noise, kernel_size):
        # Generate original 2D Gaussian noise
        # original_noise = np.random.normal(0, 1, shape)
        shape = original_noise.shape
        # Generate 2D Gaussian kernel for mean filtering
        kernel = np.ones((kernel_size, 1)) / kernel_size  # / (kernel_size ** 2) kernel_size // 2
        # Apply mean filtering to introduce correlation
        correlated_noise = signal.convolve2d(original_noise, kernel, mode='same', boundary='wrap')
        return correlated_noise

    correlated_noise= generate_correlated_gaussian_noise_2d(noise,7)
    plot_noise_space_cor(noise_space_cor(correlated_noise), 200, (3.2, 3), data_range=[0, 1], cmap=plt.cm.jet,cbar=True)
    original=GT+correlated_noise


    noisy = original
    noisy = noisy / x_max
    GT = GT / x_max
    plot_cmap(noisy, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    plot_cmap(GT, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    plot_cmap(noisy - GT, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # sio.savemat(('../output/INN_denoise/v2/v2_mms/gt1550.mat'), {'data': GT[:, :]})
    # sio.savemat(('../output/INN_denoise/v2/v2_mms/noisy1550_c7.mat'), {'data': noisy[:, :]})

    data={}
    data['L'] = torch.from_numpy(noisy).view(1, -1, noisy.shape[0], noisy.shape[1]).type(torch.FloatTensor)
    data['H'] = torch.from_numpy(GT).view(1, -1, GT.shape[0], GT.shape[1]).type(torch.FloatTensor)
    # generate alpha
    with torch.no_grad():
        BNN = model.networks['BNN'](data['L'])
        LAN = model.networks['LAN'](data['L'])
        UNet = model.networks['UNet'](data['L'])
    alpha = generate_alpha(BNN, lower=0.05, upper=0.2)
    # alpha = generate_alpha(BNN)
    plot_cmap(alpha.cpu().squeeze().numpy(), 300, (4, 3), data_range=[0.45,0.55], cmap=plt.cm.jet,cbar=True)
    plot_cmap((1-alpha).cpu().squeeze().numpy(), 300, (4, 3), data_range=[0.45,0.55], cmap=plt.cm.jet,cbar=True)
    # plot_cmap(BNN.cpu().squeeze().numpy(), 300, (3, 3),data_range=[-1,1], cmap=plt.cm.seismic)
    # plot_cmap(noisy - BNN.cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # plot_cmap(LAN.cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # plot_cmap(noisy - LAN.cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # plot_cmap(UNet.cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1,1], cmap=plt.cm.seismic)
    # plot_cmap(noisy - UNet.cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # # plot_cmap((BNN*(1-alpha)+LAN * alpha).cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1,1], cmap=plt.cm.seismic)
    # sio.savemat(('../output/marmousi_eps/bnn.mat'), {'data': BNN.cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/marmousi_eps/lan.mat'), {'data': LAN.cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/marmousi_eps/unet_dn.mat'), {'data': UNet.cpu().squeeze().numpy()[:, :]})
 #    sio.savemat(('../output/marmousi/alpha.mat'), {'data': alpha.cpu().squeeze().numpy()[:, :]})

    # denosied
    start_time = time.time()
    output = model.validation_step(data)
    end_time= time.time()
    output = output.cpu().squeeze().numpy()
    psnr_b = peak_signal_noise_ratio(GT, noisy, data_range=2)
    psnr_a = peak_signal_noise_ratio(GT, output, data_range=2)
    ssim_b = structural_similarity(GT, noisy, data_range=2)
    ssim_a = structural_similarity(GT, output, data_range=2)
    snr_b = compare_SNR(GT, noisy)
    snr_a = compare_SNR(GT, output)
    print(
        'testing...  PSNR_b : %.2f dB, PSNR_a : %.2f dB, SNR_b : %.2f dB, SNR_a : %.2f dB, SSIM_b : %.4f, SSIM_a : %.4f, time : %.2f s' % (
       psnr_b, psnr_a, snr_b, snr_a, ssim_b, ssim_a, (end_time - start_time)))
    plot_cmap(output, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    plot_cmap(noisy-output, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # sio.savemat(('../output/marmousi_eps/inn_dn.mat'), {'data': output[:, :]})
    from seis_utils.plotfunction import show_GT_NY_Dn_Rn
    show_GT_NY_Dn_Rn(x=GT, y=noisy, x_=output, dpi=300,
                     figsize=(12, 4))  # [0:64,0:64]
    sio.savemat(('../output/INN_denoise/v2/v2_mms/x1_1550_c7.mat'), {'data': output[:, :]})

    # from seis_utils.localsimi import localsimi
    # simi = localsimi(output, noisy-output, rect=[3, 3, 1], niter=20, eps=0.0, verb=1).squeeze()
    # energy_simi = np.sum(simi) / simi.size
    # print("energy_simi=", energy_simi)
    # plot_cmap(simi, 300, (3, 3), data_range=[0, 1], cmap=plt.cm.jet)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    #four_stage_seis_invDN.json three_stage_SASS
    parser.add_argument("--config_file", type=str, default='../option/three_stage_SASS.json')#three_stage_seis_invDN.json
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    opt['mat_path'] = 'SubmitSrgb.mat'
    opt['ensemble'] = False

    # opt['iter']=30000 #
    opt['BNN_iters'] = 10000  #
    opt['LAN_iters'] = 10000  #
    opt['UNet_iters'] = 10000  #
    # opt['log_dir']= 'logs/XJ_3S_invDN'
    # opt['resume_from']='..\\train\logs\XJ2\model_iter_00030000.pth'
    # opt['resume_from']='../train/logs/4S_invDN_MmsF35_2/model_iter_00010000.pth'
    # opt['validation_datasets'][0]['args']['dataset_path'] = 'D:\datasets\seismic\marmousi\\f35_s256_o128'
    # opt['resume_from'] = '../train/logs/BNN_LAN_DN_MmsF35_sgm02/model_iter_00030000.pth'
    opt['resume_from'] = '../train/logs/BNN_LAN_DN_MmsF35_c7/model_iter_00030000.pth'

    opt['validation_datasets'][0]['args']['dataset_path'] = '/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128'
    recursive_print(opt)
    main(opt)
