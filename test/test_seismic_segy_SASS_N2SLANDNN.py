import sys
sys.path.append('..')
import cv2
from utils.DAN_util import *
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
from model.three_stage_SASS_v2 import generate_alpha,std,generate_alpha_1
import matplotlib.pyplot as plt
def plot_cmap(img,dpi,figsize,data_range,cmap,cbar=False):

    plt.figure(dpi=dpi, figsize=figsize)
    # plt.imshow(img, vmin=data_range[0], vmax=data_range[1], cmap=cmap)
    ax = plt.gca()
    im = ax.imshow(img, vmin=data_range[0], vmax=data_range[1], cmap=cmap)

    # plt.title('fake_noisy')
    if cbar:
        # plt.colorbar()
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04, shrink=1.0)

    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()

def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # test_path = opt['validation_datasets'][0]['args']['path']
    # test_set = SeisMatValidationDataset(path=test_path,patch_size=256, pin_memory= True)
    # test_loader = DataLoader(test_set, batch_size=1)

    # if os.path.exists(opt['mat_path']):
    #     os.remove(opt['mat_path'])

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])

    count = 0

    # load info
    # data_dir = '/home/shendi_mcj/datasets/seismic/fielddata/'
    # data_dir = 'E:\博士期间资料\田亚军\田博给的数据\\2021.6.07 新疆数据\\'
    data_dir = '/home/shendi_mcj/datasets/seismic/test/'
    # data_dir = 'E:\博士期间资料\田亚军\田博给的数据\盘客数据\\test\\'

    im = '00-L120.sgy'
    from seis_utils.readsegy import readsegy
    original = readsegy(data_dir + '00-L120.sgy')#[50:50+128,128:128+128] #[50:50+128,500-64:500]
    # original = readsegy(data_dir+'PANKE-INline443.sgy')[:64,:64]#[50:50+64,500-64:500]#[0:480,0:480][300:780,0:480][76:876,0:480] [50:50+128,500-128:500]
    # original = readsegy(data_dir + 'PANKE-INline443.sgy')[100:228, 0:128]#[-300:-100,0:200]#[2200:2328, 0:128]# #[100:228, 0:128]  [2200:2328, 0:128]
    # original = readsegy('E:\博士期间资料\田亚军\田博给的数据\盘客数据\pk-00-L21-40-t400-4000.sgy')[1600:1728, 0:128]



    x_max=max(abs(original.max()),abs(original.min()))
    GT = readsegy(data_dir + '00-L120-Y.sgy')#[50:50+64,500-64:500]
    noisy = original  # [50:50+64,500-64:500]
    GT = GT / x_max
    noisy = original#[50:50+64,500-64:500]
    noisy = noisy / x_max

    # plot_cmap(noisy, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)

    data={}
    data['L'] = torch.from_numpy(noisy).view(1, -1, noisy.shape[0], noisy.shape[1]).type(torch.FloatTensor)

    # from model.three_stage_SASS_v2 import Masker
    # masker = Masker(width=4, mode='interpolate',
    #                 pobability=0.5)  # mode='interpolate' 'tracewise' 'zero'
    # net_input, mask = masker.mask(X=data['L'], i=1 % (4 ** 2 - 1))
    # plot_cmap(mask.cpu().squeeze().numpy(), 300, (3, 3), data_range=[0, 1], cmap='gray', cbar=False)  # viridis
    # plot_cmap((1-mask).cpu().squeeze().numpy(), 300, (3, 3), data_range=[0, 1], cmap='gray', cbar=False)  # viridis
    # plot_cmap((data['L']*mask.to(data['L'].device)).cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=False)


    with torch.no_grad():
        Pad=PadUNet(data['L'], dep_U=6)
        data['L'] = Pad.pad()

        N2SUNet_pad = model.networks['N2SUNet'](data['L']) # N2SUNet  UNet data['L']
        # Pad2 = PadUNet(net_input.to(data['L'].device), dep_U=6)
        # net_input = Pad2.pad()
        # N2SUNet_pad = model.networks['UNet'](net_input)
        # LAN_pad = model.networks['LAN'](data['L'])
        # UNet_pad = model.networks['UNet'](data['L'])


    # alpha_pad = generate_alpha(N2SUNet_pad,lower=0.05, upper=0.2)
    N2SUNet = Pad.pad_inverse(N2SUNet_pad)
    # LAN = Pad.pad_inverse(LAN_pad)
    # UNet = Pad.pad_inverse(UNet_pad)
    # alpha = Pad.pad_inverse(alpha_pad)
    alpha = generate_alpha_1(N2SUNet, lower=0.05, upper=0.20, window_size=7)  # lower=0.05, upper=0.20
    # sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_N2S_LAN_DN_bp_std.mat'), {'data': std(N2SUNet, window_size=7).cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_N2S_LAN_DN_bp_alpha.mat'), {'data': alpha.cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_N2S_LAN_DN_bt_std_w7.mat'), {'data': std(N2SUNet, window_size=7).cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_N2S_LAN_DN_bt_alpha_l005u020.mat'), {'data': alpha.cpu().squeeze().numpy()[:, :]})

    #  plot_cmap(std(N2SUNet, window_size=7).cpu().squeeze().numpy(), 300, (3.3, 3), data_range=[0, 1], cmap='viridis',
    #           cbar=True)
    # sio.savemat(('../output/INN_denoise/PK/PK_4S_N2S_invDN_bt_std.mat'), {'data': std(N2SUNet, window_size=7).cpu().squeeze().numpy()[:, :]})
    plot_cmap(alpha.cpu().squeeze().numpy(), 300, (3.3, 3), data_range=[0.45,0.55], cmap=plt.cm.jet,cbar=True) #viridis
    # plot_cmap((1-alpha).cpu().squeeze().numpy(), 300, (3.3, 3), data_range=[0.0,1.0], cmap='viridis',cbar=True)
    # plot_cmap(N2SUNet.cpu().squeeze().numpy(), 300, (3, 3),data_range=[-1,1], cmap=plt.cm.seismic)
    # plot_cmap((N2SUNet*mask.to(N2SUNet.device)).cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # plot_cmap(LAN.cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # plot_cmap(UNet.cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1,1], cmap=plt.cm.seismic)
    # sio.savemat(('../output/INN_denoise/x_tilde.mat'), {'data': UNet.cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/INN_denoise/x_tilde_n.mat'), {'data': noisy-UNet.cpu().squeeze().numpy()[:, :]})
    # plot_cmap(noisy - UNet.cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # plot_cmap((BNN*(1-alpha)+LAN * alpha).cpu().squeeze().numpy(), 300, (3, 3), data_range=[-1,1], cmap=plt.cm.seismic)

    # denosied
    start_time = time.time()
    output = model.validation_step(data)
    end_time= time.time()
    output=Pad.pad_inverse(output)
    output = output.cpu().squeeze().numpy()
    sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_N2S_LAN_DN_bt_x3.mat'), {'data': output[:, :]})
    sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_N2S_LAN_DN_bt_x3_n.mat'), {'data': noisy-output[:, :]})
    # sio.savemat(('../output/INN_denoise/PK/PANKE-INline443.mat'), {'data': noisy})
    # sio.savemat(('../output/INN_denoise/PK/PK_2S_N2S_invDN_x_tilde_dn.mat'), {'data': output[:, :]})
    # sio.savemat(('../output/INN_denoise/PK/PK_2S_N2S_invDN_x_tilde_n.mat'), {'data': noisy-output[:, :]})
    # plot_cmap(output, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)

    from seis_utils.plotfunction import show_GT_NY_Dn_Rn,show_NY_Dn_Rn
    # show_NY_Dn_Rn(y=noisy, x_=output,figsize=(9, 3))  # [0:64,0:64]
    show_NY_Dn_Rn(y=noisy, x_=output, figsize=(18, 9))  # [0:64,0:64]
    psnr_b = peak_signal_noise_ratio(GT, noisy, data_range=2)
    psnr_a = peak_signal_noise_ratio(GT, output, data_range=2)
    ssim_b = structural_similarity(GT, noisy, data_range=2)
    ssim_a = structural_similarity(GT, output, data_range=2)
    snr_b = compare_SNR(GT, noisy)
    snr_a = compare_SNR(GT, output)
    print(
        'testing...  PSNR_b : %.2f dB, PSNR_a : %.2f dB, SNR_b : %.2f dB, SNR_a : %.2f dB, SSIM_b : %.4f, SSIM_a : %.4f, time : %.2f s' % (
       psnr_b, psnr_a, snr_b, snr_a, ssim_b, ssim_a, (end_time - start_time)))
    from seis_utils.plotfunction import show_GT_NY_Dn_Rn
    show_GT_NY_Dn_Rn(x=GT, y=noisy, x_=output, dpi=300,
                     figsize=(12, 4))  # [0:64,0:64]

    # from utils.test_local_similarity import avg_correlation_coefficient
    # local_cc, cc_map = avg_correlation_coefficient(output, noisy-output, window_size=4)
    # print("local correlation_coefficient:", local_cc)
    # plot_cmap(cc_map, 300, (3, 3), data_range=[0, 1], cmap=plt.cm.jet, cbar=True)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description="Train the denoiser")
    # two_stage_seis_BSN_invDN four_stage_seis_invDN
    parser.add_argument("--config_file", type=str, default='../option/three_stage_SASS_v2.json')#
    #two_stage_seis_BSN_invDN four_stage_seis_invDN two_stage_seis_N2S_invDN  four_stage_seis_invDN_v2
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    opt['mat_path'] = 'SubmitSrgb.mat'
    opt['ensemble'] = False

    # opt['iter']=10000 #
    # opt['BNN_iters'] = 10000  #
    # opt['LAN_iters'] = 10000  #
    # opt['UNet_iters'] = 10000  #


    opt['N2SUNet_iters']= 2000
    opt['mode'] = 'tracewise' # 'interpolate' 'tracewise'
    opt['LAN_iters'] =2000
    opt['UNet_iters'] =2000
    opt['networks'][1]['args']['receptive_feild']=3


    # opt['log_dir']= 'logs/PK/4S_invDN'
    # opt['resume_from']='..\\train\logs\XJ2\model_iter_00030000.pth
    # opt['resume_from']='../train/logs/PK/4S_invDN_bs9/model_iter_00020000.pth'
    # opt['resume_from'] = '../train/logs/PK/4S_N2S_invDN_bt/model_iter_00010000.pth' #2S_invDN N2S_invDN_bt
    opt['resume_from'] = '../train/logs/XJ/N2S_LAN_DN_bt_bst/model_iter_00012000.pth'
    # opt['validation_datasets'][0]['args']['path'] = 'D:\datasets\seismic\BGP\mhs\mhs_s256_o128'
    opt['validation_datasets'][0]['args']['path'] = '/home/shendi_mcj/datasets/seismic/fielddata/train/prep/SEGY_s256_o128'
    recursive_print(opt)
    main(opt)
