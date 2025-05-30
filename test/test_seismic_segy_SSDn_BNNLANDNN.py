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
from model.three_stage import generate_alpha,std,generate_alpha_1
import matplotlib.pyplot as plt
def plot_cmap(img,dpi,figsize,data_range,cmap,cbar=False):

    plt.figure(dpi=dpi, figsize=figsize)
    # plt.imshow(img, vmin=data_range[0], vmax=data_range[1], cmap=cmap)
    ax = plt.gca()
    im = ax.imshow(img, vmin=data_range[0], vmax=data_range[1], cmap=cmap)
    # plt.title('fake_noisy')
    if cbar:
        # plt.colorbar()
        # # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # # divider = make_axes_locatable(ax)
        # # cax = divider.append_axes("right", size="5%", pad=0.05)
        # # plt.colorbar(im, cax=cax)
        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04, shrink=1.0)


    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
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

    import torchsummary as summary
    summary.summary(model.networks['UNet'], (1, 32, 32))
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_num, trainable_num
    total_num, trainable_num = get_parameter_number(model.networks['UNet'])
    print("trainable_num/total_num: %.2fM/%.2fM" % (trainable_num / 1e6, total_num / 1e6))

    count = 0

    # load info
    data_dir = '/home/shendi_mcj/datasets/seismic/fielddata/'
    # data_dir = 'E:\博士期间资料\田亚军\田博给的数据\\2021.6.07 新疆数据\\'
    im = '00-L120.sgy'
    from seis_utils.readsegy import readsegy
    original = readsegy(data_dir+'00-L120.sgy')#[50:50+64,500-64:500]#[50:50+64,500-64:500]#[0:480,0:480][300:780,0:480][76:876,0:480] [50:50+128,500-128:500]
    x_max=max(abs(original.max()),abs(original.min()))
    GT = readsegy(data_dir+'00-L120-Y.sgy')#[50:50+64,500-64:500]
    noise = readsegy(data_dir+'00-L120-N.sgy')#[50:50+64,500-64:500]
    noisy = original#[50:50+64,500-64:500]
    noisy = noisy / x_max
    GT = GT / x_max
    noise = noise / x_max
    # plot_cmap(noisy, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    # plot_cmap(GT, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
    data={}
    data['L'] = torch.from_numpy(noisy).view(1, -1, noisy.shape[0], noisy.shape[1]).type(torch.FloatTensor)
    data['H'] = torch.from_numpy(GT).view(1, -1, GT.shape[0], GT.shape[1]).type(torch.FloatTensor)
    # generate alpha
    with torch.no_grad():
        Pad=PadUNet(data['L'], dep_U=6)
        data['L'] = Pad.square_pad_1(width=896) #square_pad() #pad()
        BNN_pad = model.networks['BNN'](data['L'])
    #     LAN_pad = model.networks['LAN'](data['L'])
    #     UNet_pad = model.networks['UNet'](data['L'])
    #
    BNN = Pad.pad_inverse(BNN_pad)
    # LAN = Pad.pad_inverse(LAN_pad)
    # UNet = Pad.pad_inverse(UNet_pad)
    alpha = generate_alpha_1(BNN, lower=0.05, upper=0.27, window_size=7)  #lower=0.05, upper=0.20
    plot_cmap(std(BNN, window_size=7).cpu().squeeze().numpy(), 300, (4, 3), data_range=[0, std(BNN, window_size=7).max()], cmap=plt.cm.jet,
              cbar=True)
    plot_cmap(alpha.cpu().squeeze().numpy(), 300, (4, 3), data_range=[0.45,0.55], cmap=plt.cm.jet, cbar=True) #viridis
    # sio.savemat(('../output/INN_denoise/XJ_std_w7.mat'), {'data': std(BNN, window_size=7).cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/INN_denoise/XJ_beta_l005u020.mat'), {'data': alpha.cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_BNN_LAN_DN_std_w7.mat'), {'data': std(BNN, window_size=7).cpu().squeeze().numpy()[:, :]})
    # sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_BNN_LAN_DN_alpha_l1u5.mat'), {'data': alpha.cpu().squeeze().numpy()[:, :]})


    # plot_cmap((1-alpha).cpu().squeeze().numpy(), 300, (3.3, 3), data_range=[0.0,1.0], cmap='viridis',cbar=True)
    # plot_cmap(BNN.cpu().squeeze().numpy(), 300, (3, 3),data_range=[-1,1], cmap=plt.cm.seismic)
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

    # sio.savemat(('../output/INN_denoise/XJ_4st_x3_dn.mat'), {'data': output[:, :]})
    # sio.savemat(('../output/INN_denoise/XJ_4st_x3_n.mat'), {'data': noisy-output[:, :]})
    # sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_BNN_LAN_DN_29_x3.mat'), {'data': output[:, :]})
    # sio.savemat(('../output/INN_denoise/v2/v2_data/XJ_BNN_LAN_DN_29_x3_n.mat'), {'data': noisy-output[:, :]})


    plot_cmap(output, 300, (3, 3), data_range=[-1, 1], cmap=plt.cm.seismic)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    parser = argparse.ArgumentParser(description="Train the denoiser")
    # two_stage_seis_BSN_invDN four_stage_seis_invDN
    parser.add_argument("--config_file", type=str, default='../option/three_stage_SASS.json')#three_stage_seis_invDN.json
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    opt['mat_path'] = 'SubmitSrgb.mat'
    opt['ensemble'] = False

    # opt['iter']=10000 #
    opt['BNN_iters'] = 2000  #
    opt['LAN_iters'] = 2000  #
    opt['UNet_iters'] = 2000  #
    opt['networks'][0]['args']['blindspot'] = 9
    opt['networks'][1]['args']['receptive_feild'] = 3

    opt['log_dir']= 'logs/XJ_3S_invDN'
    # opt['resume_from']='../train/logs/XJ_4S_invDN_bnn9_1/model_iter_00010000.pth'
    opt['resume_from'] = '../train/logs/XJ/BNN_LAN_DN_t/model_iter_00006000.pth'
    opt['validation_datasets'][0]['args']['dataset_path'] = '/home/shendi_mcj/datasets/seismic/fielddata/train/prep0/SEGY_s256_o128'

    # dir_taishiji ='E:\VIRI\code_backup\SpatiallyAdaptiveSSID\\train'
    # opt['resume_from'] = dir_taishiji+'\logs\XJ_4S_invDN_bnn9_1\model_iter_00080000.pth'
    # # data_dir = '\logs\XJ_BSN_invDN\model_iter_00010000.pth'
    # opt['validation_datasets'][0]['args']['dataset_path'] = 'E:\\博士期间资料\\田亚军\\田博给的数据\\2021.6.07 新疆数据\\xinjiang\\prep\\SEGY_s256_o128'

    recursive_print(opt)
    main(opt)
