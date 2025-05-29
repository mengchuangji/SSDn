import sys
sys.path.append('..')
import argparse
# from dataset.seis_mat import SeisMatValidationDataset
from dataset.base import BaseTrainDataset
import glob
import numpy as np
import os
import scipy.io as sio
from submit_test.ensemble_wrapper import EnsembleWrapper
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.option import parse, recursive_print
import scipy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from  validate_.validate_seis import compare_SNR
import time
from model.three_stage import generate_alpha
import matplotlib.pyplot as plt
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

class SeisMatValidationDataset(BaseTrainDataset):
    # def __init__(self):
    #     super(SeisMatValidationDataset, self).__init__(mat_path)
    def __init__(self, path, patch_size, pin_memory):
        super(SeisMatValidationDataset, self).__init__(path, patch_size, pin_memory)
        self._get_img_paths(path)
        self.n_data = len(self.img_paths)

    def __getitem__(self, index):
        index = index % len(self.img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
            img_H = self.imgs[index]['H']
        else:
            img_path = self.img_paths[index]
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])

        img_L, img_H = self.crop(img_L, img_H)

        img_L, img_H = np.float32(np.ascontiguousarray(img_L)), np.float32(np.ascontiguousarray(img_H))
        return {'L': img_L, 'H': img_H}


    def _get_img_paths(self, path):
        self.img_paths = []
        L_pattern = os.path.join(path, 'RN/*.mat')
        # L_paths = sorted(glob.glob(L_pattern))
        L_paths = glob.glob(L_pattern)
        i=0
        for L_path in L_paths:
            self.img_paths.append({'L': L_path, 'H': L_path.replace('RN', 'CL')})
            # self.img_paths.append({'L': L_path.replace('f35_s256_o128\\RN\\0_', 'f35_s256_o128_orfx\\CL\\'+L_path.split("_")[-1].split(".")[0]+'_'),
            #                    'H': L_path.replace('RN', 'CL')})
        # self.img_paths = self.img_paths[int(39 / 40 * len(self.img_paths)):]#mcj
        self.img_paths = self.img_paths[int(39 / 40 * len(self.img_paths)):]  # mcj

    def _open_images(self):
        self.imgs = []
        for img_path in self.img_paths:
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])
            self.imgs.append({'L': img_L, 'H': img_H})

    def _open_image(self, path):
        img = np.array(scipy.io.loadmat(path)['data'])
        return img

    def __len__(self):
        return self.n_data

def main():
    test_path= 'D:\datasets\seismic\marmousi\\f35_s256_o128\\'
    test_set = SeisMatValidationDataset(path=test_path,patch_size=256, pin_memory= True)
    test_loader = DataLoader(test_set, batch_size=1)

    count = 0
    start_time = time.time()
    psnrs, ssims, snrs, count = 0, 0, 0, 0
    for data in tqdm(test_loader):
        output = data['L'].squeeze().numpy()
        gt = data['H'].squeeze().numpy()
        psnr = peak_signal_noise_ratio(output, gt, data_range=2)
        ssim = structural_similarity(output, gt, data_range=2)
        psnrs += psnr
        ssims += ssim
        snr = compare_SNR(gt, output)
        snrs += snr
        count += 1
    print('psnrs / count:',psnrs / count,'ssims / count:',ssims / count,'snrs / count:',snrs / count)





if __name__ == '__main__':

    main()
