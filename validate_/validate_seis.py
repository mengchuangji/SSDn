import sys
sys.path.append('..')
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.utils.data import DataLoader
from utils.option import parse, recursive_print
from utils.build import build


def validate_seis(model, seis_loader, data_range):
    psnrs, ssims, snrs, count = 0, 0, 0, 0
    for data in seis_loader:
        output = model.validation_step(data)
        # output = torch.floor(output + 0.5)
        # output = torch.clamp(output, 0, 255)

        output = torch.clamp(output, -1, 1)
        # output = torch.clamp(output, 0, 1)
        output = output.cpu().squeeze().numpy()
        gt = data['H'].squeeze().numpy()
        psnr = peak_signal_noise_ratio(output, gt, data_range=data_range)
        ssim = structural_similarity(output, gt, data_range=data_range)
        psnrs += psnr
        ssims += ssim
        snr=compare_SNR(gt,output)
        snrs += snr
        count += 1
    return psnrs / count, ssims/count, snrs/count


def compare_SNR(real_img,recov_img):
    import numpy as np
    import math
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))

    if noise_var ==0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var,10)
    return s


def main(opt):
    validation_loaders = []
    for validation_dataset_opt in opt['validation_datasets']:
        ValidationDataset = getattr(__import__('dataset'), validation_dataset_opt['type'])
        validation_set = build(ValidationDataset, validation_dataset_opt['args'])
        validation_loader = DataLoader(validation_set, batch_size=1)
        validation_loaders.append(validation_loader)

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])

    for validation_loader in validation_loaders:
        psnr = validate_sidd(model, validation_loader)
        print('%s, psnr: %6.4f' % (validation_loader.dataset.__class__.__name__, psnr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate the denoiser")
    parser.add_argument("--config_file", type=str, default='../option/three_stage.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    recursive_print(opt)

    main(opt)