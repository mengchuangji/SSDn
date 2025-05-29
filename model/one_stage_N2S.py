from model.base import BaseModel
import os
from utils import util
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt
from model.loss import ReconstructionLoss

import numpy as np

def plot_cmap(img,dpi,figsize,cmap):

    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=0, vmax=1, cmap=cmap)
    # plt.title('fake_noisy')
    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()
def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    return torch.Tensor(A)

def interpolate_mask(tensor, mask, mask_inv):
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv

def trace_mask(shape, patch_size, phase_x, phase_y):
    # A = torch.zeros(shape[-2:])
    # # for i in range(shape[-2]):
    # for j in range(shape[-1]):
    #         # if (i % patch_size == phase_x and j % patch_size == phase_y):
    #         if j % patch_size == phase_y:
    #         # if  j == phase_y:
    #             A[:, j] = 1

    # for j in range(shape[-1]):
    #         if np.random.uniform(size=1) < 0.5:
    #             A[:, j] = 1


    # Create a sample tensor, e.g., a 5x5 tensor
    tensor_size = shape
    A = torch.zeros(tensor_size)
    # Define the probability of setting a column to 0 (e.g., 30%)
    probability = 0.8
    # Determine which columns to set to 0 based on the probability
    columns_to_one = torch.rand(tensor_size[1]) < probability
    # Set the selected columns to 0
    A[:, columns_to_one] = 1
    return torch.Tensor(A)


class Masker():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='zero', infer_single_pass=False, include_mask_as_input=False):
        self.grid_size = width
        self.n_masks = width ** 2

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i):

        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size
        mask = pixel_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
        mask = mask.to(X.device)
        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        if self.mode == 'interpolate':
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        elif self.mode == 'tracewise':
            # phasex = i % self.grid_size
            # phasey = i % X[0, 0].shape[-1]
            phasey = i % self.grid_size
            mask = trace_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
            mask = mask.to(X.device)
            mask_inv = torch.ones(mask.shape).to(X.device) - mask
            masked = X * mask_inv
            # masked = interpolate_mask(X, mask, mask_inv)
        else:
            raise NotImplementedError

        if self.include_mask_as_input:
            net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        return self.n_masks

    def infer_full_image(self, X, model):

        if self.infer_single_pass:
            if self.include_mask_as_input:
                net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(X, 0)
            net_output = model(net_input)

            acc_tensor = torch.zeros(net_output.shape).cpu()

            for i in range(self.n_masks):
                net_input, mask = self.mask(X, i)
                net_output = model(net_input)
                acc_tensor = acc_tensor + (net_output * mask).cpu()

            return acc_tensor


class OneStageModel_N2S(BaseModel):
    def __init__(self, opt):
        super(OneStageModel_N2S, self).__init__(opt)
        self.stage = None
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer_UNet = torch.optim.Adam(self.networks['UNet'].parameters(), lr=opt['lr'])
        self.scheduler_UNet = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_UNet, opt['UNet_iters'])



    def train_step(self, data):
        self.iter += 1
        self.update_stage()

        input = data['L']

        if self.stage == 'UNet':
            self.networks['UNet'].train()
            # masker = Masker(width=4, mode='interpolate')  # mode='interpolate' 'tracewise'
            masker = Masker(width=4, mode='interpolate')  # mode='interpolate' 'tracewise'
            net_input, mask = masker.mask(input, self.iter % (masker.n_masks - 1))
            net_output = self.networks['UNet'](net_input)
            self.loss = self.criteron(net_output * mask, input * mask)
            self.optimizer_UNet.zero_grad()
            self.loss.backward()
            self.optimizer_UNet.step()
            self.scheduler_UNet.step()


    def validation_step(self, data):
        self.update_stage()
        input = data['L']

        if self.stage == 'UNet':
            self.networks['UNet'].eval()
            with torch.no_grad():
                output = self.networks['UNet'](input)
        return output

    def save_net(self):
        if self.stage == 'UNet':
            net = self.networks['UNet']

        if isinstance(net, DataParallel):
            net = net.module
        torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], 'net_iter_%08d.pth' % self.iter))

    def save_model(self):
        if self.stage == 'UNet':
            save_dict = {'iter': self.iter,
                         'optimizer_UNet': self.optimizer_UNet.state_dict(),
                         'scheduler_UNet': self.scheduler_UNet.state_dict(),
                         'UNet': self.networks['UNet'].state_dict()}

        torch.save(save_dict, os.path.join(self.opt['log_dir'], 'model_iter_%08d.pth' % self.iter))

    def load_model(self, path):
        load_dict = torch.load(path)
        self.iter = load_dict['iter']
        self.update_stage()
        if self.stage == 'UNet':
            self.optimizer_UNet.load_state_dict(load_dict['optimizer_UNet'])
            self.scheduler_UNet.load_state_dict(load_dict['scheduler_UNet'])
            self.networks['UNet'].load_state_dict(load_dict['UNet'])

        else:
            raise NotImplementedError

    def update_stage(self):
        self.stage = 'UNet'

