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

def std(img, window_size=7):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    return img

def generate_alpha(input, lower=0.05, upper=0.2): #lower=1, upper=5
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()

    return ratio

def generate_alpha_1(input, lower=0.05, upper=0.2, window_size=7): #lower=1, upper=5
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input,window_size)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()

    return ratio

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

def trace_mask(shape, p):
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
    probability = p
    # Determine which columns to set to 0 based on the probability
    columns_to_one = torch.rand(tensor_size[1]) < probability
    # Set the selected columns to 0
    A[:, columns_to_one] = 1
    return torch.Tensor(A)

def pixel_mask(shape,p):
    tensor_size = shape
    probability = p
    # 生成包含在[0, 1)范围内的随机值的张量
    random_tensor = torch.rand(tensor_size)
    # 根据概率设置元素为0或1
    binary_tensor = (random_tensor < probability).float()
    return torch.Tensor(random_tensor)


class Masker():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='zero', pobability=0.5, infer_single_pass=False, include_mask_as_input=False):
        self.grid_size = width
        self.n_masks = width ** 2

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input
        self.pobability = pobability

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
            p=self.pobability
            mask = trace_mask(X[0, 0].shape, p)
            mask = mask.to(X.device)
            mask_inv = torch.ones(mask.shape).to(X.device) - mask
            masked = X * mask_inv
            # masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'pixelwise':
            p=0.5
            mask = pixel_mask(X[0, 0].shape,p)
            mask = mask.to(X.device)
            mask_inv = torch.ones(mask.shape).to(X.device) - mask
            masked = X * mask_inv
            masked = interpolate_mask(X, mask, mask_inv)
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


class ThreeStageModel_v2(BaseModel):
    def __init__(self, opt):
        super(ThreeStageModel_v2, self).__init__(opt)
        self.stage = None
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer_N2SUNet = torch.optim.Adam(self.networks['N2SUNet'].parameters(), lr=opt['lr'])
        self.scheduler_N2SUNet = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_N2SUNet, opt['N2SUNet_iters'])
        self.optimizer_LAN = torch.optim.Adam(self.networks['LAN'].parameters(), lr=opt['lr'])
        self.scheduler_LAN = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_LAN, opt['LAN_iters'])
        self.optimizer_UNet = torch.optim.Adam(self.networks['UNet'].parameters(), lr=opt['lr'])
        self.scheduler_UNet = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_UNet, opt['UNet_iters'])
        self.pixelwise_width= opt['width']
        self.mode= opt['mode']
        self.pobability = opt['pobability']
        self.std_width= opt['std_width']
        self.lower = opt['beta_lower_upper'][0]['lower']
        self.upper = opt['beta_lower_upper'][0]['upper']

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims))

    def train_step(self, data):
        self.iter += 1
        self.update_stage()

        input = data['L']
        if self.stage == 'N2SUNet':
            masker = Masker(width=self.pixelwise_width, mode=self.mode,pobability=self.pobability)  # mode='interpolate' 'tracewise' 'zero'
            net_input, mask = masker.mask(input, self.iter % (masker.n_masks - 1))
            net_output = self.networks['N2SUNet'](net_input)
            self.loss = self.criteron(net_output * mask, input * mask)
            self.optimizer_N2SUNet.zero_grad()
            self.loss.backward()
            self.optimizer_N2SUNet.step()
            self.scheduler_N2SUNet.step()

        elif self.stage == 'LAN':
            self.networks['N2SUNet'].eval()
            self.networks['LAN'].train()
            with torch.no_grad():
                N2SUNet = self.networks['N2SUNet'](input)
            LAN = self.networks['LAN'](input)

            # alpha = generate_alpha(N2SUNet)
            # self.loss = self.criteron(N2SUNet.detach() * (1 - alpha), LAN * (1 - alpha))
            self.loss = self.criteron(N2SUNet, LAN)
            self.optimizer_LAN.zero_grad()
            self.loss.backward()
            self.optimizer_LAN.step()
            self.scheduler_LAN.step()

        elif self.stage == 'UNet':
            self.networks['N2SUNet'].eval()
            self.networks['LAN'].eval()
            self.networks['UNet'].train()
            with torch.no_grad():
                N2SUNet = self.networks['N2SUNet'](input)
                LAN = self.networks['LAN'](input)
            UNet = self.networks['UNet'](input)
            # alpha = generate_alpha(N2SUNet)

            alpha = generate_alpha_1(N2SUNet, lower=self.lower, upper=self.upper,
                                     window_size=self.std_width)
            self.loss = self.criteron(N2SUNet * (1 - alpha), UNet * (1 - alpha)) + self.criteron(LAN * alpha, UNet * alpha)
            self.optimizer_UNet.zero_grad()
            self.loss.backward()
            self.optimizer_UNet.step()
            self.scheduler_UNet.step()

    def validation_step(self, data):
        self.update_stage()
        input = data['L']

        if self.stage == 'N2SUNet':
            self.networks['N2SUNet'].eval()
            with torch.no_grad():
                net_outputs = []
                for i in range(16):
                    masker = Masker(width=self.pixelwise_width, mode=self.mode)  # mode='interpolate' 'tracewise' 'zero'
                    net_input, mask = masker.mask(input, i % (masker.n_masks - 1))
                    net_output = self.networks['N2SUNet'](net_input)
                    net_outputs.append(net_output)
                net_input = torch.cat(net_outputs, dim=0)
                output = net_input.mean(dim=0, keepdim=True)

                # masker = Masker(width=self.pixelwise_width, mode=self.mode)  # mode='interpolate' 'tracewise' 'zero'
                # net_input, mask = masker.mask(input, self.iter % (masker.n_masks - 1))
                # output = self.networks['N2SUNet'](net_input)

                # output = self.networks['N2SUNet'](input)
        elif self.stage == 'LAN':
            self.networks['LAN'].eval()
            with torch.no_grad():
                output = self.networks['LAN'](input)
        elif self.stage == 'UNet':
            self.networks['UNet'].eval()
            with torch.no_grad():
                output = self.networks['UNet'](input)
        return output


    def save_net(self):
        if self.stage == 'N2SUNet':
            net = self.networks['N2SUNet']
        elif self.stage == 'LAN':
            net = self.networks['LAN']
        elif self.stage == 'UNet':
            net = self.networks['UNet']

        if isinstance(net, DataParallel):
            net = net.module
        torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], 'net_iter_%08d.pth' % self.iter))

    def save_model(self):
        if self.stage == 'N2SUNet':
            save_dict = {'iter': self.iter,
                         'optimizer_N2SUNet': self.optimizer_N2SUNet.state_dict(),
                         'scheduler_N2SUNet': self.scheduler_N2SUNet.state_dict(),
                         'N2SUNet': self.networks['N2SUNet'].state_dict()}
        elif self.stage == 'LAN':
            save_dict = {'iter': self.iter,
                         'optimizer_LAN': self.optimizer_LAN.state_dict(),
                         'scheduler_LAN': self.scheduler_LAN.state_dict(),
                         'N2SUNet': self.networks['N2SUNet'].state_dict(),
                         'LAN': self.networks['LAN'].state_dict()}
        elif self.stage == 'UNet':
            save_dict = {'iter': self.iter,
                         'optimizer_UNet': self.optimizer_UNet.state_dict(),
                         'scheduler_UNet': self.scheduler_UNet.state_dict(),
                         'N2SUNet': self.networks['N2SUNet'].state_dict(),
                         'LAN': self.networks['LAN'].state_dict(),
                         'UNet': self.networks['UNet'].state_dict()}
        torch.save(save_dict, os.path.join(self.opt['log_dir'], 'model_iter_%08d.pth' % self.iter))

    def load_model(self, path):
        load_dict = torch.load(path)
        self.iter = load_dict['iter']
        self.update_stage()
        if self.stage == 'N2SUNet':
            self.optimizer_N2SUNet.load_state_dict(load_dict['optimizer_N2SUNet'])
            self.scheduler_N2SUNet.load_state_dict(load_dict['scheduler_N2SUNet'])
            self.networks['N2SUNet'].load_state_dict(load_dict['N2SUNet'])
        elif self.stage == 'LAN':
            self.optimizer_LAN.load_state_dict(load_dict['optimizer_LAN'])
            self.scheduler_LAN.load_state_dict(load_dict['scheduler_LAN'])
            self.networks['N2SUNet'].load_state_dict(load_dict['N2SUNet'])
            self.networks['LAN'].load_state_dict(load_dict['LAN'])
        elif self.stage == 'UNet':
            self.optimizer_UNet.load_state_dict(load_dict['optimizer_UNet'])
            self.scheduler_UNet.load_state_dict(load_dict['scheduler_UNet'])
            self.networks['N2SUNet'].load_state_dict(load_dict['N2SUNet'])
            self.networks['LAN'].load_state_dict(load_dict['LAN'])
            self.networks['UNet'].load_state_dict(load_dict['UNet'])
        else:
            raise NotImplementedError

    def update_stage(self):
        if self.iter <= self.opt['N2SUNet_iters']:
            self.stage = 'N2SUNet'
        elif self.iter <= self.opt['N2SUNet_iters'] + self.opt['LAN_iters']:
            self.stage = 'LAN'
        else:
            self.stage = 'UNet'
