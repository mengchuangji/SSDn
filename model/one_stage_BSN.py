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

def generate_alpha(input, lower=1, upper=5):
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()

    return ratio



class OneStageModel_BSN(BaseModel):
    def __init__(self, opt):
        super(OneStageModel_BSN, self).__init__(opt)
        self.stage = None
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer_BNN = torch.optim.Adam(self.networks['BNN'].parameters(), lr=opt['lr'])
        self.scheduler_BNN = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_BNN, opt['BNN_iters'])



    def train_step(self, data):
        self.iter += 1
        self.update_stage()

        input = data['L']
        if self.stage == 'BNN':
            self.networks['BNN'].train()
            BNN = self.networks['BNN'](input)
            self.loss = self.criteron(BNN, input)
            self.optimizer_BNN.zero_grad()
            self.loss.backward()
            self.optimizer_BNN.step()
            self.scheduler_BNN.step()

    def validation_step(self, data):
        self.update_stage()
        input = data['L']

        if self.stage == 'BNN':
            self.networks['BNN'].eval()
            with torch.no_grad():
                output = self.networks['BNN'](input)

        return output


    def save_net(self):
        if self.stage == 'BNN':
            net = self.networks['BNN']

        if isinstance(net, DataParallel):
            net = net.module
        torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], 'net_iter_%08d.pth' % self.iter))

    def save_model(self):
        if self.stage == 'BNN':
            save_dict = {'iter': self.iter,
                         'optimizer_BNN': self.optimizer_BNN.state_dict(),
                         'scheduler_BNN': self.scheduler_BNN.state_dict(),
                         'BNN': self.networks['BNN'].state_dict()}
        torch.save(save_dict, os.path.join(self.opt['log_dir'], 'model_iter_%08d.pth' % self.iter))

    def load_model(self, path):
        load_dict = torch.load(path)
        self.iter = load_dict['iter']
        self.update_stage()
        if self.stage == 'BNN':
            self.optimizer_BNN.load_state_dict(load_dict['optimizer_BNN'])
            self.scheduler_BNN.load_state_dict(load_dict['scheduler_BNN'])
            self.networks['BNN'].load_state_dict(load_dict['BNN'])
        else:
            raise NotImplementedError

    def update_stage(self):
        self.stage = 'BNN'
