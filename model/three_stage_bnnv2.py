from model.base import BaseModel
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt

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

def generate_alpha_1(input, lower=0.05, upper=0.2, window_size=7): #lower=1, upper=5
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input,window_size)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()
    return ratio

class ThreeStageModel_bnnv2(BaseModel):
    def __init__(self, opt):
        super(ThreeStageModel_bnnv2, self).__init__(opt)
        self.stage = None
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer_BNN_v2 = torch.optim.Adam(self.networks['BNN_v2'].parameters(), lr=opt['lr'])
        self.scheduler_BNN_v2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_BNN_v2, opt['BNN_v2_iters'])
        self.optimizer_LAN = torch.optim.Adam(self.networks['LAN'].parameters(), lr=opt['lr'])
        self.scheduler_LAN = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_LAN, opt['LAN_iters'])
        self.optimizer_UNet = torch.optim.Adam(self.networks['UNet'].parameters(), lr=opt['lr'])
        self.scheduler_UNet = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_UNet, opt['UNet_iters'])
        self.std_width= opt['std_width']
        self.lower = opt['beta_lower_upper'][0]['lower']
        self.upper = opt['beta_lower_upper'][0]['upper']
    def train_step(self, data):
        self.iter += 1
        self.update_stage()

        input = data['L']

        if self.stage == 'BNN_v2':
            self.networks['BNN_v2'].train()
            BNN_v2 = self.networks['BNN_v2'](input)
            self.loss = self.criteron(BNN_v2, input)
            self.optimizer_BNN_v2.zero_grad()
            self.loss.backward()
            self.optimizer_BNN_v2.step()
            self.scheduler_BNN_v2.step()

        elif self.stage == 'LAN':
            self.networks['BNN_v2'].eval()
            self.networks['LAN'].train()
            with torch.no_grad():
                BNN_v2 = self.networks['BNN_v2'](input)
            LAN = self.networks['LAN'](input)

            # alpha = generate_alpha(BNN_v2)
            # self.loss = self.criteron(BNN_v2.detach() * (1 - alpha), LAN * (1 - alpha))
            self.loss = self.criteron(BNN_v2, LAN)
            self.optimizer_LAN.zero_grad()
            self.loss.backward()
            self.optimizer_LAN.step()
            self.scheduler_LAN.step()

        elif self.stage == 'UNet':
            self.networks['BNN_v2'].eval()
            self.networks['LAN'].eval()
            self.networks['UNet'].train()
            with torch.no_grad():
                BNN_v2 = self.networks['BNN_v2'](input)
                LAN = self.networks['LAN'](input)
            UNet = self.networks['UNet'](input)

            # alpha = generate_alpha(BNN_v2)
            alpha = generate_alpha_1(BNN_v2, lower=self.lower, upper=self.upper,
                                     window_size=self.std_width)

            # plot_cmap(alpha.cpu().squeeze().numpy(), 300, (5, 10), cmap=plt.cm.seismic)
            self.loss = self.criteron(BNN_v2 * (1 - alpha), UNet * (1 - alpha)) + self.criteron(LAN * alpha, UNet * alpha)
            self.optimizer_UNet.zero_grad()
            self.loss.backward()
            self.optimizer_UNet.step()
            self.scheduler_UNet.step()


    def validation_step(self, data):
        self.update_stage()
        input = data['L']

        if self.stage == 'BNN_v2':
            self.networks['BNN_v2'].eval()
            with torch.no_grad():
                output = self.networks['BNN_v2'](input)
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
        if self.stage == 'BNN_v2':
            net = self.networks['BNN_v2']
        elif self.stage == 'LAN':
            net = self.networks['LAN']
        elif self.stage == 'UNet':
            net = self.networks['UNet']

        if isinstance(net, DataParallel):
            net = net.module
        torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], 'net_iter_%08d.pth' % self.iter))

    def save_model(self):
        if self.stage == 'BNN_v2':
            save_dict = {'iter': self.iter,
                         'optimizer_BNN_v2': self.optimizer_BNN_v2.state_dict(),
                         'scheduler_BNN_v2': self.scheduler_BNN_v2.state_dict(),
                         'BNN_v2': self.networks['BNN_v2'].state_dict()}
        elif self.stage == 'LAN':
            save_dict = {'iter': self.iter,
                         'optimizer_LAN': self.optimizer_LAN.state_dict(),
                         'scheduler_LAN': self.scheduler_LAN.state_dict(),
                         'BNN_v2': self.networks['BNN_v2'].state_dict(),
                         'LAN': self.networks['LAN'].state_dict()}
        elif self.stage == 'UNet':
            save_dict = {'iter': self.iter,
                         'optimizer_UNet': self.optimizer_UNet.state_dict(),
                         'scheduler_UNet': self.scheduler_UNet.state_dict(),
                         'BNN_v2': self.networks['BNN_v2'].state_dict(),
                         'LAN': self.networks['LAN'].state_dict(),
                         'UNet': self.networks['UNet'].state_dict()}
        torch.save(save_dict, os.path.join(self.opt['log_dir'], 'model_iter_%08d.pth' % self.iter))

    def load_model(self, path):
        load_dict = torch.load(path)
        self.iter = load_dict['iter']
        self.update_stage()
        if self.stage == 'BNN_v2':
            self.optimizer_BNN_v2.load_state_dict(load_dict['optimizer_BNN_v2'])
            self.scheduler_BNN_v2.load_state_dict(load_dict['scheduler_BNN_v2'])
            self.networks['BNN_v2'].load_state_dict(load_dict['BNN_v2'])
        elif self.stage == 'LAN':
            self.optimizer_LAN.load_state_dict(load_dict['optimizer_LAN'])
            self.scheduler_LAN.load_state_dict(load_dict['scheduler_LAN'])
            self.networks['BNN_v2'].load_state_dict(load_dict['BNN_v2'])
            self.networks['LAN'].load_state_dict(load_dict['LAN'])
        elif self.stage == 'UNet':
            self.optimizer_UNet.load_state_dict(load_dict['optimizer_UNet'])
            self.scheduler_UNet.load_state_dict(load_dict['scheduler_UNet'])
            self.networks['BNN_v2'].load_state_dict(load_dict['BNN_v2'])
            self.networks['LAN'].load_state_dict(load_dict['LAN'])
            self.networks['UNet'].load_state_dict(load_dict['UNet'])
        else:
            raise NotImplementedError

    def update_stage(self):
        if self.iter <= self.opt['BNN_v2_iters']:
            self.stage = 'BNN_v2'
        elif self.iter <= self.opt['BNN_v2_iters'] + self.opt['LAN_iters']:
            self.stage = 'LAN'
        else:
            self.stage = 'UNet'
