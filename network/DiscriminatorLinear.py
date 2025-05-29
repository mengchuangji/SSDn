import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as utils
import torch.nn.functional as F

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

class DiscriminatorLinear(nn.Module):
    def __init__(self, in_chn, ndf=64, slope=0.2):
        '''
        ndf: number of filters
        '''
        super(DiscriminatorLinear, self).__init__()
        self.ndf = ndf
        # input is N x C x 32 x 32
        main_module = [conv_down(in_chn, ndf, bias=False),
                       nn.LeakyReLU(slope, inplace=True)]
        # state size: N x ndf x 16 x 16
        main_module.append(conv_down(ndf, ndf*2, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*2) x 8 x 8
        main_module.append(conv_down(ndf*2, ndf*4, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*4) x 4 x 4
        main_module.append(conv_down(ndf*4, ndf*8, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*8) x 1 x 1
        self.main = nn.Sequential(*main_module)
        self.output = nn.Linear(ndf*8, 1)

        self._initialize()

    def forward(self, x):
        feature = self.main(x)
        feature = feature.view(-1, self.ndf*8)
        out = self.output(feature)
        return out.view(-1)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight.data, 0., 0.02)
                if not m.bias is None:
                    init.constant_(m.bias, 0)