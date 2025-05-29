import cv2
import numpy as np
import torch
import torch.nn.functional as F



def sample_generator(netG, x):
    z = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)
    x1 = torch.cat([x, z], dim=1)
    out = netG(x1)

    return out + x


def mean_match(x, y, fake_y, kernel, chn=3):
    p = kernel.shape[2]
    # estimate the real distribution
    err_real = y - x
    mu_real = gaussblur(err_real, kernel, p, chn)
    err_fake = fake_y - x
    mu_fake = gaussblur(err_fake, kernel, p, chn)
    loss = F.l1_loss(mu_real, mu_fake, reduction='mean')

    return loss


def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p - 1) / 2), ] * 4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)
    return y
def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)

def gradient_penalty(real_data, generated_data, netP, lambda_gp):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(real_data.device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated.requires_grad = True

    # Calculate probability of interpolated examples
    prob_interpolated = netP(interpolated)

    # Calculate gradients of probabilities with respect to examples
    grad_outputs = torch.ones(prob_interpolated.size(), device=real_data.device, dtype=torch.float32)
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return lambda_gp * ((gradients_norm - 1) ** 2).mean()

def estimate_sigma_gauss(img_noisy, img_gt):
    win_size = 7
    err2 = (img_noisy - img_gt) ** 2
    kernel = get_gausskernel(win_size, chn=1).to(img_gt.device)
    sigma = gaussblur(err2, kernel, win_size, chn=1)
    sigma.clamp_(min=1e-10)

    return sigma
def kl_gauss_zero_center(sigma_fake, sigma_real):
    '''
    Input:
        sigma_fake: 1 x C x H x W, torch array
        sigma_real: 1 x C x H x W, torch array
    '''
    div_sigma = torch.div(sigma_fake, sigma_real)
    div_sigma.clamp_(min=0.1, max=10)
    log_sigma = torch.log(1 / div_sigma)
    distance = 0.5 * torch.mean(log_sigma + div_sigma - 1.)
    return distance
class PadUNet:
    '''
    im: N x C x H x W torch tensor
    dep_U: depth of UNet
    '''
    def __init__(self, im, dep_U, mode='reflect'):
        self.im_old = im
        self.dep_U = dep_U
        self.mode = mode
        self.H_old = im.shape[2]
        self.W_old = im.shape[3]

    def pad(self):
        # lenU = 2 ** self.dep_U
        # padH = 0 if ((self.H_old % lenU) == 0) else ((self.H_old//lenU+1)* lenU-self.H_old)
        # padW = 0 if ((self.W_old % lenU) == 0) else ((self.W_old//lenU+1)* lenU-self.W_old)
        # padding = (0, padW, 0, padH)
        # import torch.nn.functional as F
        # out = F.pad(self.im_old, pad=padding, mode=self.mode)
        # return out

        lenU = 2 ** (self.dep_U-1)
        padH = 0 if ((self.H_old % lenU) == 0) else (lenU - (self.H_old % lenU))
        padW = 0 if ((self.W_old % lenU) == 0) else (lenU - (self.W_old % lenU))
        padding = (0, padW, 0, padH)
        import torch.nn.functional as F
        out = F.pad(self.im_old, pad=padding, mode=self.mode)
        return out

    def square_pad(self):
        lenU = 2 ** self.dep_U
        padH = 0 if ((self.H_old % lenU) == 0) else ((self.H_old // lenU + 1) * lenU - self.H_old)
        padW = 0 if ((self.W_old % lenU) == 0) else ((self.W_old // lenU + 1) * lenU - self.W_old)
        square=0
        if self.H_old+padH <= self.W_old+padW:
            square =self.W_old+padW
        else:
            square = self.H_old+padH
        padH= square-self.H_old
        padW = square - self.W_old
        padding = (0, padW, 0, padH)
        import torch.nn.functional as F
        out = F.pad(self.im_old, pad=padding, mode=self.mode)
        return out

    def square_pad_1(self, width):
        lenU = width
        padH = 0 if ((self.H_old % lenU) == 0) else ((self.H_old // lenU + 1) * lenU - self.H_old)
        padW = 0 if ((self.W_old % lenU) == 0) else ((self.W_old // lenU + 1) * lenU - self.W_old)
        square = 0
        if self.H_old + padH <= self.W_old + padW:
            square = self.W_old + padW
        else:
            square = self.H_old + padH
        padH = square - self.H_old
        padW = square - self.W_old
        padding = (0, padW, 0, padH)
        import torch.nn.functional as F
        out = F.pad(self.im_old, pad=padding, mode=self.mode)
        return out


    def pad_inverse(self, im_new):
        return im_new[:, :, :self.H_old, :self.W_old]




def cal_local_simlarity(matrix1,matrix2,window_size):
    bs = matrix1.shape[0] #batchsize
    H = matrix1.shape[2]
    W = matrix1.shape[3]
    ind_bs=list(range(0, bs, 1))
    ind_H = list(range(0, H - window_size + 1, window_size))
    if ind_H[-1] < H - window_size:
        ind_H.append(H - window_size)
    ind_W = list(range(0, W - window_size + 1, window_size))
    if ind_W[-1] < W - window_size:
        ind_W.append(W - window_size)
    cc=0
    # cc_map=np.zeros(matrix1.shape)
    for ii in ind_bs:
        for start_H in ind_H:
            for start_W in ind_W:
                patch1 = matrix1[ii,:,start_H:start_H + window_size, start_W:start_W + window_size].squeeze()
                patch2 = matrix2[ii,:,start_H:start_H + window_size, start_W:start_W + window_size].squeeze()
                cc_temp=correlation_coefficient(patch1, patch2)
                cc= cc+cc_temp
                # cc_map[start_H:start_H + window_size, start_W:start_W + window_size]=cc_temp
        avg_cc=cc/(len(ind_H)*len(ind_W))
    return avg_cc #,cc_map

def correlation_coefficient(matrix1,matrix2):
    M1 = matrix1.mean()
    M2 = matrix2.mean()
    A = matrix1-M1
    B = matrix1-M2
    alpha = torch.sum(A*B) / (torch.sqrt((torch.sum(A*A))*torch.sum(B*B)))
    return alpha.abs()
