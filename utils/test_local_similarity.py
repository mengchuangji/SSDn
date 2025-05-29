import numpy as np
import matplotlib.pyplot as plt
y= np.random.normal(0.1, 1, (64,64))
n= np.random.normal(0.4, 3, (64,64))

def correlation_coefficient(matrix1,matrix2):
    M1 = matrix1.mean()
    M2 = matrix2.mean()
    A = np.subtract(matrix1, M1)
    B = np.subtract(matrix2, M2)
    alpha = np.sum(A*B) / (np.sqrt((np.sum(A*A))*np.sum(B*B)))
    return alpha

def avg_correlation_coefficient(matrix1,matrix2,window_size):
    H = matrix1.shape[0]
    W = matrix1.shape[1]
    ind_H = list(range(0, H - window_size + 1, window_size))
    if ind_H[-1] < H - window_size:
        ind_H.append(H - window_size)
    ind_W = list(range(0, W - window_size + 1, window_size))
    if ind_W[-1] < W - window_size:
        ind_W.append(W - window_size)
    cc=0
    cc_map=np.zeros(matrix1.shape)
    for start_H in ind_H:
        for start_W in ind_W:
            patch1 = matrix1[start_H:start_H + window_size, start_W:start_W + window_size, ]
            patch2 = matrix2[start_H:start_H + window_size, start_W:start_W + window_size, ]
            cc_temp=correlation_coefficient(patch1, patch2)
            cc= cc+cc_temp
            cc_map[start_H:start_H + window_size, start_W:start_W + window_size, ]=cc_temp
    avg_cc=cc/(len(ind_H)*len(ind_W))
    return avg_cc,cc_map

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

# cc=correlation_coefficient(y,n)
# print("correlation_coefficient:",cc)
#
# local_cc,cc_map=avg_correlation_coefficient(y,n,window_size=4)
# print("local correlation_coefficient:",local_cc)
# plot_cmap(cc_map, 300, (4, 3), data_range=[0, 1], cmap=plt.cm.jet,cbar=True)