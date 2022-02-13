import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib as mpl
from matplotlib import pyplot as plt
from MALARIA2 import MALARIA
import localizerVgg
from scipy.ndimage.morphology import grey_dilation
from PIL import Image
from os import path
import utils


def generate_weight_map(heatmap):
    k_size = 3
    weight_map = np.zeros_like(heatmap)
    dilate = np.zeros_like(heatmap)
    for j in range(heatmap.shape[0]):
        dilate[j] = grey_dilation(heatmap[j], size=(k_size, k_size))
        valid_ind = np.where(dilate[j] > 0.2)
        weight_map[j, valid_ind[0], valid_ind[1]] = 1
        # plot_map(heatmap[j], dilate[j], weight_map[j])
    plot_map(heatmap, dilate, weight_map)
    return weight_map


def plot_map(gam, dilated_gam, wmap):
    num_classes = gam.shape[0]
    plt.figure(1)
    for i in range(num_classes):
        plt.subplot(num_classes, 3, 3*i+1)
        plt.imshow(gam[i], cmap='jet')
        plt.title(f'GAM[{i}]')
        plt.axis('off')
        plt.subplot(num_classes, 3, 3*i+2)
        plt.imshow(dilated_gam[i], cmap='jet')
        plt.title(rf'$GAM^d[{i}]$')
        plt.axis('off')
        plt.subplot(num_classes, 3, 3*i+3)
        plt.imshow(wmap[i], cmap='gray')
        plt.title(f'M[{i}]')
        plt.axis('off')
    plt.savefig('figures/paper/weight_map')
    plt.show()



if __name__ == '__main__':
    NUM_CLASSES = 2
    img_id = 2

    dataset = MALARIA('', 'train', train=True, num_classes=NUM_CLASSES)
    with torch.no_grad():
        # Obtain image
        data, GAM, num_cells = dataset[img_id]
        img = utils.normalize_image(data)
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.savefig('figures/paper/weight_map_img')
        weight_map = generate_weight_map(GAM)

    print('Done')

