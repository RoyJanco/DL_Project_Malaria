import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import visdom
import matplotlib as mpl
from matplotlib import pyplot as plt
from MALARIA import MALARIA
import localizerVgg
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import numpy as np
import cv2 as cv
mpl.use('Qt5Agg')


# path = 'MALARIA/Images/4da8f6c5-5f85-4280-92d0-1ba008e8c404.jpg'
# img = cv.imread(path,0)
# equ = cv.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# plt.figure(1)
# plt.imshow(img, cmap='gray')
# plt.figure(2)
# plt.imshow(equ, cmap='gray')
# plt.show()

def detect_peaks_multi_channels(image): # ***** BUG with maximum filter *****
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, size=3) == image
    max_filter = maximum_filter(image, size=3)  # We don't need it
    plot_peak_maps(max_filter, local_max, image)
    background = (image == 0)
    eroded_background = np.zeros(shape=background.shape, dtype=bool)
    for i in range(image.shape[0]):
        eroded_background[i] = binary_erosion(background[i], structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    # detected_peaks = (local_max ^ eroded_background).astype(np.int) - eroded_background.astype(np.int)
    plot_peak_maps(max_filter, detected_peaks, image)

    return detected_peaks # Original
    # return local_max, detected_peaks


def detect_peaks(image):
    detected_peaks = np.zeros_like(image)
    local_max = np.zeros_like(image, dtype=bool)
    max_filter = np.zeros_like(image)
    neighborhood = generate_binary_structure(2, 2)
    for i in range(image.shape[0]):
        max_filter[i] = maximum_filter(image[i], footprint=neighborhood)
        local_max[i] = maximum_filter(image[i], footprint=neighborhood) == image[i]
        background = (image[i] == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        detected_peaks[i] = local_max[i] ^ eroded_background
    # plot_peak_maps(max_filter, local_max, image)
    return detected_peaks


def plot_maps(data, heatmap_gt, heatmap_pred, peak_map, peak_map_gt):
    image = data.cpu().numpy().squeeze().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
    plt.figure(1)
    plt.imshow(image)
    plt.title('Image')

    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.imshow(peak_map_gt)
    plt.title('peak_map GT')
    plt.subplot(2, 2, 2)
    plt.imshow(peak_map)
    plt.title('peak_map')

    plt.subplot(2, 2, 3)
    plt.imshow(heatmap_gt)
    plt.title('GT heatmap')
    plt.subplot(2, 2, 4)
    plt.imshow(heatmap_pred)
    plt.title('Predicted heatmap')

    plt.show()


def plot_peak_maps(max_filter, peak_map, image):
    plt.figure(1)
    for i in range(3):
        plt.subplot(3, 3, 3*i+1)
        plt.imshow(image[i])
        plt.title('MAP')
        plt.subplot(3, 3, 3*i+2)
        plt.imshow(max_filter[i])
        plt.title('max filter')
        plt.subplot(3, 3, 3*i+3)
        plt.imshow(peak_map[i])
        plt.title('peak map')
    plt.show()


def plot_heatmaps(heatmap_gt, heatmap_pred, peak_maps):
    plt.figure()
    num_plots = heatmap_gt.shape[0]
    for i in range(num_plots):
        plt.subplot(3, num_plots, i+1)
        plt.imshow(heatmap_gt[i])
        plt.title(f'GT - class [{i}]')
        plt.subplot(3, num_plots, i+num_plots+1)
        plt.imshow(heatmap_pred[i])
        plt.title(f'Pred - class [{i}]')
        plt.subplot(3, num_plots, i + 2*num_plots + 1)
        plt.imshow(peak_maps[i])
        plt.title(f'Peak maps - class [{i}]')
    plt.show()

cm_jet = mpl.cm.get_cmap('jet')

model = localizerVgg.localizervgg16(pretrained=True)
# model.cuda()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

state_dict = torch.load('L2_WM_BN_AG_B09995_e10.pt', map_location=torch.device(device))
# state_dict = torch.load('model_l2_b0_e10.pt', map_location=torch.device(device))
# print(state_dict.keys())
model.load_state_dict(state_dict)
model = model.to(device)

train_dataset = MALARIA('', 'train', train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

data, GAM, num_cells = next(iter(train_loader))
data, GAM, num_cells = data.to(device, dtype=torch.float),  GAM.to(device), num_cells.to(device)

thr = 0.5
with torch.no_grad():
    MAP = model(data)
    # Create cMap for every class
    cMap = MAP[0,].data.cpu().numpy()
    cMap.min(axis=(1,2))
    cMap_min = cMap.min(axis=(1,2)).reshape((cMap.shape[0], 1, 1))
    cMap_max = cMap.max(axis=(1,2)).reshape((cMap.shape[0], 1, 1))
    cMap = (cMap - cMap_min) / (cMap_max - cMap_min)
    # cMap_copy = cMap.copy()
    # thr_avg = np.mean(cMap, axis=(1,2)).reshape((cMap.shape[0], 1, 1))
    cMap[cMap < thr] = 0

    cMap_gt = GAM[0].data.cpu().numpy()
    cMap_gt[cMap_gt < thr] = 0
    # Detect peaks in the predicted heat map BUG!!!!!!!!!!!!!!!!!!!!!!!!
    # peakMAPs = detect_peaks_multi_channels(cMap)
    # peakMAPs_gt = detect_peaks_multi_channels(cMap_gt)

    peakMAPs = detect_peaks(cMap)
    peakMAPs_gt = detect_peaks(cMap_gt)

    # Combina
    # local_max, peakMAPs = detect_peaks_multi_channels(cMap)
    # local_max_gt, peakMAPs_gt = detect_peaks_multi_channels(cMap_gt)
    # print(np.all(peakMAPs_gt == local_max))
    # plt.figure(9)
    # plt.subplot(1, 2, 1)
    # plt.imshow(peakMAPs_gt[0])
    # plt.title('peak_map GT')
    # plt.subplot(1, 2, 2)
    # plt.imshow(local_max[0])
    # plt.title('local_max combina')

    # plot_maps(data, GAM[0,0].cpu().detach().numpy(), MAP[0,0].cpu().detach().numpy(), peakMAPs[0], peakMAPs_gt[0])
    image = data.cpu().numpy().squeeze().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
    plt.figure(1)
    plt.imshow(image)
    plt.title('Image')
    plot_heatmaps(GAM[0].cpu().detach().numpy(), MAP[0].cpu().detach().numpy(), peakMAPs)
    # plot_heatmaps(GAM[0].cpu().detach().numpy(), cMap_copy, peakMAPs)


    pred_num_cells = np.sum(peakMAPs, axis=(1, 2))
    fark = abs(pred_num_cells - num_cells.cpu().detach().numpy())

    print(f'Predicted number of RBC: {pred_num_cells[0]}. GT: {num_cells[0,0]}')
    print(f'AE {fark}')

print('Done')