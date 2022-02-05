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
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from PIL import Image
import utils

if __name__ == '__main__':
    # Inputs
    NUM_CLASSES = 2
    img_id = 15
    # img_id 1 is interesting
    # model_path = 'saved models/c-2_l2_b-0.9999_wm_e-1.pt'
    model_path = 'saved models/c-2_l2_b-0.0_e-1.pt'
    # model_path = 'saved models/c-2_AW_b-0.9999_wm_e-1.pt'

    cm_jet = mpl.cm.get_cmap('jet')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load pretrained model
    dataset = MALARIA('', 'train', train=True, num_classes=NUM_CLASSES)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1100, 108], generator=torch.Generator().manual_seed(42)) # [1100, 108]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = localizerVgg.localizervgg16(num_classes=test_dataset.dataset.get_number_classes(), pretrained=True)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # thr = 0.5
    # Set threshold as vector
    thr = [0.5, 0.5]
    thr = np.array(thr).reshape(NUM_CLASSES, 1, 1)
    with torch.no_grad():
        # Obtain image
        data, GAM, num_cells = test_dataset.dataset[img_id]
        data = data.unsqueeze(dim=0).to(device, dtype=torch.float)
        GAM = torch.Tensor(GAM).unsqueeze(dim=0).to(device, dtype=torch.float)
        MAP = model(data)
        # Create cMap for multi class
        cMap = MAP.data.cpu().numpy()
        cMap_min = cMap.min(axis=(2, 3)).reshape((cMap.shape[0], cMap.shape[1], 1, 1))
        cMap_max = cMap.max(axis=(2, 3)).reshape((cMap.shape[0], cMap.shape[1], 1, 1))
        cMap = (cMap - cMap_min) / (cMap_max - cMap_min)
        cMap[cMap < thr] = 0
        # Detect peaks in the predicted heat map:
        peakMAPs = utils.detect_peaks_multi_channels_batch(cMap)

        # MAP & GAM shape is [B, C, H, W].

        pred_num_cells = np.sum(peakMAPs, axis=(2, 3))
        pred_num_cells_batch = np.sum(pred_num_cells, axis=0)

        MAP_norm = utils.normalize_map(MAP)
        GAM_norm = GAM[0].data.cpu().contiguous().numpy().copy()

        MAP_upsampled = utils.upsample_map(MAP_norm, dsr=8)
        GAM_upsampled = utils.upsample_map(GAM_norm, dsr=8)
        # Normalize image
        image = np.uint8(255*utils.normalize_image(data))
        peak_map = np.uint8(np.array(peakMAPs[0]) * 255)

        plt.figure(1)
        plt.imshow(image)
        num_plots = len(num_cells)
        plt.figure(2)
        for i in range(num_plots):
            plt.subplot(3, num_plots, i + 1)
            plt.imshow(GAM_upsampled[i])
            plt.title(f'GT - class [{i}]')
            plt.subplot(3, num_plots, i + num_plots + 1)
            plt.imshow(MAP_upsampled[i])
            plt.title(f'Pred - class [{i}]')
            plt.subplot(3, num_plots, i + 2*num_plots + 1)
            plt.imshow(peak_map[i])
            plt.title(f'Peak map - class [{i}]')

        print(f'Model counted: {pred_num_cells_batch.astype(int)}.')
        print(f'GT: {num_cells.data.cpu().numpy().astype(int)}.')
        plt.show()
