import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib as mpl
from matplotlib import pyplot as plt
from MALARIA import MALARIA
import localizerVgg
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from PIL import Image


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
    return detected_peaks


def detect_peaks_multi_channels_batch(image):  # TODO: Make this function available in only one script
    detected_peaks = np.zeros_like(image)
    neighborhood = generate_binary_structure(2, 2)
    # Loop for each image in batch
    for i in range(image.shape[0]):
        # Loop over classes per image
        for j in range(image.shape[1]):
            local_max = maximum_filter(image[i, j], footprint=neighborhood) == image[i, j]
            background = (image[i, j] == 0)
            eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
            detected_peaks[i, j] = local_max ^ eroded_background
    return detected_peaks


def normalize_image(image):
    image = data.cpu().numpy().squeeze().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
    return image


def normalize_map(map):
    map = map.data.cpu().contiguous().numpy().copy()
    map_min = map.min(axis=(2,3)).reshape((map.shape[0], map.shape[1], 1, 1))
    map_max = map.max(axis=(2,3)).reshape((map.shape[0], map.shape[1], 1, 1))
    map_norm = (map - map_min) / (map_max - map_min)
    # return first image in batch
    return map_norm[0]


def upsample_map(map, dsr):
    upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((map.shape[1]*dsr, map.shape[2]*dsr))])
    map_upsampled = np.zeros((map.shape[0], map.shape[1]*dsr, map.shape[2]*dsr, 4), dtype=np.uint8)
    for i in range(map.shape[0]):
        a = upsampler(torch.Tensor(map[i]))
        map_upsampled[i] = np.uint8(cm_jet(np.array(a)) * 255)
    return map_upsampled


def save_images(map, peak_map, batch_idx):
    # Get first peakmap from the batch
    peak_map = np.uint8(np.array(peak_map[0]) * 255)
    for i in range(map.shape[0]):
        a = map[i]
        ima = Image.fromarray(a)
        peakI = Image.fromarray(peak_map[i]).convert("RGB")
        peakI = peakI.resize((1600, 1200))
        ima.save("results/heatmap-class_" + str(i) + '-batch_' + str(batch_idx) + ".bmp")
        peakI.save("results/peakmap-class_" + str(i) + '-batch_' + str(batch_idx) + ".bmp")


if __name__ == '__main__':
    cm_jet = mpl.cm.get_cmap('jet')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load pretrained model
    model = localizerVgg.localizervgg16(pretrained=True)
    state_dict = torch.load('model_l2_b0_e10.pt', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model = model.to(device)

    data_set = MALARIA('', 'train', train=True)
    # Split dataset to train and test
    split_train, split_test = torch.utils.data.random_split(data_set, [1100, 108], generator=torch.Generator().manual_seed(42))

    data_loader = torch.utils.data.DataLoader(split_test, batch_size=1, shuffle=False, num_workers=0)
    num_classes = len(data_set.classes)
    model.eval()
    thr = 0.5
    # Initialize absolute error and MSE accumulators
    ae, se = torch.zeros(num_classes), torch.zeros(num_classes)
    with torch.no_grad():
        for batch_idx, (data, GAM, num_cells) in enumerate(data_loader, 0):
            data, GAM, num_cells = data.to(device, dtype=torch.float),  GAM.to(device), num_cells.to(device)
            MAP = model(data)
            # Create cMap for multi class
            cMap = MAP.data.cpu().numpy()
            cMap_min = cMap.min(axis=(2,3)).reshape((cMap.shape[0], cMap.shape[1], 1, 1))
            cMap_max = cMap.max(axis=(2,3)).reshape((cMap.shape[0], cMap.shape[1], 1, 1))
            cMap = (cMap - cMap_min) / (cMap_max - cMap_min)
            cMap[cMap < thr] = 0
            # Detect peaks in the predicted heat map:
            peakMAPs = detect_peaks_multi_channels_batch(cMap)

            # MAP & GAM shape is [B, C, H, W]. Reshape to [B, C, H*W]
            # MAP = MAP.view(MAP.shape[0], MAP.shape[1], -1)
            # GAM = GAM.view(GAM.shape[0], GAM.shape[1], -1)

            pred_num_cells = np.sum(peakMAPs, axis=(2, 3))
            pred_num_cells_batch = np.sum(pred_num_cells, axis=0)
            num_cells_batch = num_cells.cpu().detach().numpy().sum(axis=0)
            # Average absolute error of cells counting (average over batch)
            l = abs(pred_num_cells_batch - num_cells_batch) / num_cells.shape[0]
            ae += l
            se += l**2
            print(f'[{batch_idx}/{len(data_loader)}]\t AE: {l}')

            # Save images
            if batch_idx % 100 == 0:
                M1_norm = normalize_map(MAP)
                MAP_upsampled = upsample_map(M1_norm, dsr=8)
                save_images(MAP_upsampled, peakMAPs, batch_idx)


        print('MAE:', ae / len(data_loader))
        print('RMSE:', torch.sqrt(se / len(data_loader)))