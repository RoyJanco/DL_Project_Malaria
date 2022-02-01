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


def save_images(img, map, peak_map, batch_idx):
    # Normalzize img
    img = img.cpu().numpy().squeeze().transpose(1, 2, 0)
    img = np.uint8((img - img.min()) * 255 / (img.max() - img.min()))
    img = Image.fromarray(img)
    img.save("results/image-batch_" + str(batch_idx) + ".bmp")
    # Get first peakmap from the batch
    peak_map = np.uint8(np.array(peak_map[0]) * 255)
    for i in range(map.shape[0]):
        a = map[i]
        ima = Image.fromarray(a)
        peakI = Image.fromarray(peak_map[i]).convert("RGB")
        peakI = peakI.resize((1600, 1200))
        ima.save("results/heatmap-class_" + str(i) + '-batch_' + str(batch_idx) + ".bmp")
        peakI.save("results/peakmap-class_" + str(i) + '-batch_' + str(batch_idx) + ".bmp")


def find_inliers(x):
    """ Returns inliers from x. x shape is (num_samples, num_classes)"""
    mean = np.mean(x, axis=0)
    standard_deviation = np.std(x, axis=0)
    distance_from_mean = abs(x - mean)
    prctile = np.percentile(x, 90, axis=0)
    # max_deviations = 1.2
    # inliers = distance_from_mean < max_deviations * standard_deviation
    inliers = distance_from_mean < prctile

    inliers = np.bitwise_and.reduce(inliers, axis=1)
    return inliers


def plot_graphs(gt, pred):
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.scatter(gt[:, 0], pred[:, 0])
    plt.ylabel('Model count')
    plt.xlabel('GT count')
    plt.title('Not infected cells')
    plt.subplot(1, 2, 2)
    plt.scatter(gt[:, 1], pred[:, 1])
    plt.ylabel('Model count')
    plt.xlabel('GT count')
    plt.title('Infected cells')
    plt.show()


if __name__ == '__main__':
    NUM_CLASSES = 2

    cm_jet = mpl.cm.get_cmap('jet')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load pretrained model

    dataset = MALARIA('', 'train', train=True, num_classes=NUM_CLASSES)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1200, 8], generator=torch.Generator().manual_seed(42)) # [1100, 108]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = localizerVgg.localizervgg16(num_classes=test_dataset.dataset.get_number_classes(), pretrained=True)
    state_dict = torch.load('saved models/c2_l2_b0_e2.pt', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    # Initialize absolute error and MSE accumulators
    # ae, se = torch.zeros(NUM_CLASSES), torch.zeros(NUM_CLASSES)
    predicted_count, gt_count = [], []
    thr = 0.5
    with torch.no_grad():
        for batch_idx, (data, GAM, num_cells) in enumerate(test_loader, 0):
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

            # Append counts to lists
            predicted_count.append(pred_num_cells_batch)
            gt_count.append(num_cells_batch)

            # Average absolute error of cells counting (average over batch)
            l = abs(pred_num_cells_batch - num_cells_batch) / num_cells.shape[0]

            # ae += l
            # se += l**2
            print(f'[{batch_idx}/{len(test_loader)}]\t AE: {l}')

            # Save images
            if batch_idx % 100 == 0:
                M1_norm = normalize_map(MAP)
                MAP_upsampled = upsample_map(M1_norm, dsr=8)
                save_images(data, MAP_upsampled, peakMAPs, batch_idx)

        predicted_count = np.array(predicted_count, dtype=int)
        gt_count = np.array(gt_count, dtype=int)

        total_num_samples = gt_count.shape[0]
        # Remove outliers
        inliers = find_inliers(np.abs(predicted_count - gt_count))
        gt_count = gt_count[inliers, :]
        predicted_count = predicted_count[inliers, :]
        inliers_num = gt_count.shape[0]
        print(f'{total_num_samples - inliers_num} outliers omitted.')

        # Plot results
        plot_graphs(gt_count, predicted_count)
        mean_gt = np.mean(gt_count, axis=0)

        MAE = np.mean(np.abs(predicted_count - gt_count), axis=0)
        nMAE = MAE / mean_gt
        RMSE = np.sqrt(np.mean((predicted_count - gt_count)**2, axis=0))
        NRMSE = RMSE / mean_gt

        print('MAE: ', MAE)
        print('nMAE: ', nMAE)
        print('RMSE: ', RMSE)
        print('NRMSE: ', NRMSE)

        print('Done')

