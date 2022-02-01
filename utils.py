import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib as mpl
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from PIL import Image

cm_jet = mpl.cm.get_cmap('jet')


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


def detect_peaks_multi_channels_batch(image):
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


def normalize_image(img):
    img = img.cpu().numpy().squeeze().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    return img


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