import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import visdom
import matplotlib as mpl
from matplotlib import pyplot as plot
from MALARIA import MALARIA
import localizerVgg


from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from matplotlib import pyplot as plt


def detect_peaks(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks


def detect_peaks_multi_channels(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, size=3) == image
    max_filter = maximum_filter(image, size=3) # We don't need it
    # plot_peak_maps(max_filter, local_max, image)
    background = (image == 0)
    eroded_background = np.zeros(shape=background.shape, dtype=bool)
    for i in range(image.shape[0]):
        eroded_background[i] = binary_erosion(background[i], structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    # plot_peak_maps(max_filter, detected_peaks, image)
    return detected_peaks

class nllloss(nn.Module):
    def __init__(self):
        super(nllloss, self).__init__()

    def forward(self, y_pred, y, num_car):
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_pred = torch.abs(y_pred - y.float())
        ret = torch.sum(y_pred) / (y_pred.shape[0]*y_pred.shape[1])
        return ret


class mcloss(nn.Module):
    def __init__(self):
        super(mcloss, self).__init__()

    """Forward: inputs y_pred and y with shape [B, C, H, W]"""
    def forward(self, y_pred, y):
        # y_pred = y_pred.view(y_pred.shape[0], -1)
        # y = y.view(y.shape[0], -1)
        y_pred = torch.abs(y_pred - y.float())
        ret = torch.sum(y_pred) / (y_pred.shape[0]*y_pred.shape[1])
        return ret


def plot_peak_maps(max_filter, peak_map, image):
    plt.figure(1)
    for i in range(3):
        plt.subplot(3, 3, 3*i+1)
        plt.imshow(image[i])
        plt.subplot(3, 3, 3*i+2)
        plt.imshow(max_filter[i])
        plt.subplot(3, 3, 3*i+3)
        plt.imshow(peak_map[i])
    plt.show()


def vis_MAP(MAP, vis, epoch, batch_idx, mapId, upsampler):
    M1 = MAP.data.cpu().contiguous().numpy().copy()
    M1_norm = (M1[0,] - M1[0,].min()) / (M1[0,].max() - M1[0,].min())
    b = np.zeros((M1_norm.shape[0], M1_norm.shape[1]*8, M1_norm.shape[2]*8))
    for i in range(b.shape[0]):
        b[i] = upsampler(torch.Tensor(M1_norm[i]))
    b0 = upsampler(torch.Tensor(M1_norm[0]))
    b0 = np.uint8(cm_jet(np.array(b0)) * 255)
    vis.image(np.transpose(b0, (2, 0, 1)), opts=dict(
        title=str(epoch) + '_' + str(batch_idx) + '_' + str(mapId) + '_heatmap'))

    # This doesn't work TODO: check why
    # b = np.uint8(cm_jet(np.array(b)) * 255)
    # b = b[0] # Select which class to visuallize
    # vis.image(np.transpose(b, (2, 0, 1)), opts=dict(
    #     title=str(epoch) + '_' + str(batch_idx) + '_' + str(mapId) + '_heatmap'))

vis = visdom.Visdom(server='http://localhost', port='8097')
cm_jet = mpl.cm.get_cmap('jet')

model = localizerVgg.localizervgg16(pretrained=True)
# model.cuda()


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = model.to(device)

# train_dataset = CARPK('', 'train', train=True)
train_dataset = MALARIA('', 'train', train=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

criterionGAM = mcloss()

optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
model.train()

for epoch in range(35):

    scheduler.step(epoch)
    for batch_idx, (data, GAM, num_cells) in enumerate(train_loader):
        data, GAM, num_cells = data.to(device, dtype=torch.float),  GAM.to(device), num_cells.to(device)

        MAP = model(data)
        if batch_idx % 1 == 0 and epoch % 1 == 0:
            img_vis = data[0].cpu()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
            vis.image(img_vis, opts=dict(title=str(epoch) + '_' + str(batch_idx) + '_image'))

            upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((data.shape[2], data.shape[3]))])
            vis_MAP(MAP, vis, epoch, batch_idx, 1, upsampler)

        # Create cMap for every class
        cMap = MAP[0,].data.cpu().numpy()
        cMap.min(axis=(1,2))
        cMap_min = cMap.min(axis=(1,2)).reshape((cMap.shape[0], 1, 1))
        cMap_max = cMap.max(axis=(1,2)).reshape((cMap.shape[0], 1, 1))
        cMap = (cMap - cMap_min) / (cMap_max - cMap_min)
        cMap[cMap < 0.1] = 0
        # Detect peaks in the predicted heat map:
        peakMAPs = detect_peaks_multi_channels(cMap)

        # # Original
        # cMap = MAP[0,0,].data.cpu().numpy()
        # cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())
        # cMap[cMap < 0.1] = 0
        # peakMAP = detect_peaks(cMap)

        # MAP & GAM shape is [B, C, H, W]. Reshape to [B, C, H*W]
        MAP = MAP.view(MAP.shape[0], MAP.shape[1], -1)
        GAM = GAM.view(GAM.shape[0], GAM.shape[1], -1)

        fark = abs(np.sum(peakMAPs, axis=(1,2)) - num_cells.numpy())

        loss = criterionGAM(MAP, GAM)
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()


        if batch_idx % 1 == 0: # was 20
            print('Epoch: [{0}][{1}/{2}]\t' 'Loss: {3}\ AE:{4}'
                 .format(epoch, batch_idx, len(train_loader), loss,  abs(fark)))

    torch.save(model.state_dict(), 'trained_model')
