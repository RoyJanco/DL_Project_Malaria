import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import visdom
import matplotlib as mpl
from matplotlib import pyplot as plt
from MALARIA import MALARIA
import localizerVgg


from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, grey_dilation

# Hyper parameters
BETA = 0.9999

# def detect_peaks(image):
#     neighborhood = generate_binary_structure(2, 2)
#     local_max = maximum_filter(image, footprint=neighborhood) == image
#     background = (image == 0)
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
#     detected_peaks = local_max ^ eroded_background
#     return detected_peaks

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
    plot_peak_maps(max_filter, local_max, image)
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
    def forward(self, y_pred, y, Eny):
        y_pred = torch.abs(y_pred - y.float())
        y_pred_sum = torch.sum(y_pred, axis=2)
        # Divide each class by Effective number of samples
        ret = torch.div(y_pred_sum, Eny)
        # Sum over all elements and normalize
        ret = torch.sum(ret) / torch.numel(y_pred)
        return ret


class l2_loss(nn.Module):
    def __init__(self):
        super(l2_loss, self).__init__()

    """Forward: inputs y_pred and y with shape [B, C, H, W]"""
    def forward(self, y_pred, y, Eny):
        y_pred = torch.square(y_pred - y.float())
        y_pred_sum = torch.sum(y_pred, axis=2)
        # Divide each class by Effective number of samples
        ret = torch.div(y_pred_sum, Eny)
        # Sum over all elements and normalize
        ret = torch.sum(ret) / torch.numel(y_pred)
        return ret


class l3_loss(nn.Module):
    def __init__(self):
        super(l3_loss, self).__init__()

    """Forward: inputs y_pred and y with shape [B, C, H, W]"""
    def forward(self, y_pred, y, Eny):
        y_pred = (torch.abs(y_pred - y.float()))**3
        y_pred_sum = torch.sum(y_pred, axis=2)
        # Divide each class by Effective number of samples
        ret = torch.div(y_pred_sum, Eny)
        # Sum over all elements and normalize
        ret = torch.sum(ret) / torch.numel(y_pred)
        return ret


class shrinkage_loss(nn.Module):
    def __init__(self, a, c):
        super(shrinkage_loss, self).__init__()
        self.a = a
        self.c = c

    """Forward: inputs y_pred and y with shape [B, C, H, W]"""
    def forward(self, y_pred, y, Eny):
        l = torch.abs(y_pred - y.float())
        y_pred = torch.square(l) / (1+torch.exp(self.a * (self.c - l)))
        y_pred_sum = torch.sum(y_pred, axis=2)
        # Divide each class by Effective number of samples
        ret = torch.div(y_pred_sum, Eny)
        # Sum over all elements and normalize
        ret = torch.sum(ret) / torch.numel(y_pred)
        return ret


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target, weight_map):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(
            delta_y1 / self.epsilon, self.alpha - y1)) * weight_map[delta_y < self.theta]
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * \
            torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = (A * delta_y2 - C) * weight_map[delta_y >= self.theta]
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def generate_weight_map(heatmap):
    k_size = 3
    weight_map = np.zeros_like(heatmap)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            dilate = grey_dilation(heatmap[i,j], size=(k_size, k_size))
            valid_ind = np.where(dilate > 0.2)
            weight_map[i, j, valid_ind[0], valid_ind[1]] = 1
    return weight_map


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


def plot_maps(data, heatmap_gt, heatmap_pred, peak_map):
    image = data.cpu().numpy().squeeze().transpose(1, 2, 0)
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Image')
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


# def plot_heatmaps(heatmap_gt, heatmap_pred):
#     plt.figure()
#     num_plots = heatmap_gt.shape[0]
#     for i in range(num_plots):
#         plt.subplot(2, num_plots, i+1)
#         plt.imshow(heatmap_gt[i])
#         plt.title(f'GT - class [{i}]')
#         plt.subplot(2, num_plots, i+num_plots)
#         plt.imshow(heatmap_pred[i])
#         plt.title(f'Pred - class [{i}]')
#     plt.show()


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


def plot_heatmaps(image, heatmap_gt, heatmap_pred, peak_maps):
    image = image.cpu().numpy().transpose(1, 2, 0)
    plt.figure(1)
    plt.imshow(image)
    plt.title('Image')
    plt.figure(2)
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


# vis = visdom.Visdom(server='http://localhost', port='8097')
cm_jet = mpl.cm.get_cmap('jet')

model = localizerVgg.localizervgg16(pretrained=True)
# model.cuda()


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = model.to(device)

# train_dataset = CARPK('', 'train', train=True)
train_dataset = MALARIA('', 'train', train=True)
# Count instances of each class
ny = torch.DoubleTensor((list(train_dataset.instances_count().values()))).to(device)
Eny = (1 - BETA**ny)/(1 - BETA)
W = torch.unsqueeze(max(Eny) / Eny, dim=1)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

# criterionGAM = mcloss()
# criterionGAM = l2_loss()
# criterionGAM = l3_loss()
# criterionGAM = shrinkage_loss(a=10, c=0.2)
criterionGAM = AdaptiveWingLoss()

optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
model.train()

for epoch in range(35):

    scheduler.step(epoch)
    for batch_idx, (data, GAM, num_cells) in enumerate(train_loader):
        data, GAM, num_cells = data.to(device, dtype=torch.float),  GAM.to(device), num_cells.to(device)

        MAP = model(data)
        # if batch_idx % 1 == 0 and epoch % 1 == 0:
        #     img_vis = data[0].cpu()
        #     img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
        #     vis.image(img_vis, opts=dict(title=str(epoch) + '_' + str(batch_idx) + '_image'))
        #
        #     upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((data.shape[2], data.shape[3]))])
        #     vis_MAP(MAP, vis, epoch, batch_idx, 1, upsampler)

        # Create cMap one class
        # cMap = MAP[0,].data.cpu().numpy()
        # cMap_min = cMap.min(axis=(1,2)).reshape((cMap.shape[0], 1, 1))
        # cMap_max = cMap.max(axis=(1,2)).reshape((cMap.shape[0], 1, 1))
        # cMap = (cMap - cMap_min) / (cMap_max - cMap_min)
        # cMap[cMap < 0.1] = 0
        # # Detect peaks in the predicted heat map:
        # peakMAPs = detect_peaks_multi_channels(cMap)

        # Create cMap for multi class
        cMap = MAP.data.cpu().numpy()
        cMap_min = cMap.min(axis=(2,3)).reshape((cMap.shape[0], cMap.shape[1], 1, 1))
        cMap_max = cMap.max(axis=(2,3)).reshape((cMap.shape[0], cMap.shape[1], 1, 1))
        cMap = (cMap - cMap_min) / (cMap_max - cMap_min)
        cMap[cMap < 0.1] = 0
        # Detect peaks in the predicted heat map:
        peakMAPs = detect_peaks_multi_channels_batch(cMap) # BUG

        # dilate GAM
        weight_map = generate_weight_map(GAM.cpu().detach().numpy())
        weight_map = torch.Tensor(weight_map.reshape((weight_map.shape[0], weight_map.shape[1], -1))).to(device)



        if batch_idx % 2 == 0:
        #     # plot_maps(data, GAM[0,0], MAP[0,0].detach().numpy(), peakMAPs[0])
            plot_heatmaps(data[0], GAM[0].cpu().detach().numpy(), MAP[0].cpu().detach().numpy(), peakMAPs[0])

        # MAP & GAM shape is [B, C, H, W]. Reshape to [B, C, H*W]
        MAP = MAP.view(MAP.shape[0], MAP.shape[1], -1)
        GAM = GAM.view(GAM.shape[0], GAM.shape[1], -1)

        # Batch size 1
        # pred_num_cells = np.sum(peakMAPs, axis=(1,2))
        # Any batch size
        pred_num_cells = np.sum(peakMAPs, axis=(2, 3))
        pred_num_cells_batch = np.sum(pred_num_cells, axis=0)
        num_cells_batch = num_cells.cpu().detach().numpy().sum(axis=0)
        # Average absolute error of cells counting (average over batch)
        fark = abs(pred_num_cells_batch - num_cells_batch) / num_cells.shape[0]

        # loss = criterionGAM(MAP, GAM, Eny)
        loss = criterionGAM(MAP, GAM, weight_map*W + 1) # For AW loss

        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()


        if batch_idx % 1 == 0: # was 20
            # print('Epoch: [{0}][{1}/{2}]\t' 'Loss: {3}\ AE:{4}'
            #      .format(epoch, batch_idx, len(train_loader), loss,  abs(fark)))
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t loss: {loss: .3e}, AE:{fark}')
            print(f'Average predicted counting of RBC: {pred_num_cells_batch[0]//num_cells.shape[0]}. GT: {num_cells_batch[0]//num_cells.shape[0]}')

    torch.save(model.state_dict(), 'trained_model')
