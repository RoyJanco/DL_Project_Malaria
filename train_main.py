"""Main training script."""
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import visdom
import matplotlib as mpl
from matplotlib import pyplot as plt
# from MALARIA import MALARIA
from MALARIA2 import MALARIA
import localizerVgg
import utils
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, grey_dilation
import argparse
import os.path


# Arguments
def parse_args():
    """Parse script arguments """
    parser = argparse.ArgumentParser(description='Training models with Pytorch')
    parser.add_argument('--num_classes', '-c', default=2, type=int,
                        help='Number of classes')
    parser.add_argument('--loss_type', '-l', default='l2', type=str,
                        help='Loss function name: l2 or AW')
    parser.add_argument('--weight_map', '-w', default=False, type=bool,
                        help='Weight map multiplied with loss: True or False')
    parser.add_argument('--beta', '-b', default=0.9, type=float,
                        help='Beta value for balancing the loss function. should be between 0 to 1')
    parser.add_argument('--epochs', '-e', default=1, type=int,
                        help='Number of epochs to run')
    return parser.parse_args()


class Loss(nn.Module):
    """e_ny is the effective number of samples in class y"""
    def __init__(self, loss_type, weight_map_mode, e_ny):
        super(Loss, self).__init__()
        self.e_ny = e_ny
        self.loss_type = loss_type
        self.weight_map_mode = weight_map_mode

        # AW loss parameters
        self.omega = 14
        self.theta = 0.5
        self.epsilon = 1
        self.alpha = 2.1

        # Shrinkage loss parameters
        self.a = 10
        self.c = 0.2

    def forward(self, y_pred, y, weight_map=1):
        if self.loss_type == 'l2':
            y_pred = torch.square(y_pred - y.float()) * weight_map
            y_pred_sum = torch.sum(y_pred, dim=2)
            # Sum over all elements and normalize
            ret = torch.sum(y_pred_sum) / torch.numel(y_pred)
            return ret

        elif self.loss_type == 'AW':
            delta_y = (y - y_pred).abs()
            delta_y1 = delta_y[delta_y < self.theta]
            delta_y2 = delta_y[delta_y >= self.theta]
            y1 = y[delta_y < self.theta]
            y2 = y[delta_y >= self.theta]
            if self.weight_map_mode:
                loss1 = self.omega * torch.log(1 + torch.pow(
                    delta_y1 / self.epsilon, self.alpha - y1)) * weight_map[delta_y < self.theta]
            else:
                loss1 = self.omega * torch.log(1 + torch.pow(
                    delta_y1 / self.epsilon, self.alpha - y1))
            A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
                torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
            C = self.theta * A - self.omega * \
                torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
            if self.weight_map_mode:
                loss2 = (A * delta_y2 - C) * weight_map[delta_y >= self.theta]
            else:
                loss2 = (A * delta_y2 - C)
            return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

        elif self.loss_type == 'sl':
            l = torch.abs(y_pred - y.float())
            y_pred = torch.square(l)*torch.exp(y) / (1+torch.exp(self.a * (self.c - l)))
            y_pred_sum = torch.sum(y_pred, dim=2)
            # Sum over all elements and normalize
            ret = torch.sum(y_pred_sum) / torch.numel(y_pred)
            return ret


def generate_weight_map(heatmap):
    k_size = 3
    weight_map = np.zeros_like(heatmap)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            dilate = grey_dilation(heatmap[i,j], size=(k_size, k_size))
            valid_ind = np.where(dilate > 0.2)
            weight_map[i, j, valid_ind[0], valid_ind[1]] = 1
    return weight_map


def merge_weight_map(w_map):
    w_map = w_map.astype(bool)
    for i in range(w_map.shape[0]):
        w_merged = np.bitwise_or.reduce(w_map[i, 1:w_map.shape[1]], axis=0)
        w_map[i, 1:w_map.shape[1]] = w_merged
    return w_map


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
    image = (image - image.min()) / (image.max() - image.min())
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


def plot_heatmaps(image, heatmap_gt, heatmap_pred, peak_maps):
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
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


def get_model_name(arguments):
    if arguments.weight_map:
        model_name = f'c-{arguments.num_classes}_{arguments.loss_type}_b-{arguments.beta}_wm_e-{arguments.epochs}.pt'
    else:
        model_name = f'c-{arguments.num_classes}_{arguments.loss_type}_b-{arguments.beta}_e-{arguments.epochs}.pt'
    return model_name


if __name__ == '__main__':
    """Parse arguments and train model on dataset."""
    args = parse_args()
    model_name = get_model_name(args)
    path_save = os.path.join('saved models', model_name)
    print('Model will be saved at ' + path_save)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    cm_jet = mpl.cm.get_cmap('jet')

    dataset = MALARIA('', 'train', train=True, num_classes=args.num_classes)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1100, 108], generator=torch.Generator().manual_seed(42))  # [1100, 108]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    model = localizerVgg.localizervgg16(num_classes=train_dataset.dataset.get_number_classes(), pretrained=True)
    model = model.to(device)

    # Count instances of each class
    ny = torch.DoubleTensor((list(train_dataset.dataset.instances_count().values()))).to(device)
    Eny = (1 - args.beta**ny)/(1 - args.beta)
    W = torch.unsqueeze(max(Eny) / Eny, dim=1)
    # W = torch.unsqueeze(1 / (1 - BETA**ny), dim=1)
    # W2 = torch.unsqueeze(max(ny) / ny, dim=1)

    criterionGAM = Loss(args.loss_type, args.weight_map, Eny)

    optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    model.train()

    thr = 0.5
    for epoch in range(args.epochs):
        # scheduler.step(epoch)
        for batch_idx, (data, GAM, num_cells) in enumerate(train_loader):
            data, GAM, num_cells = data.to(device, dtype=torch.float),  GAM.to(device), num_cells.to(device)

            MAP = model(data)

            # Create cMap for multi class
            cMap = MAP.data.cpu().numpy()
            cMap_min = cMap.min(axis=(2,3)).reshape((cMap.shape[0], cMap.shape[1], 1, 1))
            cMap_max = cMap.max(axis=(2,3)).reshape((cMap.shape[0], cMap.shape[1], 1, 1))
            cMap = (cMap - cMap_min) / (cMap_max - cMap_min)
            cMap[cMap < thr] = 0
            # Detect peaks in the predicted heat map:
            peakMAPs = utils.detect_peaks_multi_channels_batch(cMap) # BUG

            # Generate weight map
            if args.weight_map:
                # dilate GAM
                weight_map = generate_weight_map(GAM.cpu().detach().numpy())
                # Merge weight maps
                # weight_map = merge_weight_map(weight_map)
                weight_map = torch.Tensor(weight_map.reshape((weight_map.shape[0], weight_map.shape[1], -1))).to(device)


            # if batch_idx % 2 == 0:
            #     plot_heatmaps(data[0], GAM[0].cpu().detach().numpy(), MAP[0].cpu().detach().numpy(), peakMAPs[0])

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

            if args.weight_map:
                loss = criterionGAM(MAP, GAM, weight_map*W + 1)
            else:
                loss = criterionGAM(MAP, GAM)

            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

            if batch_idx % 1 == 0: # was 20
                print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t loss: {loss: .3e}, AE:{fark}')
                print(f'Average predicted counting of RBC: {pred_num_cells_batch[0]//num_cells.shape[0]}. GT: {num_cells_batch[0]//num_cells.shape[0]}')

        torch.save(model.state_dict(), path_save)
