import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
from skimage import transform as sktransform
import math
from data_aug import *
import random
import json
from collections import Counter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

object_categories = ['red blood cell', 'leukocyte', 'gametocyte', 'ring', 'trophozoite', 'schizont', 'difficult']
categories_dict = {'red blood cell': 0, 'leukocyte': 1, 'gametocyte': 2, 'ring': 3, 'trophozoite': 4, 'schizont': 5, 'difficult': 6}

def twoD_Gaussian(m, n, amplitude, sigma_x, sigma_y):
    x = np.linspace(-m, m, 2 * m + 1)
    y = np.linspace(-n, n, 2 * n + 1)
    x, y = np.meshgrid(x, y)
    xo = 0.0
    yo = 0.0
    theta = 0.0
    offset = 0.0
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    sumg = g.sum()
    if sumg != 0:
        g /= sumg
    return g


class MALARIA(data.Dataset):

    def __init__(self, root, set, train = True):
        self.root = root
        self.path_devkit = os.path.join(root, 'MALARIA')
        self.path_images = os.path.join(root, 'MALARIA', 'Images')
        self.classes = object_categories
        self.train = train
        id_list_file = os.path.join(self.path_devkit, 'ImageSets/{0}.json'.format(set))
        # Opening JSON file
        f = open(id_list_file)
        # returns JSON object as a dictionary
        data = json.load(f)
        # self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data = data
        self.ids = [data_obj['image']['pathname'].replace('/images/', '') for data_obj in data]
        self.bbox = {data_obj['image']['pathname'].replace('/images/', '') : data_obj['objects'] for data_obj in data}

        print('MALARIA dataset set=%s number of classes=%03d  number of images=%d' % (
        set, len(self.classes), len(self.ids)))


    def preprocess(self, img, min_size=720, max_size=1280):
        H, W, C = img.shape
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        img = sktransform.resize(img, (int(H * scale), int(W * scale), C), mode='reflect', anti_aliasing=True)
        img = np.asarray(img, dtype=np.float32)
        return img

    def resize_bbox(self, bbox, in_size, out_size):
        bbox = bbox.copy()
        x_scale = float(out_size[0]) / in_size[0]
        y_scale = float(out_size[1]) / in_size[1]
        bbox[:, 0] = np.round(y_scale * bbox[:, 0])
        bbox[:, 2] = np.round(y_scale * bbox[:, 2])
        bbox[:, 1] = np.round(x_scale * bbox[:, 1])
        bbox[:, 3] = np.round(x_scale * bbox[:, 3])
        return bbox


    # def read_gt_bbox(self, annoFile):
    #     gt_boxes = []
    #     for line in annoFile:
    #         bbox = line.split()
    #         gt_boxes.append([int(bbox[0])+1, int(bbox[1])+1, int(bbox[2])-int(bbox[0])+1, int(bbox[3])-int(bbox[1]), int(bbox[4])])
    #     return gt_boxes

    def read_gt_bbox(self, bbox_data):
        gt_boxes = []
        for obj in bbox_data:
            obj_bbox = obj['bounding_box']
            category = obj['category']
            bbox = [obj_bbox['minimum']['c'], obj_bbox['minimum']['r'], obj_bbox['maximum']['c'], obj_bbox['maximum']['r'], categories_dict[category]]
            gt_boxes.append([int(bbox[0])+1, int(bbox[1])+1, int(bbox[2])-int(bbox[0])+1, int(bbox[3])-int(bbox[1]), int(bbox[4])])
        return gt_boxes

    def __getitem__(self, index):
        id_ = self.ids[index]
        # id_ = '164'
        path = os.path.join(self.path_images, id_)
        img = Image.open(os.path.join(path)).convert('RGB')
        if self.train:
            if random.random() > 0.5:#0.5
                transformsColor = transforms.Compose([transforms.ColorJitter(hue=0.2, saturation=0.2)])
                img = transformsColor(img)
        # img = np.asarray(img, dtype=np.float32)
        img = np.asarray(img, dtype=np.int32)

        H, W, _ = img.shape

        # img = self.preprocess(img)
        img = self.preprocess(img, H, W)

        o_H, o_W, _ = img.shape
        dSR = 1

        # GAM = np.zeros((1, int(o_H / dSR), int(o_W / dSR)))
        GAM = np.zeros((len(object_categories), int(o_H / dSR), int(o_W / dSR)))

        # annoFile = open('%s/Annotations/%s.txt' % (self.path_devkit, id_), 'r')
        bbox_data = self.bbox[id_]
        gt_bbox = np.asarray(self.read_gt_bbox(bbox_data))

        # Initialize cells count list according to the dictionary bellow:
        # {'red blood cell': 0, 'leukocyte': 1, 'gametocyte': 2, 'ring': 3, 'trophozoite': 4, 'schizont': 5, 'difficult': 6}
        num_cells = [0, 0, 0, 0, 0, 0, 0]
        # numCar = 0

        if gt_bbox.shape[0] > 0:
            gt_boxes = np.asarray(self.resize_bbox(gt_bbox, (H, W), (o_H, o_W)), dtype=np.float)

            gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] # rescaled maximum row
            gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] # rescaled maximum col

            if self.train:
                if random.random() > 1:
                    transforms_aug = Sequence([RandomRotate(45)])
                    img, gt_boxes = transforms_aug(img, gt_boxes[:,:4])
            gt_boxes = gt_boxes / dSR
            gt_boxes[:, 0::2] = np.clip(gt_boxes[:, 0::2], 0, int(o_W / dSR))
            gt_boxes[:, 1::2] = np.clip(gt_boxes[:, 1::2], 0, int(o_H / dSR))

            gt_boxes[:, 2] = abs(gt_boxes[:, 2] - gt_boxes[:, 0])
            gt_boxes[:, 3] = abs(gt_boxes[:, 3] - gt_boxes[:, 1])


            #gt_boxes = np.delete(gt_boxes, np.where(gt_boxes[:, 3]==0),0)
            #gt_boxes = np.delete(gt_boxes, np.where(gt_boxes[:, 2]==0),0)

            # Count cells from each category
            cells_counter = Counter(np.int32(gt_boxes[:, 4]))
            for key in cells_counter:
                num_cells[key] = cells_counter[key]
            total_cells_num = np.sum(num_cells)
            # numCar = gt_boxes.shape[0]

            # prepare GAM image

            for bbox in gt_boxes:
                category = np.int32(bbox[4])
                bbox = np.asarray(bbox, dtype=np.int)

                dhsizeh = int(bbox[3] / 2)
                dhsizew = int(bbox[2] / 2)

                if dhsizeh % 2 == 0:
                    dhsizeh = dhsizeh + 1

                if dhsizew % 2 == 0:
                    dhsizew = dhsizew + 1

                sigma = np.sqrt(dhsizew * dhsizeh) / (1.96*1.5)
                h_gauss = np.array(twoD_Gaussian(dhsizew, dhsizeh, sigma, math.ceil(dhsizew / 4), math.ceil(dhsizeh / 4)))
                h_gauss = h_gauss / np.max(h_gauss)

                cmin = bbox[1]
                rmin = bbox[0]
                cmax = bbox[1] + int(2*dhsizeh)+1
                rmax = bbox[0] + int(2*dhsizew)+1

                if cmax > int(o_H / dSR):
                    cmax = int(o_H / dSR)

                if rmax > int(o_W / dSR):
                    rmax = int(o_W / dSR)
                # GAM[0, cmin:cmax, rmin:rmax] = GAM[0, cmin:cmax, rmin:rmax] + h_gauss[0:cmax-cmin, 0:rmax-rmin]
                GAM[category, cmin:cmax, rmin:rmax] = GAM[category, cmin:cmax, rmin:rmax] + h_gauss[0:cmax-cmin, 0:rmax-rmin]

        # plt.figure(1)
        # plt.imshow(img)
        # plt.imshow(GAM[0], cmap='gray', alpha=0.7)
        # plt.show()

        downsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((int(o_H / 8), int(o_W / 8)), interpolation=Image.LANCZOS)])
        #
        GAM_downsampled = np.zeros((len(object_categories), int(o_H / 8), int(o_W / 8)))
        for i in range(len(object_categories)):
            GAM_downsampled[i] = downsampler(torch.Tensor(GAM[i]))
        # GAM = downsampler(torch.Tensor(GAM[4]))
        GAM_downsampled = np.array(GAM_downsampled)
        GAM_downsampled = (GAM_downsampled / GAM_downsampled.max()) * 1
        plt.figure(2)
        plt.imshow(img)
        # # plt.show()
        # plt.figure(2)
        plt.imshow(GAM[0], cmap='jet', alpha=0.4)
        plt.show()

        if img.ndim == 2:
            img = img[np.newaxis]
        else:
            img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        normalize = transforms.Normalize(mean=[0.39895892, 0.42411209, 0.40939609], std=[0.19080092, 0.18127358, 0.19950577])
        img = normalize(torch.from_numpy(img))

        return img, GAM, num_cells # Maybe return total_cells_num instead?

    def __len__(self):
        return len(self.ids)

    def get_number_classes(self):
        return len(self.classes)


if __name__ == '__main__':
    train_dataset = MALARIA('', 'train', train=True)
    x = train_dataset[26]
    print('Done')
