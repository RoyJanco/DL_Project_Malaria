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



cm_jet = mpl.cm.get_cmap('jet')



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


train_dataset = MALARIA('', 'train', train=True)

dataset_a, dataset_b = torch.utils.data.random_split(train_dataset, [1100, 108], generator=torch.Generator().manual_seed(42))

ny = torch.DoubleTensor((list(train_dataset.instances_count().values()))).to(device)


train_loader = torch.utils.data.DataLoader(dataset_a, batch_size=1, shuffle=True, num_workers=0)

print('Done')