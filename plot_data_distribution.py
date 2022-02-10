import numpy as np
from matplotlib import pyplot as plt
from MALARIA2 import MALARIA
import torch


dataset_original = MALARIA('', 'train', train=True, num_classes=7)
dataset = MALARIA('', 'train', train=True, num_classes=2)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1100, 108], generator=torch.Generator().manual_seed(42))  # [1100, 108]

# Count instances of each class
ny_original = np.array(list(dataset_original.instances_count().values()))
ny = np.array(list(train_dataset.dataset.instances_count().values()))

# Plot data distribution
plt.figure(1, figsize=(8, 8))
bars1 = plt.bar(dataset_original.classes, ny_original, log=True, color='lightblue', edgecolor='black')
for bar in bars1:
    height = np.ceil(bar.get_height()*1.05)
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        round(bar.get_height(), 1),
        horizontalalignment='center',
        weight='bold'
    )
plt.xticks(rotation=30)
plt.title('Cells distribution')
plt.ylabel('# of cells')
# plt.savefig('figures/paper/cells_distribution')

plt.figure(2, figsize=(8, 8))
bars2 = plt.bar(dataset.classes, ny, log=True, color='lightblue', edgecolor='black')
for bar in bars2:
    height = np.ceil(bar.get_height()*1.02)
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        round(bar.get_height(), 1),
        horizontalalignment='center',
        weight='bold'
    )
plt.title('Cells distribution')
plt.xticks(rotation=30)
plt.ylabel('# of cells')
# plt.savefig('figures/paper/cells_distribution_combined')

# Plot Class Balanced term
beta = np.array([0.99, 0.999, 0.9999, 0.99999])
n = np.repeat(np.linspace(1, 100000, 1000).reshape(-1, 1), len(beta), axis=1)
W = 1 / (1 - beta**n)

labels = [rf'$\beta = {beta_val}$' for beta_val in beta]
plt.figure(3)
plt.loglog(n, W, label=labels)
plt.title('Class weighting')
plt.ylabel('W')
plt.xlabel('# of samples')
plt.legend()
plt.savefig('figures/paper/class_balance_weighting')
plt.show()


print('Done')