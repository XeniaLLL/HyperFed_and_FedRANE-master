import os.path

import torch
import torchvision
from torchvision.transforms import ToTensor
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("/home/lxt/code/datasets", train=False, download=True, transform=ToTensor()), batch_size=128, shuffle=True
)
examples= enumerate(test_loader)
batch_idx, (example_data, example_labels)= next(examples)

import matplotlib.pyplot as plt

if not os.path.exists("mnist"):
    os.makedirs("mnist")
for i in range(128):
    fig = plt.figure(figsize=(2.8, 2.8))
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'mnist/{i}.png', transparent=True)
    # plt.show()
