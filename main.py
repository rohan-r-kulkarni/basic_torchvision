import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Download training data from open datasets.
#does not download if present
data = datasets.Caltech101(
    root="data",
    #train=True,
    download=True,
    transform=
        transforms.Compose([
            transforms.CenterCrop(size=200),
            transforms.PILToTensor(),
        ])
)

classes=data.categories #list

batch_size = 1 #number of pictures processed at once

# Create data loaders.
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

for X, y in dataloader: #image, classification
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

def imshow(tensor_image, label):
    
    # img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    plt.title(label)
    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.show()

# get some random training images
for i in range(0,5):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images), classes[labels[0]])