# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 01:01:22 2022

@author: liz27
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# ini each model
# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# define a downtask model using pretrained transformer
class DownTask(nn.Module):
    def __init__(self):
        super(DownTask, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
classifer = DownTask().to(device)
loss_ent = nn.MSELoss()
optimizer_downtask = torch.optim.Adam(classifer.parameters(), lr = 0.0001, weight_decay=0.1)


print(classifer)   
    
   # define a train function for downstreaming task
def trainDownSteam(loss_fn):
    size = len(train_dataloader.dataset)
    classifer.train()
    
    # train
    for batch, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device).to(torch.float32)
        
        # Compute prediction error
        pred = classifer(X).view(-1).to(torch.float32)
        loss = loss_ent(pred, y)
        
        # Backpropagation
        optimizer_downtask.zero_grad()
        loss.backward()
        optimizer_downtask.step()
        
        if batch % 100 == 0:
            loss, current = loss, batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        print("loss: {}".format(loss.item()))

# pre-training encoder

epochs = 10
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    trainDownSteam(loss_ent)
print("Done!")   
    
    
    
    
    
    
    
    
    
    
    
    