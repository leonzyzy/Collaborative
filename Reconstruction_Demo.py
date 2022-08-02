# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:35:19 2021

@author: liz1aq
"""

import torch
from torch import nn
from ProposedModel import *
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from MaskAugmentation import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# get self-supervised learning data
X_all = np.load('AllData.npz')['arr_0']

# make a copy of X, each 87 times
X_all_true = np.repeat(X_all, repeats=87, axis = 0)

# get labeled
y = np.load('label.npz')['arr_0']
X_labeled = np.load('features.npz')['arr_0']

# train test split only on label data
X_train_labeled, X_test_labeled, y_train, y_test = train_test_split(X_labeled,y,test_size=0.3, random_state=528)

# delete X_test_labeled from all data
def setDiff3DArray(x1,x2):
    idx = []
    
    # get shape
    m1,m2,m3 = x1.shape
    n1,n2,n3 = x2.shape
    
    x1 = x1.reshape(m1,-1)
    x2 = x2.reshape(n1,-1)


    for i in range(m1):
        for j in range(n1):
            if np.array_equal(x1[i],x2[j]):
                idx.append(i)
    return idx

# get training data for self-train
idx = setDiff3DArray(X_all,X_test_labeled)
X_self_train = np.delete(X_all, idx, axis = 0)

# normalize data
sc = StandardScaler()
X_self_train_scaled = sc.fit_transform(X_self_train.reshape(-1,87*100)).reshape(-1,87,100)
X_test_labeled = sc.transform(X_test_labeled.reshape(-1,87*100)).reshape(-1,87,100)

# Make a mask data and true data
X_ssl_masked = dataAugument(X_self_train_scaled, 1)

# make a copy of X, each 87 times
X_ssl_true = np.repeat(X_self_train_scaled, repeats=87, axis = 0)

            
################################################ Pre-training: SSL ###################################################
# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# get data loader for mini-batch training 
masked_dataloader = DataLoader(X_ssl_masked, batch_size=32)
groundtruth_loader = DataLoader(X_ssl_true, batch_size=32)

# ini each model
transformer = Transformer().to(device)
linearback = LinearBack().to(device)
proposedModel = ProposedModel(transformer, linearback).to(device)        
        
# define a JS divergence
def JSDiv(q,p):
    KL = nn.KLDivLoss(reduction='batchmean')
    p = F.softmax(p,dim=1)
    q = F.softmax(q,dim=1)
    
    log_out = ((p+q)/2).log()
    js = (KL(log_out,p)+KL(log_out,q))/2
    return js
           
# define a mse loss
mse = nn.MSELoss()        
        
# use Adam
optimizer_pretrain = torch.optim.Adam(proposedModel.parameters(), lr = 1e-3)

# demo
for _,(X, Z) in enumerate(zip(masked_dataloader, groundtruth_loader)):
    X, Z = X.float().to(device), Z.float().to(device)
    out = proposedModel(X)
    print(mse(out,Z)+JSDiv(out,Z))

    break;

# define train function
def train(masked_dataloader, groundtruth_loader):
    size = len(masked_dataloader.dataset)
    
    # open training for bn, drop out
    transformer.train()
    linearback.train()
    proposedModel.train()
    
    for batch, (X, Z) in enumerate(zip(masked_dataloader, groundtruth_loader)):
        # set to GPU
        X, Z = X.float().to(device), Z.float().to(device)
        
        # get prediction
        X_hat = proposedModel(X)
        
        # compute loss
        loss = mse(X_hat, Z)

        # Backpropagation
        optimizer_pretrain.zero_grad()
        loss.backward()
        optimizer_pretrain.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    
# pre-training model
print("========Pretraining Transformer Model=========")
epochs = 3
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train(masked_dataloader, groundtruth_loader)
print("Done!")   
        
