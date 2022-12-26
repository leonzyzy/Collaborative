# -*- coding: utf-8 -*-
import torch
from torch import nn,Tensor
from ProposedModel import *
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from MaskAugmentation import *
from sklearn.model_selection import train_test_split
from SubjectDistLoss import *

# use your dataset
X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.5, random_state=528)

# switch to torch
X_train, X_test, y_train, y_test = torch.from_numpy(X_train),torch.from_numpy(X_test),torch.from_numpy(y_train),torch.from_numpy(y_test)

# Make a mask data 
X_ssl_masked_train = dataAugument(X_train, 20)
X_ssl_masked_train = X_ssl_masked_train.reshape(100,10,87,100)
X_ssl_masked_train = Tensor(X_ssl_masked_train)


# set batch size
b = 8
masked_contrasive_dataloader = DataLoader(X_ssl_masked_train, batch_size=b)

# ini each model
# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

transformer = Transformer().to(device)

# use Adam
criterion = DiscriminationLoss(b, temperature = 1, div='cosine')
optimizer_pretrain = torch.optim.Adam(transformer.parameters(), lr = 0.001,  momentum=0.9, nesterov=True)

def train(masked_contrasive_dataloader):
    transformer.train()
    for _,X in enumerate(masked_contrasive_dataloader):
        X = X.float().to(device)
        optimizer_pretrain.zero_grad()
        losses = 0
        
        # for each pair
        for i in range(0,10):
            for j in range(0,10):
                xi = X[:,i,:,:]
                xj = X[:,j,:,:]
                # compute code
                zi = torch.flatten(transformer(xi), start_dim=1)
                zj = torch.flatten(transformer(xj), start_dim=1)
                    
                # compute loss
                loss = criterion(zi,zj)
                losses =+ loss
        
        losses.mean().backward()
        optimizer_pretrain.step()
        print("loss: {}".format(losses))
        
# pre-training model
print("========Pretraining Transformer Model=========")
epochs = 10
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train(masked_contrasive_dataloader)
print("Done!")   

# Make a mask data 
X_ssl_masked_train_context = dataAugument(X_train, 20)

# make a copy of X, each 10 times, only used for context prediction
X_ssl_true_train_context = np.repeat(X_train, repeats=10, axis = 0)

# Test task 2
groundtruth_loader = DataLoader(X_ssl_masked_train_context, batch_size=b)
masked_dataloader = DataLoader(X_ssl_true_train_context, batch_size=b)

# define models
linearback = LinearBack().to(device)
proposedModel = ProposedModel(transformer, linearback).to(device)        

# define a mse loss
mse = nn.MSELoss()   

# use Adam
optimizer_context = torch.optim.Adam(proposedModel.parameters(), lr = 0.001)

        
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
        optimizer_context.zero_grad()
        loss.backward()
        optimizer_context.step()
        
        print("loss: {}".format(loss.item()))
       
            
# pre-training model
print("========Pretraining Transformer Model=========")
epochs = 10
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train(masked_dataloader, groundtruth_loader)
print("Done!")   


# downsteam task:
classifer = DownTask(transformer).to(device)
loss_ent = nn.BCEWithLogitsLoss()
optimizer_downtask = torch.optim.Adam(transformer.parameters(), lr = 0.001, momentum=0.9, nesterov=True)


# define a train function for downstreaming task
def trainDownSteam(X_train, y_train, loss_fn):
    classifer.train()
    
    # get data loader
    train_dataloader = DataLoader(X_train, batch_size=b)
    label_loader = DataLoader(y_train, batch_size=b)
    
    # train
    for X,y in zip(train_dataloader, label_loader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = classifer(X).view(-1)
        loss = loss_ent(pred, y)
        
        # Backpropagation
        optimizer_downtask.zero_grad()
        loss.backward()
        optimizer_downtask.step()
        
        print("loss: {}".format(loss.item()))

# pre-training encoder
epochs = 5
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    trainDownSteam(X_train, y_train, loss_ent)
print("Done!")   
    
# predict
y_pred = torch.round(classifer(X_test.to(device)).view(-1))
