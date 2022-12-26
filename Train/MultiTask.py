import torch
from torch import nn,Tensor
from ProposedModel import *
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
from MaskAugmentation import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from contrastiveLoss import *
from sklearn.model_selection import train_test_split


# set seed
np.random.seed(528)

# define a dataset (200, 87, 108)
torch.manual_seed(528)
df = torch.rand(100,87,100).numpy()
label = torch.empty(100).random_(2).numpy()

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.5, random_state=528)

# switch to torch
X_train, X_test, y_train, y_test = torch.from_numpy(X_train),torch.from_numpy(X_test),torch.from_numpy(y_train),torch.from_numpy(y_test)

# Make a mask data 
X_ssl_masked_train = dataAugument(X_train, 1)

# make a copy of X, each 87 times, only used for context prediction
X_ssl_true_train = np.repeat(X_train, repeats=87, axis = 0)

# random shuffling 
np.random.shuffle(X_ssl_masked_train)
np.random.shuffle(X_ssl_true_train)


# create a pair data
l1 = int(len(X_ssl_masked_train)/2)
l2 = len(X_ssl_masked_train)

X_ssl_masked_train_p1 = X_ssl_masked_train[0:l1,:,:]
X_ssl_masked_train_p2 = X_ssl_masked_train[l1:l2,:,:]

# pair dataset
b = 25
masked_dataset_contrasive = TensorDataset( Tensor(X_ssl_masked_train_p1), Tensor(X_ssl_masked_train_p2) )
masked_dataset_contrasive_dataloader = DataLoader(masked_dataset_contrasive, batch_size=b)


for batch,(Xi,Xj) in enumerate(masked_dataset_contrasive_dataloader):
    Xi = Xi.float().to(device)
    Xj = Xj.float().to(device)
    
    zi = torch.flatten(transformer(Xi), start_dim=1)
    zj = torch.flatten(transformer(Xj), start_dim=1)
    print(zi.shape)
    break

# ini each model
# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

transformer = Transformer().to(device)

# get data loader for mini-batch training 
b = 25
masked_dataset_contrasive_dataloader = DataLoader(masked_dataset_contrasive, batch_size=b)

# use Adam
criterion = SimCLR_Loss(batch_size = b, temperature = 1)
optimizer_pretrain = torch.optim.SGD(transformer.parameters(), lr = 1e-3)

def train(masked_dataset_contrasive_dataloader):
    size = len(masked_dataset_contrasive_dataloader.dataset)
    
    # open training for bn, drop out
    transformer.train()
    
    for batch,(Xi,Xj) in enumerate(masked_dataset_contrasive_dataloader):
        Xi = Xi.float().to(device)
        Xj = Xj.float().to(device)
      
        zi = torch.flatten(transformer(Xi), start_dim=1)
        zj = torch.flatten(transformer(Xj), start_dim=1)
        
        loss = criterion(zi,zj)

        # Backpropagation
        optimizer_pretrain.zero_grad()
        loss.backward()
        optimizer_pretrain.step()
        
        print("loss: {}".format(loss.item()))
       
    
# pre-training model
print("========Pretraining Transformer Model=========")
epochs = 10
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train(masked_dataset_contrasive_dataloader)
print("Done!")   



# training second task
groundtruth_loader = DataLoader(X_ssl_true_train, batch_size=25)
masked_dataloader = DataLoader(X_ssl_masked_train, batch_size=25)


# define models
linearback = LinearBack().to(device)
proposedModel = ProposedModel(transformer, linearback).to(device)        

# define a mse loss
mse = nn.MSELoss()   

# use Adam
optimizer_context = torch.optim.Adam(proposedModel.parameters(), lr = 1e-2)

        
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
loss_ent = nn.BCELoss()
optimizer_downtask = torch.optim.SGD(classifer.parameters(), lr = 0.1, weight_decay=0.01)


# define a train function for downstreaming task
def trainDownSteam(X_train, y_train, loss_fn):
    classifer.train()
    
    # get data loader
    train_dataloader = DataLoader(X_train, batch_size=10)
    label_loader = DataLoader(y_train, batch_size=10)
    
    # train
    for X,y in zip(train_dataloader, label_loader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = classifer(X).view(-1)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer_downtask.zero_grad()
        loss.backward()
        optimizer_downtask.step()
        
        print("loss: {}".format(loss.item()))

# pre-training encoder

epochs = 2000
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    trainDownSteam(X_train, y_train, loss_ent)
print("Done!")   
    

# predict
y_pred = torch.round(classifer(X_test.to(device)).view(-1))
accuracy = (y_pred.to(device) == y_test.to(device)).sum()/len(y_test)
accuracy
    
train_dataloader = DataLoader(X_train, batch_size=10)
label_loader = DataLoader(y_train, batch_size=10)
    
 # train
for X,y in zip(train_dataloader, label_loader):
    X, y = X.to(device), y.to(device)
        
    # Compute prediction error
    pred = classifer(X).view(-1)
    loss = loss_ent(pred, y)
