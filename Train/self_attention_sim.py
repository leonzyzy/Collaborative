# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:42:43 2021
@author: Zhiyuan Li
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# define a dataset (200, 87, 108)
torch.manual_seed(528)
df = torch.rand(1000,87,108).numpy()
label = torch.empty(1000).random_(2).numpy()

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.3, random_state=528)

# switch to torch
X_train, X_test, y_train, y_test = torch.from_numpy(X_train),torch.from_numpy(X_test),torch.from_numpy(y_train),torch.from_numpy(y_test)


# define a mask function to randomly mask k ROIs 
# a: input matrix 
# k: number of masked ROIs
def randomNoise(a,k,seed=528):
    np.random.seed(seed)
    
    # make a copy
    Z = np.copy(a)
    
    # size of matrix
    m,n = Z[0].shape
    
    # make a random mask
    pos_mask = np.random.randint(0,m,k)
    
    # mask selected ROIs
    for z in Z:
        z[:,pos_mask] = z[:,pos_mask] + np.random.normal(0,1,m).reshape(87,1)
    
    # return noise matrix
    return torch.from_numpy(Z), pos_mask


# define a self-attention encoder
class SelfAttentionEncoder(nn.Module):
     def __init__(self):
        super(SelfAttentionEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(108,4,batch_first=True),
            num_layers=1
        )
     def forward(self, x):
        x = self.encoder(x)
        return x
    
# define a fully connected layer only for noise data
class PartialConnectNN(nn.Module):
    def __init__(self):
        super(PartialConnectNN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(108,108)  
        )
    def forward(self, x):
        x = self.linear(x)
        return x
    
# define the proposed model
class ProposedModel(nn.Module):
    def __init__(self, SelfAttentionEncoder, PartialConnectNN):
        super(ProposedModel, self).__init__()
        self.selfAttentionEncoder = SelfAttentionEncoder
        self.partialConnectNN = PartialConnectNN
    
    def forward(self, x):
        atten_output = self.selfAttentionEncoder(x)
        mask_output = atten_output[:,pos_mask,:]
        output = self.partialConnectNN(mask_output)
        return output
    
# define a classifier using pre-trained model
class DownStreamingNN(nn.Module):
    def __init__(self, SelfAttentionEncoder):
        super(DownStreamingNN, self).__init__()
        self.pre_trained_encoder = SelfAttentionEncoder
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(87*108,256),
            nn.ReLU(),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.pre_trained_encoder(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
    
    
encoder = SelfAttentionEncoder().to(device)
regressor = PartialConnectNN().to(device)
model = ProposedModel(encoder, regressor).to(device)
predictive_classifier = DownStreamingNN(encoder).to(device)

# define loss: mse and binary cross entropy
loss_mse = nn.MSELoss()
loss_ent = nn.BCELoss()

# all model using same optimizer: Adam
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
optimizer_downtask = torch.optim.Adam(predictive_classifier.parameters(), lr = 0.01)

# get noise matrix
train_data, pos_mask = randomNoise(X_train,10)


# define a train function
def train(train_data, ground_truth, pos_mask, loss_fn):
    encoder.train()
    regressor.train()
    model.train()
    
    # get data loader
    train_dataloader = DataLoader(train_data, batch_size=32)
    groundtruth_loader = DataLoader(ground_truth, batch_size=32)
    
    
    for (X, Z) in zip(train_dataloader, groundtruth_loader):
        X, Z = X.to(device), Z.to(device)
        
        # find masked ground truth
        truth = Z[:,pos_mask,:]
        pred = model(X)
        
        # compute loss
        loss = loss_fn(pred, truth)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("loss: {}".format(loss.item()))
       
# pre-training encoder
print("========Pretraining Self-Attention Encoder=======")
epochs = 100
for step in range(epochs):
    if (step+1) % 10 == 0:
        print("Epoch: {}, ".format(step+1))   
        train(train_data, X_train, pos_mask, loss_mse)


# define a train function for downstreaming task
def trainDownSteam(train_data, y_train, loss_fn):
    predictive_classifier.train()
    
    # get data loader
    train_dataloader = DataLoader(train_data, batch_size=32)
    label_loader = DataLoader(y_train, batch_size=32)
    
    # train
    for X,y in zip(train_dataloader, label_loader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = predictive_classifier(X).view(-1)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("loss: {}".format(loss.item()))
    

# pre-training encoder
print("\n======Training Predictive Classifier===========")
e = 2000
for step in range(e):
    if (step+1) % 100 == 0:
        print("Epoch: {}, ".format(step+1))   
        trainDownSteam(train_data, y_train, loss_ent)
    

# predict
y_pred = torch.round(predictive_classifier(X_test.to(device)).view(-1))
accuracy = (y_pred.to(device) == y_test.to(device)).sum()/len(y_test)
accuracy
