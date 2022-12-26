# -*- coding: utf-8 -*-
from torch import nn
from lib.normalize import Normalize

# define a transformer  
class Transformer(nn.Module):
     def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(100,4,activation ='gelu',batch_first=True),
            num_layers=6
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(100,8),
            nn.LeakyReLU()
        )
        
     def forward(self, x):
        x = self.encoder(x)
        x = self.linear_stack(x)

        return x

# define a linear layer to increase dim to original
class LinearBack(nn.Module):
     def __init__(self):
        super(LinearBack, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(8,100),
            nn.ReLU()
        )
    
     def forward(self, x):
        x = self.linear_stack(x)
        return x

# define a linear layer to increase dim to original
class Distcrin(nn.Module):
     def __init__(self, Transformer):
        super(LinearBack, self).__init__()
        self.transformer = Transformer
        self.l2norm = Normalize(2)
    
     def forward(self, x):
        x = self.transformer(x)
        x = self.l2norm(x)
        return x


# define a proposed model
class ProposedModel(nn.Module):
    def __init__(self, Transformer, LinearBack):
        super(ProposedModel, self).__init__()
        self.transformer = Transformer
        self.linearBack = LinearBack
        self.Distcrin = Distcrin
    
    def forward(self, x):
        x1 = self.transformer(x)
        x2 = self.linearBack(x1)
        z = self.Distcrin(x1)
        return x2, z

# define a downtask model using pretrained transformer
class DownTask(nn.Module):
    def __init__(self, Transformer):
        super(DownTask, self).__init__()
        self.transformer = Transformer
        self.flatten = nn.Flatten()
        
        # choose any # of layer you want
        self.layer_1 = nn.Linear(87*8, 256) 
        self.layer_2 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.flatten(x)
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
