import itertools
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from math import comb

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

 
# define a dataset (200, 87, 108)
torch.manual_seed(528)
df = torch.rand(300,87,108).numpy()
label = torch.empty(300).random_(2).numpy()

# switch to torch

# define a mask function to randomly mask k ROIs
# a: input a single 2D matrix
# k: number of masked ROIs
def randomMask(a,k,seed=528):
    np.random.seed(seed)
    
    # make a copy
    mask = np.copy(a).reshape(1,87,108)

    # size of matrix
    m = a.size()[0]

    # make a random mask
    for pos in itertools.combinations(list(range(0,m)), k):    
        # make two copies
        z = np.copy(a)
        z[pos[0],:] = 0
        z = z.reshape(1,87,108)
        mask = np.concatenate((mask, z), axis=0)
    
    return mask[1:mask.shape[0],:,:]

# return a masked sample
mask_sample = randomMask(df[0],1)


# define a function to augument the data size
def dataAugument(X):
    # make a copy
    X_mask = np.copy(X[0]).reshape(1,87,108)
    
    # random mask for each sample
    for x in X:
        X_mask = np.concatenate((X_mask, randomMask(x, 1)), axis=0)
    
    return X_mask[1:X_mask.shape[0],:,:]


# pull all data
df_sim = dataAugument(df)
df_sim.shape
