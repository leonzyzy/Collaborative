import itertools 
import numpy as np
import random 
from itertools import product

# define a mask function to randomly mask k ROIs
# a: input a single 2D matrix
# k: number of masked ROIs
def randomMask(a,k,seed=528):
    np.random.seed(seed)
    
    # make a copy
    mask = np.copy(a).reshape(1,87,100)

    # size of matrix
    m = a.shape[0]

    # make a random mask
    for i in range(10):
        # make two copies
        z = np.copy(a)
        z[random.sample(range(0,87), k),:] = 0  #np.random.normal(0, 1,size = 100)
        z = z.reshape(1,87,100)
        mask = np.concatenate((mask, z), axis=0)
    
    return mask[1:mask.shape[0],:,:]

# define a function to augument the data size
def dataAugument(X, k):
    # make a empty numpy array
    X_mask = np.empty((0,87,100)).astype('float64')
    
    # random mask for each sample
    i = 0
    for x in X:
        print(i,X_mask.shape)
        X_mask = np.append(X_mask, randomMask(x, 1), axis=0)
        i += 1
        
    return X_mask
