# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 20:37:06 2021

@author: liz27
"""

import torch
import math
from positional_encodings import PositionalEncoding1D
p_enc_1d = PositionalEncoding1D(10)
x = torch.zeros((1,6,10))
print(p_enc_1d(x).shape) # (1, 6, 10)