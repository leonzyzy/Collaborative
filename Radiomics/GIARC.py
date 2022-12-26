# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:01:51 2022

@author: liz27
"""

import radiomics
import sys, os
import numpy as np
import nibabel as nib
import nrrd
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from collections import OrderedDict

path = 'C:/Users/liz27/OneDrive/Documents/CCHMC/radiomics'

params = 'C:/Users/liz27/OneDrive/Documents/CCHMC/radiomics/Params.yaml'
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# read data
img, head = nrrd.read('SSFE_38_upper.nrrd')
label, head = nrrd.read('seg_38_upper.nrrd')
mask = (label == 1)*1 # first region

all_subject_features = []
features = pd.DataFrame(columns = range(100))

data_spacing = [0.7031,0.7031,6]


# get mask for each ROI
for j in np.unique(label):
    # skip 0
    if j > 0:
        mask = (label==j)*1
        # only check existed ROI
        for x in range(mask.shape[2]):
            if len(np.unique(mask[:,:,x])) > 1:
                # labelled slice img
                img_slice = img[:,:,x]   
                sitk_img = sitk.GetImageFromArray(img_slice)
                sitk_img.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2])))
                sitk_img = sitk.JoinSeries(sitk_img)
                    
                # labelled slice mask
                mask_slice = mask[:,:,x]
                sitk_mask = sitk.GetImageFromArray(mask_slice)
                sitk_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2])))
                sitk_mask = sitk.JoinSeries(sitk_mask)
                sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32)
                    
                try:
                    radiomics_features = extractor.execute(sitk_img, sitk_mask) 
                    kept_features = OrderedDict((k, radiomics_features[k]) for k in radiomics_features if 'diagnostics' not in k)
                    features.loc[j] = kept_features.values()
                    features.columns = kept_features.keys()
                except ValueError:
                    print('Single Voxel, fill 0 instead')
                    features.loc[j] = [0]*100
                    features.columns = kept_features.keys()
   # add into a list
all_subject_features.append(features.to_numpy())
    
    

# remove duplicates 

    
# save the new data to csv
for i in range(len(all_subject_features)):
    df = pd.DataFrame(all_subject_features[i], columns = features.columns)
    
    
# rename
ROIs = ['TC2', 'TC2_ROI', 'TC1', 'TC1_ROI', 'DC1', 'J2', 'DC2', 'DC2_ROI', 'J1', 'AC1', 'AC2', 'AC2_ROI']

df.index = ROIs
df.to_csv('AllRadiomicsFeatures_PT-767_ARCCoreMRE_MRI038_2.csv')













    
    
    