#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:48:04 2018

@author: wenyue
"""
import misc
import utils
import torch

import numpy as np
import pandas as pd


from torchvision import models
from torch.autograd import Variable
import torch.nn as nn

from tqdm import tqdm
import transform as trans
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

#%%
def predict_process(path):
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    #model.fc = nn.Sequential(nn.Dropout(), nn.Linear(num_ftrs, 128))
    model.fc = nn.Linear(num_ftrs, 128)
    model = model.cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    
    tta = [trans.preprocess, trans.preprocess_hflip]

    data_loaders = []
    for transform in tta:
        data_loaders.append(misc.test_prep(transform_=transform))
    
    # Do the prediction
    lx, px = utils.predict_tta(model, data_loaders)

    # Create the prediction corredponding index list
    idx_list = []
    pbar = tqdm(data_loaders[0], total=len(data_loaders[0]))
    for image,idx in pbar:
        idx_list = idx_list + idx.cpu().numpy().tolist()

    test_prob = F.softmax(Variable(px.cpu()), dim = 1).data.numpy()
    test_prob = test_prob.mean(axis = 2)

    test_predicted = np.argmax(test_prob, axis = 1)
    test_predicted += 1
    #print(test_predicted.size)
    
    idx_series = pd.Series(idx_list).to_frame()
    predicted = pd.Series(test_predicted).to_frame()
    temp = pd.merge(idx_series, predicted, left_index= True, right_index = True, how = 'outer')
    temp.columns = ['id','predicted']
    
    read_id = pd.Series(range(1,12801)).rename('id').to_frame()
    
    output = pd.merge(read_id, temp, left_on = 'id',right_on = 'id', how = 'left').fillna(20)
    output['predicted'] = output['predicted'].astype(int)
    
    name = path.split('/')[-1][:-4]
    output.to_csv('/home/wenyue/Desktop/data_playground/kaggle_furniture/output/'+name+'.csv', index = False)
    
    return output

#%%
output = predict_process('/home/wenyue/Desktop/data_playground/kaggle_furniture/model_dump/month_058_acc_0.736_log_loss_1.038_densenet201.pth')
output.info()




