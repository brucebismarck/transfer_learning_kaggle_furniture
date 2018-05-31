#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:41:19 2018

@author: wenyue

"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from os.path import join
from torch.utils.data import Dataset
from utils import return_label_dataset
from torch.utils.data import DataLoader
from os import listdir
from torch import nn
from functools import partial
import pretrainedmodels


def data_prep(ds_trans_, ds_trans_aug_):
    os.chdir("/home/wenyue/Desktop/data_playground/kaggle_furniture/")
    data_dir = '/home/wenyue/Desktop/data_playground/kaggle_furniture/'
    train_label = pd.read_csv(data_dir+"train_data.csv", index_col = 0).drop(['url'], axis = 1)
    valid_label = pd.read_csv(data_dir + "validation_data.csv",  index_col = 0).drop(['url'], axis = 1)
    train = return_label_dataset(label = train_label, data_dir = data_dir, data_name = 'train')
    valid = return_label_dataset(label = valid_label, data_dir = data_dir, data_name = 'valid')
    
    train_ds = FurnitureDataset(train, data_dir + 'train/', transform= ds_trans_aug_)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle= True, num_workers = 35  )
    valid_ds = FurnitureDataset(valid, data_dir + 'valid/', transform= ds_trans_)
    valid_dl = DataLoader(valid_ds, batch_size=16, shuffle= False, num_workers = 35) 
    
    return train_dl, valid_dl

def test_prep(transform_):
    test_ds = TestDataset('/home/wenyue/Desktop/data_playground/kaggle_furniture/', transform = transform_)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle= False, num_workers=1)
    
    return test_dl


class TestDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        test_ = pd.read_csv(root_dir+'test_data.csv', index_col = 0).drop('url', axis = 1)
        pic_list = listdir(root_dir + 'test')
        available_pic = [int(pic.rsplit(".", 1)[0]) for pic in pic_list]
        test_clean = test_.loc[test_['image_id'].isin(available_pic)]
        
        self.data = test_clean   
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = 'test/{}.jpg'.format(self.data.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)
        
        image = Image.open(fullname)
    
        if self.transform:
            image = self.transform(image)
    
        return image, self.data.iloc[idx, 0]
    

class FurnitureDataset(Dataset):
    def __init__(self, labels, root_dir, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)

        image = Image.open(fullname)
        labels = self.labels.iloc[idx, 1:].as_matrix().astype('float')
        labels = np.argmax(labels)

        if self.transform:
            image = self.transform(image)
        
        return [image, labels]  
    

class FinetunePretrainedmodels(nn.Module):
    finetune = True

    def __init__(self, num_classes: int, net_cls, net_kwards):
        super().__init__()
        self.net = net_cls(**net_kwards)
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)

xception_finetune = partial(FinetunePretrainedmodels,
                            net_cls=pretrainedmodels.xception,
                            net_kwards={'pretrained': 'imagenet'})
        


















