#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:37:14 2018

@author: wenyue
"""
import numpy as np
from os import listdir
from tqdm import tqdm
from torch.autograd import Variable
import torch


def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    axis.imshow(inp)
    
def return_label_dataset(label, data_dir, data_name = 'train'):
    pic_list = listdir(data_dir + data_name)
    # some pictures are missing, only take available pics
    available_pic = [int(pic.rsplit(".", 1)[0]) for pic in pic_list]
    
    label_clean = label.loc[label['image_id'].isin(available_pic)]
    
    print(len(pic_list)) # 190118 training pictures
    print(len(label_clean))

    label_clean['target'] = 1
    label_clean['rank'] = label_clean.groupby('label_id').rank()['image_id']
    output = label_clean.pivot('image_id', 'label_id', 'target').reset_index().fillna(0)
    return output


def predict(model, dataloader):
    use_gpu = torch.cuda.is_available()
    all_labels = []
    all_outputs = []
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, labels in pbar:
        all_labels.append(labels)

        inputs = Variable(inputs, volatile = True)
        if use_gpu:
            inputs = inputs.cuda()
        
        outputs = model(inputs)
    
        all_outputs.append(outputs.data.cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    if use_gpu:
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()
    return all_labels, all_outputs


def safe_stack_2array(a, b, dim=0):
    if a is None:
        return b
    return torch.stack((a, b), dim=dim)


def predict_tta(model, dataloaders):
    prediction = None
    lx = None
    for i in range(len(dataloaders)):
        dataloader = dataloaders[i]
        lx, px = predict(model, dataloader)
        if i == 0 or i == 1:
            prediction = safe_stack_2array(prediction, px, dim=-1)
        else:
            prediction = torch.cat((prediction, px.unsqueeze_(-1)), dim = -1)        
    return lx, prediction
