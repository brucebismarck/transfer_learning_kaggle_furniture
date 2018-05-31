#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 20:17:32 2018

@author: wenyue
"""
from __future__ import print_function, division
import os

os.chdir("/home/wenyue/Desktop/data_playground/kaggle_furniture/src")

import misc
import utils
from transform import preprocess,preprocess_aug ,preprocess_hflip, preprocess_scale_1, preprocess_scale_2, \
    preprocess_brightness_darker,  preprocess_brightness_lighter, preprocess_contrast_blur,\
    preprocess_contrast_sharp, preprocess_gamma_large,  preprocess_gamma_small
import transform as trans
    
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import time
import datetime    
import glob
import copy

num_ftrs = 128
netname = 'densenet201'

def get_model(model_name):
    if 'resnet' in model_name: 
        if model_name == 'resnet50':
            model = models.resnet50(pretrained= True)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained= True)
        else:
            print('You need redefine the function')
        model.fc = nn.Linear(model.fc.in_features, num_ftrs)
    
    elif 'densenet' in model_name:
        if model_name == 'densenet121':
            model = models.densenet121(pretrained= True)
        elif model_name == 'densenet161':
            model = models.densenet161(pretrained = True)
        elif model_name == 'densenet201':
            model = models.densenet201(pretrained= True)
        model.classifier = nn.Linear(model.classifier.in_features, num_ftrs)
    elif model_name == 'inception':
        model = misc.xception_finetune(128)
    else:
        print('need more information, dense201 used')
        model = models.densenet201(pretrained= True)
        model.classifier = nn.Linear(model.classifier.in_features, num_ftrs)        
    return model


def train_model(dataloaders, model_name, num_epochs=30):
    use_gpu = torch.cuda.is_available()
    model = get_model(model_name)
    if use_gpu:
        model = model.cuda()    
    since = time.time()
     
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    if 'densenet' in model_name:
        params = model.classifier.parameters()
    elif 'resnet' in model_name:
        params = model.fc.parameters()
    elif model_name == 'inception':
        params = model.fresh_params()
  
    min_loss = float("Inf")
    max_acc = 0.0
    lr = 0
    patience = 0

    #CPU seed
    torch.manual_seed(1234)
    # GPU seed
    torch.cuda.manual_seed_all(1234)
      
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    
        if epoch == 1:
            lr = 0.00003
            print(f'[*] set lr = {lr}')
        if patience == 2:
            patience == 0
            model.load_state_dict(torch.load('model_dump/best_val_weight.pth'))
            lr = lr/10
            print(f'[*] set lr = {lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[*] set lr = {lr}')
            optimizer = torch.optim.Adam(params, lr = lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=lr/10)
        
        running_loss = 0.0
        running_corrects = 0
        
        model.train()                
        pbar = tqdm(dataloaders['train'], total=len(dataloaders['train']))
        for inputs, labels in pbar:
            inputs,labels =  Variable(inputs), Variable(labels)
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
                
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim = 1)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
            
            loss.backward()
            optimizer.step()

            train_epoch_loss = running_loss / len(dataloaders['train'].dataset)
            train_epoch_acc = running_corrects / len(dataloaders['train'].dataset)
            
        print(f'[+] epoch {epoch} {train_epoch_loss:.5f} {train_epoch_acc:.3f}')
        print('lr for epoch %d is %s' %(epoch, optimizer.param_groups[0]["lr"]))
        
        lx, px = utils.predict(model, dataloaders['valid'])
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
        #if accuracy > max_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),'model_dump/best_val_weight.pth')
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            #max_acc = accuracy
            patience = 0
        else:
            patience += 1

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    
    modelname = model_name
    #return model, min_loss, accuracy, modelname
    return model, log_loss, max_acc, modelname


def predict_process(paths, model_names):
    use_gpu = torch.cuda.is_available()
    
    
    for i in range(len(paths)):
        model = get_model(model_names[i])
        if use_gpu:
            model = model.cuda()

        model.load_state_dict(torch.load(paths[i]))
        model.eval()
    
        tta = [preprocess, preprocess_hflip, 
               preprocess_aug, preprocess_scale_1, preprocess_scale_2]#,\
         #preprocess_brightness_darker, preprocess_brightness_lighter\
        # , preprocess_contrast_blur,\
        # preprocess_contrast_sharp, preprocess_gamma_large, \
        # preprocess_gamma_small]

        data_loaders = [] 
        for transform in tta:
            data_loaders.append(misc.test_prep(transform_=transform))
    
        lx, px = utils.predict_tta(model, data_loaders)
    
        test_prob = F.softmax(Variable(px.cpu()), dim = 1).data.numpy()
        
        if i == 0:
            test_prob_1 = test_prob
        else:
            test_prob_1 = np.concatenate((test_prob,test_prob_1), axis = 2)
        test_prob_save = test_prob.mean(axis = 2)
        np.savetxt('/home/wenyue/Desktop/data_playground/kaggle_furniture/prep/' + paths[i].split('/')[-1].split('.pth')[0],  test_prob_save, delimiter = ',')

    test_prob_1 = test_prob_1.mean(axis = 2)
    test_predicted = np.argmax(test_prob_1, axis = 1)
    test_predicted += 1    
        
    idx_list = []
    pbar = tqdm(data_loaders[0], total=len(data_loaders[0]))
    for image,idx in pbar:
        idx_list = idx_list + idx.cpu().numpy().tolist()
    
    
    #print(test_predicted.size)
    
    idx_series = pd.Series(idx_list).to_frame()
    predicted = pd.Series(test_predicted).to_frame()
    temp = pd.merge(idx_series, predicted, left_index= True, right_index = True, how = 'outer')
    temp.columns = ['id','predicted']
    
    read_id = pd.Series(range(1,12801)).rename('id').to_frame()
    
    output = pd.merge(read_id, temp, left_on = 'id',right_on = 'id', how = 'left').fillna(20)
    output['predicted'] = output['predicted'].astype(int)
    
    name = paths[0].split('/')[-1][:-4]
    output.to_csv('/home/wenyue/Desktop/data_playground/kaggle_furniture/output/'+name+'_' + str(len(tta))+'tta_new.csv', index = False)
    
    return output


def main():  
    train_dl, valid_dl = misc.data_prep(preprocess, preprocess_aug)
    dloaders = {'train':train_dl, 'valid':valid_dl}    
    model, log_loss, accuracy, modelname = train_model(dloaders, netname, num_epochs=25)    
    now =  datetime.datetime.now()

    dumpname = './model_dump/month_0' + str(now.month) +  str(now.day)  +  '_acc_' + str(round(accuracy,3)) + '_log_loss_' + str(round(log_loss,3)) +'_'+ modelname + '.pth'
    torch.save(model.state_dict(), dumpname)

    paths = ['/home/wenyue/Desktop/data_playground/kaggle_furniture/model_dump/month_0529_acc_0.0_log_loss_0.577_densenet201.pth']
    # path can have multiple paths to make it an ensemble model    
     
    model_names = ['densenet201']   
    output, prediction_list = predict_process(paths,model_names)

def faster():
    file_list = glob.glob("/home/wenyue/Desktop/data_playground/kaggle_furniture/prep/month*")

    for i in range(len(file_list)):
        if i == 0:
            result_ = np.genfromtxt(file_list[i], delimiter = ',')
        else:
            result_ = np.concatenate((result_, np.genfromtxt(file_list[i], delimiter = ',')) , axis = 1)

    result_list = []
    for i in range(result_.shape[0]):
        list_ = np.where(result_[i] > 1)[0].tolist()
        list_1 = [item%128 + 1 for item in list_]
        if len(list_1) > 0:
            result_1 = max(set(list_1), key=list_1.count)
        else:
        #result_1 = np.argmax(result_[i])%128 + 1
            result_1 = np.argmax(np.sum(result_[i].reshape(len(file_list),128), axis = 0))+1
        result_list.append(result_1)

    pic_list = glob.glob('/home/wenyue/Desktop/data_playground/kaggle_furniture/test/*')
    pic_num_list = sorted([int(pic_list[i].split('/')[-1].split('.')[0]) for i in range(len(pic_list))])

    output = pd.DataFrame(data = {'index': pic_num_list, 'prediction': result_list})

    final = pd.DataFrame(data = {'real_index': range(1, 12801)})
    final = final.merge(output, left_on = 'real_index', right_on = 'index', how = 'left').drop('index', axis = 1).fillna(128)
    final.columns = ['id','predicted']
    final.predicted = final.predicted.astype(int)

    final.to_csv('/home/wenyue/Desktop/data_playground/kaggle_furniture/output/test.csv', index = False)


if __name__ == '__main__':
    main()
 