import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import os
from pandas import DataFrame

import urllib


os.chdir("E:\\Data playground\\kaggle_furniture\\")

with open("train.json") as datafile1: #first check if it's a valid json file or not
    data1 = json.load(datafile1)
with open("test.json") as datafile2: #first check if it's a valid json file or not
    data2 = json.load(datafile2)
with open("validation.json") as datafile3: #first check if it's a valid json file or not
    data3 = json.load(datafile3)

datafile1.close()
datafile2.close()
datafile3.close()


# for training data
my_dic_data = data1
keys= my_dic_data.keys()
dict_you_want1={'my_items1':my_dic_data['annotations']for key in keys}
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
df=pd.DataFrame(dict_you_want1)
fd = pd.DataFrame(dict_you_want2)
df2=df['my_items1'].apply(pd.Series)
fd2=fd['my_items2'].apply(pd.Series)
train_data = pd.merge(df2, fd2, on='image_id', how='outer')

# for validation data
my_dic_data = data3
keys= my_dic_data.keys()
dict_you_want1={'my_items1':my_dic_data['annotations']for key in keys}
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
df=pd.DataFrame(dict_you_want1)
fd = pd.DataFrame(dict_you_want2)
df2=df['my_items1'].apply(pd.Series)
#print ("df2",df2.head())
fd2=fd['my_items2'].apply(pd.Series)
#print ("fd2",fd2.head())
validation_data = pd.merge(df2, fd2, on='image_id', how='outer')

# for test data
my_dic_data = data2
keys= my_dic_data.keys()
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
fd = pd.DataFrame(dict_you_want2)
test_data=fd['my_items2'].apply(pd.Series)

train_data['url'] = train_data['url'].apply(lambda x:str(x[0]))
test_data['url'] = test_data['url'].apply(lambda x:str(x[0]))
validation_data['url'] = validation_data['url'].apply(lambda x:str(x[0]))


train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')
validation_data.to_csv('validation_data.csv')