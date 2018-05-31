import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import os
from pandas import DataFrame
os.chdir("E:\\Data playground\\kaggle_furniture")
#%matplotlib inline 


#%%
with open("train.json") as datafile1: #first check if it's a valid json file or not
    data1 = json.load(datafile1)
with open("test.json") as datafile2: #first check if it's a valid json file or not
    data2 = json.load(datafile2)
with open("validation.json") as datafile3: #first check if it's a valid json file or not
    data3 = json.load(datafile3)

datafile1.close()
datafile2.close()
datafile3.close()

train = pd.DataFrame(data1)
train.head()
test = pd.DataFrame(data2)
test.head()
validation = pd.DataFrame(data3)
validation.head()

