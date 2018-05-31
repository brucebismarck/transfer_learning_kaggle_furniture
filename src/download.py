# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:08:14 2018

@author: bruce
"""

import os, multiprocessing, urllib3
from PIL import Image
from io import BytesIO
import pandas as pd
#from itertools import repeat
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def DownloadImage(key_url):
  out_dir = "/home/wenyue/Desktop/data_playground/kaggle_furniture/test_1"
  current_dir = "/home/wenyue/Desktop/data_playground/kaggle_furniture/test"
  (key, url) = key_url
  
  filename = os.path.join(current_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skip downloading.' % filename)
    return 
    
  try:
    #print('Trying to get %s.' % url)
    http = urllib3.PoolManager()
    response = http.request('GET', url, timeout = 5)
    image_data = response.data
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    output = open(out_dir + 'error.log', 'a')
    output.write('Warning: Could not download image %s from %s' % (key, url) + '\n')
    output.close()
    return 

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s %s' % (key,url))
    output = open(out_dir + 'error.log', 'a')
    output.write('Warning: Failed to parse image %s %s' % (key,url)+ '\n')
    output.close()
    return 

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print('Warning: Failed to convert image %s to RGB' % key)
    output = open(out_dir + 'error.log', 'a')
    output.write('Warning: Failed to convert image %s to RGB' % key + '\n')
    output.close()
    return 

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    print('Warning: Failed to save image %s' % filename)
    output = open(out_dir + 'error.log', 'a')
    output.write('Warning: Failed to save image %s' % filename + '\n')
    output.close()
    return 

       
def multi_downloader(csv_file):
    data  = pd.read_csv(csv_file,index_col = 0)
    try:
        data_new = data.drop('label_id', 1)
    except ValueError:
        data_new = data
    
    key_url_list = [tuple(x) for x in data_new.values]
                    
                    
    pool = multiprocessing.Pool(40)
    length = len(key_url_list)
    
    with tqdm(total = length) as t :
        for _ in pool.imap_unordered(DownloadImage, key_url_list):
            t.update(1)
            
if __name__ == '__main__':
    multi_downloader("/home/wenyue/Desktop/data_playground/kaggle_furniture/test_data.csv")
 

data = pd.read_csv("/home/wenyue/Desktop/data_playground/kaggle_furniture/test_data.csv",index_col = 0)
exist_pic = os.listdir('/home/wenyue/Desktop/data_playground/kaggle_furniture/test')
num_list = [int(item.split('.')[0]) for item in exist_pic]

data = data[data['image_id'].isin(num_list) == False]      

key_url_list = [tuple(x) for x in data.values]

pool = multiprocessing.Pool(40)
length = len(key_url_list)
    
with tqdm(total = length) as t :
    for _ in pool.imap_unordered(DownloadImage, key_url_list):
        t.update(1)