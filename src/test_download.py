#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:12:28 2018

@author: wenyue
"""



import os, multiprocessing, urllib3
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm

os.chdir('/home/wenyue/Desktop/data_playground/kaggle_furniture')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



def DownloadImage(key_url):
  out_dir = "/home/wenyue/Desktop/data_playground/kaggle_furniture/test/"
  (key, url) = key_url
  
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return 
    
  try:
    #print('Trying to get %s.' % url)
    http = urllib3.PoolManager()
    response = http.request('GET', url, timeout = 5)
    image_data = response.data
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    output = open(out_dir + 'error.log', 'a')
    output.write('Warning: Could not download image %s from %s' % (key, url))
    output.close()
    return 

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s %s' % (key,url))
    output = open(out_dir + 'error.log', 'a')
    output.write('Warning: Failed to parse image %s %s' % (key,url))
    output.close()
    return 

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print('Warning: Failed to convert image %s to RGB' % key)
    output = open(out_dir + 'error.log', 'a')
    output.write('Warning: Failed to convert image %s to RGB' % key)
    output.close()
    return 

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    print('Warning: Failed to save image %s' % filename)
    output = open(out_dir + 'error.log', 'a')
    output.write('Warning: Failed to save image %s' % filename)
    output.close()
    return 


os.getcwd()
#%%
test_json = json.load(open('test.json'))
#%%
data = test_json['images']
key_url_list = []
#%%
for i in range(len(data)):
    key_url = (data[i]['image_id'],data[i]['url'][0])
    key_url_list.append(key_url)
                 
pool = multiprocessing.Pool(20)
length = len(key_url_list)
#%%   
with tqdm(total = length) as t :
    for _ in pool.imap_unordered(DownloadImage, key_url_list):
        t.update(1)

'''
if __name__ == '__main__':
    multi_downloader("test.json")
'''        
        


        
        
