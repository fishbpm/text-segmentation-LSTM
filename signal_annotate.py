# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 16:08:39 2018

@author: Fish
"""
import os
import json
import re
import io
import codecs
from pathlib2 import Path

path = 'C:/Users/Fish/Documents/GitHub/datasets/signal/annotated/summary_articles/'
labels = [] 

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

def extract_labels(num, input_file):
    with codecs.open(input_file, 'r', 'utf-8') as json_file:
        for line in json_file:
            #if i < 10:
            dataset = json.loads(line)
    articles = dataset['data']['concept']['examples']
    for article in articles:
        try:
            labels.append(str(num)  + '|' + article['label'] + '|' + json.loads(article['data'])['id'])
        except:
            labels.append(str(num)  + '|' + article['label'] + '|' + json.loads(article['data'])['original_article_id'])
    return
   
for f, file in enumerate(get_files(path)):
    extract_labels(f, file)

with codecs.open(path + 'annotations', 'w', 'utf-8') as f: 
    print('\n'.join(labels).strip('\n'), file=f)
    f.close()

# id name kind createdate data description examples
    
# example (in list of examples)  ::  data  label
    
# data -JSON  ::  content  source-name    original_article_id   title