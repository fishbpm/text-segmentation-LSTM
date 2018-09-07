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

path = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/topical/'

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

#def extract_articles(num, input_file):
#    file_count = 0
#    with codecs.open(input_file, 'r', 'utf-8') as json_file:
#        for line in json_file:
#            #if i < 10:
#            dataset = json.loads(line)
#    topic = dataset['name']
#    os.makedirs(path + '/' + topic)
#    samples = dataset['examples']
#    for sample in samples:
#        file_count += 1
#        articles.append(sample['data'])
#        article = json.loads(sample['data'])
#        with codecs.open(path + '/' + topic + '/' + str(file_count)  + '–' + article['original_article_id'], 'w', 'utf-8') as f: 
#            print(article['title'], '\n', file=f)
#            print(article['content'], file=f)
#            f.close()
#    return file_count, topic
   
for f, file in enumerate(get_files(path+'SOURCES/')):
    #file_count, topic = extract_articles(f, file)
    #if f > 30:
    articles = []
    file_count = 0
    with codecs.open(file, 'r', 'utf-8') as json_file:
        for line in json_file:
            #if i < 10:
            dataset = json.loads(line)
    topic = re.sub('\s+',' ',re.sub('[:;]',' ',dataset['name'])).strip()
    try:
        os.makedirs(path + '/' + topic)
    except:
        print(topic, 'folder already written')
    else:
        samples = dataset['examples']
        for sample in samples:
            file_count += 1
            articles.append(sample['data'])
            article = json.loads(sample['data'])
            #with codecs.open(path + '/' + topic + '/' + str(file_count).rjust(3, '0')  + '–' + re.sub('[^\w\d\s]','',article['title'][:40]), 'w', 'utf-8') as f: 
            with codecs.open(path+'/'+topic+'/'+str(file_count).rjust(3, '0')+'–'+re.sub('\s+',' ',re.sub('[^\w\d\s]','',article['title'][:40])), 'w', 'utf-8') as f: 
                print(article['title'], '\n', file=f)
                print(article['content'], file=f)
                f.close()
        print("printed", file_count, topic, "articles")
        with codecs.open(path+'OUTPUT/'+str(file_count).rjust(3, '0')+' '+topic+'.jsonl', 'w', 'utf-8') as out_file:
            #json.dump(files, out_file)
            for file in articles:
                out_file.write(json.dumps(file) + '\n') #json.dump(file, out_file)
            out_file.close()

#with codecs.open(path + '/' + topic + '/' + annotations', 'w', 'utf-8') as f: 
#    print('\n'.join(labels).strip('\n'), file=f)
#    f.close()

# id name kind createdate data description examples
    #(name is the topic)
    
# example (in list of examples)  ::  data  label
    
# data -JSON  ::  content  source-name    original_article_id   title