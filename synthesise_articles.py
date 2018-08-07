# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 12:20:16 2018

@author: fishb
"""
import os
import json
import re
import io
import codecs
import random
from pathlib2 import Path

#from elasticsearch import Elasticsearch, helpers

PATH = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/topical_sources/OUTPUT/'#TEMP/'#
OUT = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/synthetic/'
SEPARATOR = "========,1,__"
NUM_ARTICLES = 1000 #number of articles we wish to synthesise

fileset = []
filenames = []

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

#def clean_articles(input_files):
#    files = []
#    file_count = 0
#    
#    for article in input_files:
#        text = re.sub(' +',' ',article['content'])   
#        lines = [l.strip() for l in text.strip().split("\n") if len(l.strip()) > 50 and l != "\n"]
#        new_lines = []
#        num_lines = 0
#        active = False
#        for line in lines:
#            sentences = [s.strip() for s in re.split("[\.!?:;…]+[\s]+…*", line.strip('.'))]
#            num_sents = 0
#            if active:
#                num_lines += 1
#                active = False
#            for sent in sentences:
#                if not sent.isnumeric() and re.search('\s',sent) and not re.fullmatch('[\(\[{].*[\)\]}]',sent):
#                    #if the sentence comprises only a single word or numeric then just discard it
#                    if len(sent) > 35:
#                        if active: #in which case this must also be the first sentence (num_sents must be 0)
#                            new_lines[num_lines] = new_lines[num_lines] + " " + sent + "."
#                            #print(num_lines, num_sents, new_lines[num_lines])
#                        else:
#                            new_lines.append(sent + ".")
#                        num_lines += 1
#                        num_sents += 1
#                        active = False
#                    else:
#                        if num_sents == 0:
#                            if active:
#                                new_lines[num_lines] = new_lines[num_lines] + " " + sent + "."
#                            else:
#                                new_lines.append(sent + ".")
#                                active = True
#                        else:
#                            new_lines[num_lines - 1] = new_lines[num_lines - 1] + " " + sent + "."
#        if num_lines > 1:
#            files.append(new_lines) #append as list, ready for use in assembly
#            file_count += 1
#        
#    print("loaded", file_count, "articles")# from", input_file)
#    
#    return files

#with io.open(input_file, 'r', encoding = 'utf8') as json_file:
for t, topic in enumerate(get_files(PATH)):
    with codecs.open(topic, 'r', 'utf-8') as json_file:
        for d, document in enumerate(json_file):
            #if i < 10:
            fileset.append([t, json.loads(document)])
        #filenames.append([topic, d])#this was just for auditing
            
#dataset = clean_articles(fileset)
#files = []
dataset = []
file_count = 0

for article in fileset:
    text = re.sub(' +',' ',json.loads(article[1])['content'])   
    lines = [l.strip() for l in text.strip().split("\n") if len(l.strip()) > 50 and l != "\n"]
    new_lines = []
    num_lines = 0
    active = False
    for line in lines:
        sentences = [s.strip() for s in re.split("[\.!?:;…]+[\s]+…*", line.strip('.'))]
        num_sents = 0
        if active:
            num_lines += 1
            active = False
        for sent in sentences:
            if not sent.isnumeric() and re.search('\s',sent) and not re.fullmatch('[\(\[{].*[\)\]}]',sent):
                #if the sentence comprises only a single word or numeric then just discard it
                if len(sent) > 35:
                    if active: #in which case this must also be the first sentence (num_sents must be 0)
                        new_lines[num_lines] = new_lines[num_lines] + " " + sent + "."
                        #print(num_lines, num_sents, new_lines[num_lines])
                    else:
                        new_lines.append(sent + ".")
                    num_lines += 1
                    num_sents += 1
                    active = False
                else:
                    if num_sents == 0:
                        if active:
                            new_lines[num_lines] = new_lines[num_lines] + " " + sent + "."
                        else:
                            new_lines.append(sent + ".")
                            active = True
                    else:
                        new_lines[num_lines - 1] = new_lines[num_lines - 1] + " " + sent + "."
    if num_lines > 1:
        dataset.append([article[0], new_lines]) #append as list, ready for use in assembly
        file_count += 1
    
print("loaded", file_count, "articles")# from", input_file)
    
num_docs = len(dataset) #NOTE cannot assume all articles were cleaned successfully
full_range = int(4 / 7 * NUM_ARTICLES)
articles = []

for article in range(NUM_ARTICLES):
    lower = 2
    upper = 11
    if article > full_range:
        lower = random.randint(lower, upper - 2)
        if lower < (upper - 2):
            upper = random.randint(lower + 2, upper)

    sources = []
    topics = []
    segment = 0
    sentences = [SEPARATOR + str(segment) + '.']
    num_segs = random.randint(2, 10)
    while segment < num_segs:
        source = random.randint(0, num_docs - 1)
        topic = dataset[source][0]
        seg_size = random.randint(lower, upper)
        if (source not in sources) and (topic not in topics) and not len(dataset[source][1]) < seg_size:#ensure we use each source once only
            sources.append(source)
            topics.append(topic)
            for sent in range(seg_size):
                sentences.append(dataset[source][1][sent])
            segment += 1
            if segment < num_segs:
                sentences.append(SEPARATOR + str(segment) + '.')
    result = '\n'.join(sentences).strip('\n')
#    with codecs.open(OUT + '/' + str(article).rjust(4, '0'), 'w', 'utf-8') as f: 
#        print(result, file=f)
#        f.close()
    articles.append(result)
    
with codecs.open('synth_set.jsonl', 'w', 'utf-8') as out_file:
    #json.dump(files, out_file)
    for article in articles:
        out_file.write(json.dumps(article) + '\n') #json.dump(file, out_file)
    out_file.close()   

