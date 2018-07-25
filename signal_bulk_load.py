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

#from elasticsearch import Elasticsearch, helpers

path = 'C:/Users/Fish/Documents/GitHub/datasets/signalmedia-1m.jsonl/'
out_path = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/signal_raw'#'/mnt/c/Users/Fish/Documents/GitHub/datasets/wiki_727'
input_file = path + 'sample-1M.jsonl'
#input_file = out_path + '/articles.jsonl'
#output_file= path + 'test.json'
categories = ['aggregate', 'geographic', 'domain', 'topical', 'reject', 'problem']#['bulk']

#es = Elasticsearch()

dataset = []

#with io.open(input_file, 'r', encoding = 'utf8') as json_file:
with codecs.open(input_file, 'r', 'utf-8') as json_file:
    for i, line in enumerate(json_file):
        #if i < 10:
        dataset.append(json.loads(line))

def load_file(title):   
    file_count = 0
    files = []
    indexes = []
    #title = 'domain'
    with open('kibana_'+title+'.txt', 'r') as in_file:
        indexes = in_file.read().splitlines()
        in_file.close()
    
    for article in dataset:
        i = 0
        while i < len(indexes) and article['id'] != indexes[i]:
            i += 1
        if i < len(indexes):
            #if indexes[i] == 'c18ea744-0754-4a7c-8fe6-46766e6250d9':#== 9:#        
            #if article['id'] in indexes:
            #raw = article['content']
            #print(article['id'])
            text = re.sub(' +',' ',article['content'])   
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
                files.append(article)
                file_count += 1
                #with io.open(out_path + '/' + str(i).rjust(2, '0') + '_' + article['id'], 'w', encoding='utf8') as f:
                #with codecs.open(out_path + '/' + str(i).rjust(2, '0') + '_' + article['id'], 'w', 'utf-8') as f:
                with codecs.open(out_path + '/' + title + '/' + article['id'], 'w', 'utf-8') as f: 
                    print(article['title'], '\n', file=f)
                    print(article['content'], file=f)
                    
                    #print('\n'.join(new_lines).strip('\n'), file=f)
                    
                    #f.write('\n'.join(new_lines).strip('\n'))
                    #text = re.sub('^.{1,70}\\n','', article['content'])
                    #print(re.sub("((\s*)\\n)+", "\\n", text), file=f)
                    #json.dump(article['content'], f)
                    f.close()
        
    print("loaded ", file_count, " articles of the ", len(indexes), "retrieved in Kibana")
    with codecs.open(out_path + '/' + title + '/' + title + '.jsonl', 'w', 'utf-8') as out_file:
        #json.dump(files, out_file)
        for file in files:
            out_file.write(json.dumps(file) + '\n') #json.dump(file, out_file)
        out_file.close()
        #for i in range(0,40):
        #	es.index(index = "signalbool", doc_type='doc', id=dataset[i]['id'], body=dataset[i])
    
    #for record in dataset[40:]:
    	#es.index(index = "boolsignal", doc_type='doc', id=record['id'], body=record)
    
    
    #path = 'C:/Users/Fish/signalmedia-1m.jsonl/'
    #input_file=open(path + 'sample-1M.jsonl', 'r')
    #output_file=open(path + 'test.json', 'w')
    #json_decode=json.load(input_file)
    #for item in json_decode:
    #    my_dict={}
    #    my_dict['title']=item.get('labels').get('en').get('value')
    #    my_dict['description']=item.get('descriptions').get('en').get('value')
    #    my_dict['id']=item.get('id')
    #    print my_dict
    #back_json=json.dumps(my_dict, output_file)
    #output_file.write(back_json)
    #output_file.close()

for category in categories:
    load_file(category)
            
