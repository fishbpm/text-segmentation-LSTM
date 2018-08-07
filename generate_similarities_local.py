# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 12:42:31 2018

@author: Fish
"""
import os
import json
import re
import io
import codecs
import gensim
import math
import en_core_web_sm
from nltk.tokenize import RegexpTokenizer
import xlsxwriter as excel
import pickle as pkl
from pathlib2 import Path

#from elasticsearch import Elasticsearch, helpers

#def main():
PATH = 'C:/Users/Fish/Documents/GitHub/datasets'
RESOURCES = 'C:/Users/Fish/Documents/GitHub/graphseg/source/res/'#stopwords.txt'

categories = ['bulk']#['aggregate', 'geographic', 'domain', 'topical', 'reject', 'problem']
stop_words = []
word_freqs = []
sum_freqs = 0
vocabulary = 0
#CONTENT = ['VBN', 'VBD', 'VB', 'VBG', 'NN', 'NNP', 'NNS', 'ADJ', 'ADV'] - more granular spacy tags, requires much longer list
CONTENT = ['ADJ', 'VERB', 'ADV', 'NOUN', 'PROPN', 'PRON', 'INTJ']#these are POStags - high level
#es = Elasticsearch()

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

def extract_sentence_words(sentence, remove_stop_words = True):
    sentence_words = RegexpTokenizer.tokenize(sentence)
    if remove_stop_words:
        sentence_words = [w for w in sentence_words if w not in stop_words]
    return sentence_words

def get_word(word):#unclear whether this is a list of LEMMAS or not????
    w = 0
    while word != word_freqs[w][0] and w < vocabulary:
        w += 1
    #if w == vocabulary:
        #return 1
    return w

word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH+'/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
nlp = en_core_web_sm.load()
workbook = excel.Workbook('sim_output.xlsx')
worksheet = workbook.add_worksheet()

with open(RESOURCES+'stopwords.txt', 'r') as in_file:
    stop_words = in_file.read().splitlines()
    in_file.close()
    
with open(RESOURCES+'freqs.txt', 'r') as in_file:
    for line in in_file.read().splitlines():
        word, freq = line.split()
        sum_freqs += int(freq)
        vocabulary += 1
        word_freqs.append([word, int(freq)])
    word_freqs.append(['', 726]) #padding for unmatched words
    in_file.close()

similarities = []
exceptions = []
div_factor = vocabulary + sum_freqs
manual_fix = 48 #number files in target - to prevent codec error (hidden corrupted file at end of fodler)
for f, file in enumerate(get_files(PATH+'/signal/topical/test')):
    if f < manual_fix:
        similarities.append([])
        with codecs.open(file, 'r', 'utf-8') as article:
            sentences = article.read().splitlines()
            #sentences = article.split('\n') #should return the identical list
            #similarities have already been sanitised and prepared with \n delimiters 
            tokens = []
            lengths = []
            tokens.append([])#start a first empty dummy sentence (boundary condition)
            lengths.append(0)
            for s, line in enumerate(sentences):
                sentence = re.sub('[^a-zA-Z0-9\s,\.]+','',re.sub('-',' ',line)).strip()#.lower()#strip any non-alphanumerics
                #dont lowercase yet spaCy ent tagger can make use of the Caps
                similarity = 0
                tokens.append([])#initilaise this sentence
                lengths.append(0)
                #for word in extract_sentence_words(sentence): #spacy is also a tokenizer
                for token in nlp(sentence):
                    word = re.sub('\W','',token.text.lower())#apostrophes (like don't) have been stripped from freq resource
                    cleansed = re.sub('\d','#',re.sub("[^\w']",'',token.text.lower()))#retain commas as word2vec includes don't etc.
                    if len(cleansed) > 0 and not word.isnumeric() and word not in stop_words and token.lemma_ not in stop_words and (token.pos_ in CONTENT or token.ent_iob_ != 'O'):
                        try:
                            temp = word2vec[cleansed]
                        except:
                            exceptions.append(cleansed)
                        else:
                            if token.ent_iob_ != 'O':
                                factor = 3
                            else:
                                factor = 1
                            tokens[s+1].append([cleansed])
                            tokens[s+1][lengths[s+1]].append(-factor*math.log10((word_freqs[get_word(re.sub('\d','',word))][1] + 1)/div_factor))
                            #-log((word_freq + 1)/(vocab_size + corpus_size)
                            for prev_token in tokens[s]:#this will not function on the first pass
                                ic_factor = min(tokens[s+1][lengths[s+1]][1], prev_token[1])
                                try:
                                    similarity += (1 - ic_factor*(word2vec.similarity(tokens[s+1][lengths[s+1]][0], prev_token[0])))
                                except:
                                    print('sim fail:',tokens[s+1][lengths[s+1]][0],'/',prev_token[0])
                            lengths[s+1] += 1
                try:
                    similarities[f].append([similarity/(lengths[s]*lengths[s+1])])
                    #similarities[f].append([(similarity/lengths[s] + similarity/lengths[s+1])/2])
                except:
                    similarities[f].append([0])
                similarities[f][s].append(lengths[s+1])#this length will always be 0 on the first pass
    
for col, article in enumerate(similarities):
    for row, boundary in enumerate(article):
        worksheet.write(row, 2*col, boundary[0])
        worksheet.write(row, 2*col + 1, boundary[1])
workbook.close()

##Save dataset as pickle
#with open('GRAPHSEG_probs.pkl', 'wb') as f:
##with open('/output/LSTM_probs.pkl', 'wb') as f:#when rnuning from container
#    pkl.dump({ 'probs': similarities }, f, pkl.HIGHEST_PROTOCOL)#, 'labels': y_train }, f, pkl.HIGHEST_PROTOCOL)
#    f.close()

