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
#CONTENT = ['ADJ', 'VERB', 'ADV', 'NOUN', 'PROPN', 'PRON', 'INTJ']#these are POStags - high level
CONTENT = ['NOUN', 'PROPN'] #need to re-set this to the line above (this is for testing ideas)
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

exceptions = []
boundaries = []
WINDOW = 3
div_factor = vocabulary + sum_freqs
manual_fix = 48 #number files in target - to prevent codec error (hidden corrupted file at end of fodler)
for f, file in enumerate(get_files(PATH+'/signal/topical/test')):
    if f < manual_fix:
        similarities = []
        with codecs.open(file, 'r', 'utf-8') as article:
            sentences = article.read().splitlines()
            #sentences = article.split('\n') #should return the identical list
            #similarities have already been sanitised and prepared with \n delimiters 
            tokens = []
            exceptions = []#not found in google news vectors
            boundaries.append([])
            for s, line in enumerate(sentences):
                sentence = re.sub('[^a-zA-Z0-9\s,\.]+','',re.sub('-',' ',line)).strip()#.lower()#strip any non-alphanumerics
                #dont lowercase yet spaCy ent tagger can make use of the Caps
                tokens.append([])#initilaise this sentence
                exceptions.append([])
                length = 0
                boundaries[f].append([0, 0]) #similarity & number contributing pairs
                #for word in extract_sentence_words(sentence): #spacy is also a tokenizer
                for token in nlp(sentence):
                    word = re.sub('\W','',token.text.lower())#apostrophes (like don't) have been stripped from freq resource
                    cleansed = re.sub('\d','#',re.sub("[^\w']",'',token.text.lower()))#retain commas as word2vec includes don't etc.
                    if len(cleansed) > 0 and not word.isnumeric() and word not in stop_words and token.lemma_ not in stop_words and (token.pos_ in CONTENT or token.ent_iob_ != 'O'):
                        if token.ent_iob_ != 'O':
                            factor = 3
                        else:
                            factor = 1
                        try:
                            temp = word2vec[cleansed]
                        except:
                            #exceptions.append(cleansed)
                            exceptions[s].append(cleansed)
                        #else:
                        tokens[s].append([cleansed])
                        tokens[s][length].append(-factor*math.log10((word_freqs[get_word(re.sub('\d','',word))][1] + 1)/div_factor))
                        #-log((word_freq + 1)/(vocab_size + corpus_size)
                        length += 1
                        
            for s1, sent_1 in enumerate(tokens):
                similarities.append([])
                for sent_2 in tokens[s1+1:s1+WINDOW]:
                    similarity = 0
                    num_pairs = 0
                    #similarities[s1].append([]) #initilaise columns
                    for token_1 in sent_1:
                        for token_2 in sent_2:#this will not function on the first pass
                            num_pairs += 1
                            ic_factor = min(token_1[1], token_2[1])
                            if token_1[0] == token_2[0]:
                                similarity += 2*(1 - ic_factor) #similarity is 1 - ic_factor will always be > 2 ("the"=2.018579)
                            else:
                                try:
                                    similarity += (1 - ic_factor*(word2vec.similarity(token_1[0], token_2[0])))
                                except:
                                    similarity += (1 - ic_factor/10) #for entities, this will contribute a dis-sim of 1.5
                                    #num_pairs -= 1
                                    #all un-paired exceptions will arrive here - so message is suppressed
                                    #print('sim fail:',token_1[0] ,'/', token_2[0])
                    try:
                        similarities[s1].append(similarity/num_pairs)#control vars are base zero
                        #similarities[f].append([(similarity/lengths[s] + similarity/lengths[s+1])/2])
                    except:
                        similarities[s1].append(0)         

            position = 2 #the END of the window
            while not position > (len(tokens) + WINDOW - 2):
                for boundary in range(max(0, position-WINDOW), min(len(tokens),position)):
                    for sent_1 in range(max(0, position-WINDOW), boundary):
                        for sent_2 in range(boundary, min(len(tokens),position)):
                            boundaries[f][boundary][0] += similarities[sent_1][sent_2 - sent_1 - 1]
                            boundaries[f][boundary][1] += 1
                position += 1

for col, article in enumerate(boundaries):
    for row, boundary in enumerate(article[1:]): #the first boundary is 0
        worksheet.write(row, col, max(0, boundary[0]/boundary[1]))
workbook.close()

##Save dataset as pickle
#with open('GRAPHSEG_probs.pkl', 'wb') as f:
##with open('/output/LSTM_probs.pkl', 'wb') as f:#when rnuning from container
#    pkl.dump({ 'probs': boundaries }, f, pkl.HIGHEST_PROTOCOL)#, 'labels': y_train }, f, pkl.HIGHEST_PROTOCOL)
#    f.close()

