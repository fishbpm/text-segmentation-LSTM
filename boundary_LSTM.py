# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 19:53:47 2018

@author: Fish
"""

from __future__ import print_function
import numpy as np
import pickle as pkl

#from keras.datasets import mnist
from keras.layers import Input, Activation, Dense, Flatten, Dropout, LSTM, TimeDistributed#, PReLU
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.utils import np_utils

from sklearn.metrics import log_loss, accuracy_score

from utils import imshow_grid
import wiki_utils

import re
import gensim
import codecs
import en_core_web_sm
import pickle as pkl
from pathlib2 import Path

PATH = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/synthetic'
RESOURCES = 'C:/Users/Fish/Documents/GitHub/graphseg/source/res/'#stopwords.txt'
VECTORS = 'C:/Users/Fish/big_datasets/cphrase/cphrase.txt'
NB_CLASSES = 10
stop_words = []
CONTENT = ['ADJ', 'VERB', 'ADV', 'NOUN', 'PROPN', 'PRON', 'INTJ']#these are POStags - high level

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

def load_vectors(path):
    data = [] 
    in_file = open(path, 'r', encoding='utf8')
    for l in in_file.read().split('\n'):
        data.append(l.split('\t'))
    datadict = {}
    for item in data:
        datadict[item[0]] = item[1:]
    return datadict

class KerasBatchGenerator(object):

    def __init__(self, data, labels, num_steps, batch_size):#, classes)#, skip_step=5):
        self.data = data
        self.labels = labels
        self.num_steps = num_steps
        self.batch_size = batch_size
        #self.classes = classes #our single class is segmentation point?[0,1]
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset back to zero
        self.current_idx = 0
        # skip_step is the number of sentences which will be skipped before the next
        # batch is skimmed from the data set
        #self.skip_step = skip_step
    
    def generate(self):
        X = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps))#, self.classes))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                X[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                y[i, :] = self.labels[self.current_idx:self.current_idx + self.num_steps]
                #self.current_idx += self.skip_step
            yield X, y

def load_data(file_source):
    #word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH+'/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    cphrase = load_vectors(VECTORS)
    nlp = en_core_web_sm.load()
    
    with open(RESOURCES+'stopwords.txt', 'r') as in_file:
        stop_words = in_file.read().splitlines()
        in_file.close()
    
    #boundaries = np.zeros(10000, 300)
    boundaries = []
    labels = []
    manual_stop = 50
    sample = 0
    exceptions = []#not found in google news vectors
    separator = wiki_utils.get_seperator_foramt()
    
    for f, file in enumerate(get_files(file_source)):
        if f < manual_stop:
            with codecs.open(file, 'r', 'utf-8') as article:
                segments = [s.strip('\n') for s in re.split(separator, article.read())]
                #segments = [s for s in segments if len(s) > 0] - we dont need this as synthetic segments are guarnateed to be >=2
                new_sent = True #force initialisation of first sentence container
                for segment in segments:
                    sentences = segment.split('\n') #documents have already been sanitised and prepared with \n delimiters
                    #sentences = segment.splitlines()
                    exceptions.append([])#this is just for auditing purposes
                    #sentences = article.split('\n') #should return the identical list
                    for s, line in enumerate(sentences):
                        if new_sent: #do not append new sent unless we successfully built a sentence on the last pass
                            boundaries.append(np.zeros(300))
                            labels.append(0) #default as a negative label
                            new_sent = False
                        else:
                            sample -= 1
                        sentence = re.sub('[^a-zA-Z0-9s,.]+','',re.sub('-',' ',line)).strip()#.lower()#strip any non-alphanumerics
                        #dont lowercase yet spaCy ent tagger can make use of the Caps
                        exceptions[f].append([])#this is just for auditing purposes
                        for token in nlp(sentence):
                            word = re.sub('W','',token.text.lower())#apostrophes (like don't) have been stripped from freq resource
                            cleansed = re.sub('d','#',re.sub("[^w']",'',token.text.lower()))#retain commas as word2vec includes don't etc.
                            if len(cleansed) > 0 and not word.isnumeric() and word not in stop_words and token.lemma_ not in stop_words and (token.pos_ in CONTENT or token.ent_iob_ != 'O'):
                                try:
                                    boundaries[sample] += cphrase[cleansed]#word2vec[cleansed]
                                except:
                                    exceptions[f][s].append(cleansed)
                                else:
                                    new_sent = True #successfully embeddeded at least one sentence for this segment
                        sample += 1
                    if new_sent:
                        labels[-1] = 1 #set positive label for last sentence in this segment
    return np.asarray(boundaries), np.asarray(labels)


X, y = load_data(PATH+'/signal/topical/test') #check code - convert list to a numpy array
BATCH_SIZE = 8 #number articles per iteration
NUM_STEPS = 30 #average approximate #sents per article
EPOCHS = 20
HIDDEN_SIZE = 128 #as per pytorch model
use_dropout = False

#train_data_generator = KerasBatchGenerator(X, y, num_steps, batch_size, vocabulary, skip_step=num_steps)
train_data_generator = KerasBatchGenerator(X, y, NUM_STEPS, BATCH_SIZE)#, 1)
#valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size)#, vocabulary, skip_step=num_steps)

model = Sequential()
#model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

model.fit_generator(train_data_generator.generate(), len(X)//(BATCH_SIZE*NUM_STEPS), EPOCHS,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

#we need to save the model







