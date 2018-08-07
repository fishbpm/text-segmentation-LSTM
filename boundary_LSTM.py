# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 19:53:47 2018

@author: Fish
"""

from __future__ import print_function
import numpy as np
import pickle as pkl

#from keras.datasets import mnist
from keras.layers import Input, Activation, Dense, Flatten, Dropout, LSTM#, PReLU
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.utils import np_utils

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

from utils import imshow_grid

import re
import gensim
import codecs
import en_core_web_sm
import pickle as pkl
from pathlib2 import Path

PATH = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/synthetic'
RESOURCES = 'C:/Users/Fish/Documents/GitHub/graphseg/source/res/'#stopwords.txt'
BATCH_SIZE = 128
NB_CLASSES = 10
EPOCHS = 20
TRAIN_SIZE = 13500
use_dropout = False
stop_words = []
CONTENT = ['ADJ', 'VERB', 'ADV', 'NOUN', 'PROPN', 'PRON', 'INTJ']#these are POStags - high level

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

class KerasBatchGenerator(object):

    def __init__(self, data, labels, num_steps, batch_size):#, classes, skip_step=5):
        self.data = data
        self.labels = labels
        self.num_steps = num_steps
        self.batch_size = batch_size
        #self.classes = classes #not needed as our labels are already 1-hot
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset back to zero
        self.current_idx = 0
        # skip_step is the number of sentences which will be skipped before the next
        # batch is skimmed from the data set
        #self.skip_step = skip_step
    
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.targets))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                y[i, :] = self.labels[self.current_idx:self.current_idx + self.num_steps]
                #self.current_idx += self.skip_step
            yield x, y

word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH+'/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
nlp = en_core_web_sm.load()

with open(RESOURCES+'stopwords.txt', 'r') as in_file:
    stop_words = in_file.read().splitlines()
    in_file.close()

#boundaries = np.zeros(10000, 300)
boundaries = []
manual_stop = 50
sample = 0
exceptions = []#not found in google news vectors
for f, file in enumerate(get_files(PATH+'/signal/topical/test')):
    if f < manual_stop:
        with codecs.open(file, 'r', 'utf-8') as article:
            sentences = article.read().splitlines()
            exceptions.append([])
            #sentences = article.split('\n') #should return the identical list
            #documents have already been sanitised and prepared with \n delimiters 
            tokens = []
            for s, line in enumerate(sentences):
                boundaries.append(np.zeros(300)) #what if we have no successful lookups for this sentgence??
                sentence = re.sub('[^a-zA-Z0-9\s,\.]+','',re.sub('-',' ',line)).strip()#.lower()#strip any non-alphanumerics
                #dont lowercase yet spaCy ent tagger can make use of the Caps
                tokens.append([])#initilaise this sentence
                exceptions[f].append([])#this is just for auditing purposes
                length = 0
                for token in nlp(sentence):
                    word = re.sub('\W','',token.text.lower())#apostrophes (like don't) have been stripped from freq resource
                    cleansed = re.sub('\d','#',re.sub("[^\w']",'',token.text.lower()))#retain commas as word2vec includes don't etc.
                    if len(cleansed) > 0 and not word.isnumeric() and word not in stop_words and token.lemma_ not in stop_words and (token.pos_ in CONTENT or token.ent_iob_ != 'O'):
                        try:
                            boundaries[sample] += word2vec[cleansed]
                        except:
                            exceptions[f][s].append(cleansed)
                sample += 1


train_data_generator = KerasBatchGenerator(X, y, num_steps, batch_size)#, vocabulary, skip_step=num_steps)
#valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size)#, vocabulary, skip_step=num_steps)

model = Sequential()
#model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

#we need to save the model







