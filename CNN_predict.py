# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:08:25 2018

@author: Fish
"""

#import re
#import os
import random
import numpy as np
from PIL import Image
#import xlsxwriter as excel
import pickle as pkl
from text_normaliser import normalise
#from text_normaliser_ORIGINAL import normalise
from keras.layers import Input, MaxPooling1D, Conv1D, Dense, Flatten, Dropout#, PReLU
#from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.utils import np_utils, to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

PATH = 'C:/Users/Fish/docker/OUTPUTS/Testing/'
LABELS = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/TESTING/signal400/'
#HEIGHT = 17
#WIDTH = 1
NUM_FEATURES = 40
annotations = [] #labels are stored as TRUE (aggregate) and FALSE
CONTROL = 10000#338#

def rescale(article):
    if len(article) == NUM_FEATURES:
        return article
    else:
        imaged = Image.fromarray(np.asarray([article]))
        rescaled = np.asarray(imaged.resize((NUM_FEATURES, 1), Image.BICUBIC))
        return rescaled.tolist()[0]

"""***********************import original natural corpus labels of 400 articles***********************"""
with open(LABELS+'labels_JSON-no_reversions.txt', 'r') as in_file:
#with open(LABELS+'labels_FILE-no_reversions.txt', 'r') as in_file:
    for line in in_file.read().splitlines():
        annotation = line.split()[1] #labels stored in order in which they were put into the LSTM
        annotations.append(1 if annotation=='TRUE' else 0) #do we need to put an integer? Or could we just load as boolean/string?
    in_file.close()

true_count = len([a for a in annotations if a == 1])
#labels = annotations[:2*true_count]#:2*true_count] #balance the dataset - topical articles are all at the end of the dataset
labels = annotations#[:2*true_count] #balance the dataset - topical articles are all at the end of the dataset
print(true_count, 'summaries in', len(labels), 'dataset')

##-----------this is to test corruipting the labels
#for step in range(0, 25):
#    random.shuffle(labels)

"""***********************import original natural corpus LSTM softmax ***********************"""
THRESHOLD = 0.05 # all boundaries above this will be preserved on normalisation
MAX_PAD = 0.005
testset = []    
LSTM_output = pkl.load(open(PATH+'signal_1000K/1000K_softmax_signal400.pkl', 'rb'))
#LSTM_output = pkl.load(open(PATH+'wiki_700K/wiki700K_GPU_softmax_signal_400.pkl', 'rb'))
articles = LSTM_output['probs']
for a, article in enumerate(articles):
#for article, sentences in enumerate(scaled_articles):
    if a < len(labels):
        boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]#deos this need to be a numpy array?
        boundaries.append(a) #index needed to inspect results after shuffling
        #boundaries.append([random.randint(0, 1)]) #-----------this is to test corruipting the labels
        boundaries.append(labels[a]) #append label to facilitate in-place shuffle
        testset.append(boundaries)
#
###shuffling is only needed when TRAINING on the natural samples
##for step in range(0, 25):#Im unsure whwther keras shuffles on each cross-validation, so lets shuffle here
##    random.shuffle(dataset)
# 
X_test = np.asarray([boundaries[:-2] for boundaries in testset]) # excluding index and labels
y = [b[-1] for b in testset] #this is needed for sklearn metrics which cant handle 1-hot encoding
y_test = to_categorical(np.asarray(y), num_classes=2)
print('test features have shape -', X_test.shape)
print('test labels have shape -', y_test.shape)   

"""***********************import Graphsim test probabilities***********************"""
#THRESHOLD = 0.3 # all boundaries above this will be preserved on normalisation
#MAX_PAD = 0.05
#testset = []
#GRAPHSIM = 'C:/Users/Fish/outputs/graphsim/'
#LSTM_output = pkl.load(open(GRAPHSIM + 'GRAPHSIM_w6ent_FULL.pkl', 'rb')) 
##LSTM_output = pkl.load(open(PATH+'wiki_700K/wiki700K_GPU_softmax_signal_400.pkl', 'rb'))
#articles = LSTM_output['probs']
#for a, article in enumerate(articles):
##for article, sentences in enumerate(scaled_articles):
#    if a < len(labels): 
#        boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
#        boundaries.append(a) #index needed to inspect results after shuffling
#        #boundaries.append([random.randint(0, 1)]) #-----------this is to test corruipting the labels
#        boundaries.append(labels[a]) #append label to facilitate in-place shuffle
#        testset.append(boundaries)
#        
##shuffling is only needed when TRAINING on the natural samples
##for step in range(0, 25):#Im unsure whwther keras shuffles on each cross-validation, so lets shuffle here
##    random.shuffle(dataset)
# 
#X_test = np.asarray([boundaries[:-2] for boundaries in testset]) # excluding index and labels
#y = [b[-1] for b in testset] #this is needed for sklearn metrics which cant handle 1-hot encoding
#y_test = to_categorical(np.asarray(y), num_classes=2)
#print('test features have shape -', X_test.shape)
#print('test labels have shape -', y_test.shape)

"""***********************import synthetic and topical GRAPHSIM training set of 20000 articles***********************"""
#selected = [] #obtain random unique indexes, to ensure a randomised training set
#while len(selected) < (CONTROL + true_count):#CONTROL:#additional samples are needed to replace topical component of test set
#    index = random.randint(0, 2152)#30870)#9999)#synth_top contains 10,000, final_top contains 2153, topical pools 30,871
#    i = 0
#    while i < len(selected) and index != selected[i]:
#        i += 1
#    if i == len(selected):
#        selected.append(index)
#        
#THRESHOLD = 0.3 # all boundaries above this will be preserved on normalisation
#MAX_PAD = 0.05
#dataset = []
#surplus = []
##files = ['GRAPHSIM_w6ent_TOPICAL.pkl', 'GRAPHSIM_w6ent_FINAL.pkl']
#files = ['GRAPHSIM_w6ent_FINAL_topical.pkl', 'GRAPHSIM_w6ent_FINAL.pkl']
#GRAPHSIM = 'C:/Users/Fish/outputs/graphsim/'
#for f, file in enumerate(files):
#    LSTM_output = pkl.load(open(GRAPHSIM + file, 'rb'))
#    #LSTM_output = pkl.load(open(PATH+'wiki_700K/wiki700K_GPU_softmax_signal_400.pkl', 'rb'))
#    articles = LSTM_output['probs']
#    for a, article in enumerate(articles):
##        boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
##        boundaries.append(a) #index needed to inspect results after shuffling
##        #boundaries.append([random.randint(0, 1)]) #-----------this is to test corruipting the labels
##        boundaries.append(f) #append label to facilitate in-place shuffle
##        dataset.append(boundaries)
#        if f == 0:
#            if a in selected[:CONTROL]:
#                boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]#deos this need to be a numpy array?
#                boundaries.append(a) #index needed to inspect results after shuffling
#                boundaries.append(f) #append label to facilitate in-place shuffle - these are all aggregates
#                dataset.append(boundaries)
#            if a in selected[CONTROL:]:
#                boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]#deos this need to be a numpy array?
#                boundaries.append(a) #index needed to inspect results after shuffling
#                boundaries.append(f) #append label to facilitate in-place shuffle - these are all aggregates
#                surplus.append(boundaries)
#        else:
#            #if a in selected[:CONTROL]:
#            boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]#deos this need to be a numpy array?
#            boundaries.append(a) #index needed to inspect results after shuffling
#            boundaries.append(f) #append label to facilitate in-place shuffle - these are all aggregates
#            dataset.append(boundaries) 
#
#for step in range(0, 25):#Im unsure whwther keras shuffles on each cross-validation, so lets shuffle here
#    random.shuffle(dataset)
#    
#X_train = np.asarray([boundaries[:-2] for boundaries in dataset]) # excluding index and labels
#y_tr = np.asarray([b[-1] for b in dataset])
#y_train = to_categorical(y_tr, num_classes=2)
#print('training features have shape -', X_train.shape)
#print('training labels have shape -', y_train.shape)

"""***********************import synthetic and topical training set of 20000 articles***********************"""       
selected = [] #obtain random unique indexes, to ensure a randomised training set
while len(selected) < (CONTROL + true_count):#CONTROL:#additional samples are needed to replace topical component of test set
    index = random.randint(0, 11000)#9999)#30870)#2152)#synth_top contains 10,000, final_top contains 2153, topical pools 30,871
    i = 0
    while i < len(selected) and index != selected[i]:
        i += 1
    if i == len(selected):
        selected.append(index)

THRESHOLD = 0.05 # all boundaries above this will be preserved on normalisation
MAX_PAD = 0.005
dataset = []
surplus = []
#files = ['wiki700K_softmax_topical_10K.pkl', 'wiki700K_GPU_signal_FINAL.pkl']
#files = ['1000K_softmax_topical_10K.pkl', '1000K_softmax_signalFINAL.pkl']
#files = ['1000K_softmax_FINAL_topical.pkl', '1000K_softmax_signalFINAL.pkl']
files = ['1000K_softmax_topical_30K.pkl', '1000K_softmax_synth_100K.pkl']
for f, file in enumerate(files):
    #LSTM_output = pkl.load(open(PATH+'wiki_700K/'+file, 'rb'))
    LSTM_output = pkl.load(open(PATH+'signal_1000K/'+file, 'rb'))
    #LSTM_output = pkl.load(open(PATH+'signal_100K/'+file, 'rb'))
    articles = LSTM_output['probs']
    for a, article in enumerate(articles):#[:375]):#30000]):
#        boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]#deos this need to be a numpy array?
#        boundaries.append(a) #index needed to inspect results after shuffling
#        boundaries.append(f) #append label to facilitate in-place shuffle - these are all aggregates
#        dataset.append(boundaries)
        if f == 0:
            if a in selected[:CONTROL]:
                boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]#deos this need to be a numpy array?
                boundaries.append(a) #index needed to inspect results after shuffling
                boundaries.append(f) #append label to facilitate in-place shuffle - these are all aggregates
                dataset.append(boundaries)
            if a in selected[CONTROL:]:
                boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]#deos this need to be a numpy array?
                boundaries.append(a) #index needed to inspect results after shuffling
                boundaries.append(f) #append label to facilitate in-place shuffle - these are all aggregates
                surplus.append(boundaries)
        else:
            if a in selected[:CONTROL]:
                boundaries = [[abs(probability)] for probability in rescale(article)]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]#deos this need to be a numpy array?
                boundaries.append(a) #index needed to inspect results after shuffling
                boundaries.append(f) #append label to facilitate in-place shuffle - these are all aggregates
                dataset.append(boundaries)            
                 
for step in range(0, 25):#Im unsure whwther keras shuffles on each cross-validation, so lets shuffle here
    random.shuffle(dataset)
 
X_train = np.asarray([boundaries[:-2] for boundaries in dataset]) # excluding index and labels
y_tr = np.asarray([b[-1] for b in dataset])
y_train = to_categorical(y_tr, num_classes=2)
print('training features have shape -', X_train.shape)
print('training labels have shape -', y_train.shape)

"""*****************************import a supplementary set from pickle (already shuffled)************************************""" 
#temp = pkl.load(open(PATH+'signal_1000K/1000K_signal_60K_scaled.pkl', 'rb'))
##temp = pkl.load(open(PATH+'wiki_700K/wiki700K_signal_20K_scaled.pkl', 'rb'))
##temp = pkl.load(open(GRAPHSIM+'graphsim_w6ent_scaled.pkl', 'rb'))

"""*****************************import a training set from pickle************************************""" 
##trainset = pkl.load(open(PATH+'wiki_700K/wiki700K_signal_FINAL_scaled.pkl', 'rb'))
##trainset = pkl.load(open(PATH+'signal_100K/100K_signal_20K_scaled.pkl', 'rb'))
#trainset = pkl.load(open(PATH+'signal_1000K/1000K_signal_FINAL_scaled.pkl', 'rb'))
##trainset = pkl.load(open(GRAPHSIM+'graphsim_w6ent_FINAL_scaled.pkl', 'rb'))
##trainset = pkl.load(open(PATH+'signal_1000K/1000K_signal_20K_normalised.pkl', 'rb'))
#X_train = trainset['features']#[:8000]#[:-2000]#[2000:]#
#y_train = trainset['labels']#[:8000]#[:-2000]#[2000:]#
##X_test = trainset['features'][-2000:]#[:2000]#
##y_test = trainset['labels'][-2000:]#[:2000]#
##y = np.zeros(len(y_test), dtype=np.int)
##for sample, probs in enumerate(y_test):
##    y[sample] = np.where(probs == max(probs))[0][0]

"""*************replace topical component of test sampling from signal corpus of 10000 articles***********************"""       
testset.sort(key = lambda x: x[-1], reverse=True) #move summary samples to top of the data set

X_list = [boundaries[:-2] for boundaries in testset[:true_count]]#get the summary half of the dataset
y = [b[-1] for b in testset[:true_count]] #this is needed for sklearn metrics which cant handle 1-hot encoding

#test samples MUST be drawn from those NOT used for training the CNN
#count = 0
#for f, l in zip(temp['features'][-2000:], temp['labels'][-2000:]):#:2000], trainset['labels'][:2000]):#
#    if l[0] == 1 and count < true_count: #one-hot encoded - topical is in position 0
#        X_list.append(f)
#        y.append(0)
#        count += 1
        
for s in surplus:#:2000], trainset['labels'][:2000]):#
    X_list.append(s[:-2])
    y.append(0)
        
X_test = np.asarray(X_list) # excluding index and labels
y_test = to_categorical(np.asarray(y), num_classes=2)
print('test features have shape -', X_test.shape)
print('test labels have shape -', y_test.shape)

"""*************merge and shuffle to maximise training size***********************"""
X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_tr, y), axis=0)
dataset = np.asarray([(x, y) for x, y in zip(X_all, y_all)])
#dataset = np.concatenate((X_all, y_all), axis=1)
for step in range(0, 10):
    np.random.shuffle(dataset)

X = np.asarray([sample[0] for sample in dataset])
target = np.asarray([sample[1] for sample in dataset])

"""**************************compile model***********************************"""
inputs = Input(shape=(40, 1))
#x = Dense(64, activation='relu')(inputs)
x = Conv1D(24, kernel_size=5, activation='relu')(inputs)
#x = PReLU()(x) # Non-linearity
x = MaxPooling1D(pool_size=2, strides=1)(x)
x = Dropout(rate=0.2)(x)
x = Conv1D(36, kernel_size=5, activation='relu')(inputs)
#x = PReLU()(x) # Non-linearity
x = MaxPooling1D(pool_size=2, strides=1)(x)
x = Dropout(rate=0.2)(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(100, activation='relu')(x)
#features = Dropout(rate=0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',#adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.save_weights('model.h5')

"""**************************single training run***********************************"""
true_count = len([a for a in y if a == 1])
print(true_count, 'summaries in', len(y), 'dataset')

BATCH_SIZE = 128#49#45#
EPOCHS = 5#20#100#

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    verbose=0, validation_data=(X_test, y_test))#validation_split = 0.2, shuffle=True)#
#score = model.evaluate(mnist_test, mnist.test.labels, verbose=0)

#model.evaluate(X_test, y_test)
#
results = [0, 0]
y_pred = model.predict(X_test)
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y_pred):
    y_class[sample] = np.where(probs == max(probs))[0][0]
    if y_class[sample] == y[sample]: #if this is a correct prediction
        results[y[sample]] += 1 #increment result count for this class

print('accuracy', round(accuracy_score(y, y_class),4))
print('precision', round(precision_score(y, y_class),4))
print('recall', round(recall_score(y, y_class),4))
print('F1-score', round(f1_score(y, y_class),4))
print(round(results[0]/(len(y) - true_count), 4), 'topicals correctly classified')
print(round(results[1]/true_count, 4), 'summaries correctly classified')

"""**************************multiple folds (new model each time)***********************************"""
#indices = []
#folds = 5
#skf = StratifiedKFold(n_splits=folds, shuffle=True)
##ensure we are using the same folds for every experiment
##for train, test in skf.split(X, target):
##    indices.append([train, test])
#
#score = [[],[],[],[],[],[]]
#for train, test in skf.split(X, target):
#    X_train = X[train]# excluding index and labels
#    y_train = to_categorical(target[train], num_classes=2)
#    print('train features have shape -', X_train.shape)
#    print('train labels have shape -', y_train.shape)
#    
#    X_test = X[test] # excluding index and labels
#    y = target[test]
#    y_test = to_categorical(y, num_classes=2)
#
#    print('test features have shape -', X_test.shape)
#    print('test labels have shape -', y_test.shape)
#    
#    """******************************AUDIT labels******************************************"""
#    true_count = len([a for a in y if a == 1])
#    print(true_count, 'summaries in', len(y), 'dataset')
#    
#    BATCH_SIZE = 128#49#45#
#    EPOCHS = 20#5#100#
#    
#    model.load_weights('model.h5')  
#    
#    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
#                        verbose=0, validation_data=(X_test, y_test))#validation_split = 0.2, shuffle=True)#
#    #score = model.evaluate(mnist_test, mnist.test.labels, verbose=0)
#    
#    #model.evaluate(X_test, y_test)
#    #
#    results = [0, 0]
#    y_pred = model.predict(X_test)
#    y_class = np.zeros(len(y), dtype=np.int)
#    for sample, probs in enumerate(y_pred):
#        y_class[sample] = np.where(probs == max(probs))[0][0]
#        if y_class[sample] == y[sample]: #if this is a correct prediction
#            results[y[sample]] += 1 #increment result count for this class
#    
#    score[0].append(accuracy_score(y, y_class))
#    score[1].append(precision_score(y, y_class))
#    score[2].append(recall_score(y, y_class))
#    score[3].append(f1_score(y, y_class))
#    score[4].append(results[0]/(len(y) - true_count))
#    score[5].append(results[1]/true_count)
#    
#    print('accuracy', round(accuracy_score(y, y_class),4))
#    print('precision', round(precision_score(y, y_class),4))
#    print('recall', round(recall_score(y, y_class),4))
#    print('F1-score', round(f1_score(y, y_class),4))
#    print(round(results[0]/(len(y) - true_count), 4), 'topicals correctly classified')
#    print(round(results[1]/true_count, 4), 'summaries correctly classified')
#    
#    #del model
#    
#    #with open('softmax_normalised.pkl', 'wb') as f:
#    #    pkl.dump({ 'features': X_train, 'labels': y_train }, f, pkl.HIGHEST_PROTOCOL)
#
#print('Avg CV Acc:', round(np.mean(score[0]),4), end='')
#print(' precsn', round(np.mean(score[1]),4), end='')
#print(' recall', round(np.mean(score[2]),4), end='')
#print(' F1', round(np.mean(score[3]),4), end='')
#print(' topical', round(np.mean(score[4]),4), end='')
#print(' summary', round(np.mean(score[5]),4))














