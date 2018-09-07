# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:46:38 2018

@author: Fish
"""

import os
import json
import re
import io
import codecs
import math
import random
from pathlib2 import Path
import pickle as pkl
import gensim
import en_core_web_sm
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from text_normaliser import normalise

DATASETS = 'C:/Users/Fish/Documents/GitHub/datasets/'
SEPARATOR = "=========="
segment_seperator = "========"
RESOURCES = 'C:/Users/Fish/Documents/GitHub/graphseg/source/res/'#stopwords.txt'
CONTENT = ['ADJ', 'VERB', 'ADV', 'NOUN', 'PROPN', 'PRON', 'INTJ']#these are POStags - high level
#CONTENT = ['NOUN', 'PROPN'] #need to re-set this to the line above (this is for testing ideas)
NUM_FEATURES = 40
THRESHOLD = 0.05 # all boundaries above this will be preserved on normalisation
SEG_THRESHOLD = 0.4
MAX_PAD = 0.005   

CONTROL = 1200#338#num signal20K set being processed (total will be double)

samples = []
filenames = []
dataset = []

#!!!!this should be callable functions in a util module
with open(RESOURCES+'stopwords.txt', 'r') as in_file:
    stop_words = in_file.read().splitlines()
    in_file.close()

"""***********************import original natural corpus of 400 articles***********************"""
annotations = []
LABELS = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/TESTING/signal400/'
with open(LABELS+'labels_JSON-no_reversions.txt', 'r') as in_file:
#with open(LABELS+'labels_FILE.txt', 'r') as in_file:
    for line in in_file.read().splitlines():
        annotation = line.split()[1] #labels stored in order in which they were put into the LSTM
        annotations.append(1 if annotation=='TRUE' else 0) #do we need to put an integer? Or could we just load as boolean/string?
    in_file.close()

#targets = [random.randint(0, 1) for t in range(0, 2*CONTROL)]

true_count = len([a for a in annotations if a == 1])
targets = annotations[:2*true_count] #balance the dataset - topical articles are all at the end of the dataset
print(true_count, 'summaries in', len(targets), 'dataset')

temp = [] #temporary containers while we replace topical articles
#with io.open(input_file, 'r', encoding ls = 'utf8') as json_file:
with codecs.open(LABELS+'signal400.jsonl', 'r', 'utf-8') as json_file:
    for l, line in enumerate(json_file):
        if l < len(targets): #balance the dataset - topical articles are all at the end of the dataset
            dataset.append(json.loads(line))#temp.append(json.loads(line))#
        
temp2 = []
probs = [] #boundary probs from LSTM softmax      
PATH = 'C:/Users/Fish/docker/OUTPUTS/Testing/'
LSTM_output = pkl.load(open(PATH+'signal_1000K/1000K_softmax_signal400.pkl', 'rb'))
#LSTM_output = pkl.load(open(PATH+'wiki_700K/wiki700K_GPU_softmax_signal_400.pkl', 'rb'))
articles = LSTM_output['probs']
for a, article in enumerate(articles):
#for article, sentences in enumerate(scaled_articles):
    if a < len(targets):
        boundaries = [abs(probability) for probability in article]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
        probs.append([sum(boundaries)/float(len(boundaries)),
                      len([b for b in boundaries if b > SEG_THRESHOLD])/float(len(boundaries))])
#        temp2.append([sum(boundaries)/float(len(boundaries)), 
#                      len([b for b in boundaries if b > SEG_THRESHOLD])/float(len(boundaries))])

"""***********************import synthetic and topical training set of 20000 articles***********************"""       
selected = [] #obtain random unique indexes, to ensure a randomised training set
while len(selected) < (CONTROL + true_count): #additional samples are needed to replace topical component of test set
    index = random.randint(0, 9999)#2152)#synth_top contains 10,000, final_top contains 2153
    i = 0
    while i < len(selected) and index != selected[i]:
        i += 1
    if i == len(selected):
        selected.append(index)

temp3 = []
temp4 = [] #temporary containers while we replace topical articles
surplus = []
files = ['topical10K/synth_topical.jsonl', 'synthetic10K/test.jsonl']
#files = ['final/final_topical.jsonl', 'final/final.jsonl']
#files = ['topical10K/synth_topical.jsonl', 'final/final.jsonl']
for f, file in enumerate(files):   
    input_file = DATASETS + '/signal_working/TESTING/' + file
    with codecs.open(input_file, 'r', 'utf-8') as json_file:
        for i, line in enumerate(json_file):
            if f == 0:
                if i in selected[:CONTROL]:
                    temp3.append(json.loads(line))#dataset.append(json.loads(line))
                    temp4.append(f)#targets.append(f) #0 for topical, 1 for summary
                if i in selected[CONTROL:]:
                    surplus.append(json.loads(line))
            else:
                #this conditional only required when loading synth summaries (otherwise we just pull the whole summary set)
                if i in selected[:CONTROL]:
                    temp3.append(json.loads(line))#dataset.append(json.loads(line))
                    temp4.append(f)#targets.append(f) #0 for topical, 1 for summary
            #if i in selected[CONTROL:]:
                #surplus.append(json.loads(line))

####!!!! latest pkls are all in a certain order as per labels/indexs in docker output file
#this section is only needed for boundary evaln (Pk or homogeneity score)
###!!!these could also be used to generate an average softmax feature which we could add to regressor
"""***********************import synthetic and topical LSTM output of 20000 articles***********************"""       
#---these articles are drawn in the same sequence as above (as they were processed from the same JSON source)
temp5 = []
surplus2 = []
PATH = 'C:/Users/Fish/docker/OUTPUTS/Testing/signal_1000K/'
files = ['1000K_softmax_topical_10K.pkl', '1000K_softmax_synth_10K.pkl']
#files = ['1000K_softmax_FINAL_topical.pkl', '1000K_softmax_signalFINAL.pkl']
#files = ['1000K_softmax_topical_10K.pkl', '1000K_softmax_signalFINAL.pkl']
for f, file in enumerate(files):
    LSTM_output = pkl.load(open(PATH + file, 'rb'))
    articles = LSTM_output['probs']
    count = 0
    for a, article in enumerate(articles):#[:CONTROL]):
        if f == 0:
            if a in selected[:CONTROL]: #articles are retrieved in the identical order as the JSON source above
                #NOTE - as this NOT a neural net, there is no need to [wrap] each sample
                boundaries = [abs(probability) for probability in article]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
                temp5.append([sum(boundaries)/float(len(boundaries)),
                              len([b for b in boundaries if b > SEG_THRESHOLD])/float(len(boundaries))])
                #probs.append([sum(boundaries)/float(len(boundaries)), len([b for b in boundaries if b > THRESHOLD])])
            if a in selected[CONTROL:]:
                boundaries = [abs(probability) for probability in article]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
                surplus2.append([sum(boundaries)/float(len(boundaries)),
                             len([b for b in boundaries if b > SEG_THRESHOLD])/float(len(boundaries))])           
        else:
            #this conditional only required when loading synth summaries (otherwise we just pull the whole summary set)
            if a in selected[:CONTROL]:
                boundaries = [abs(probability) for probability in article]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
                temp5.append([sum(boundaries)/float(len(boundaries)),
                              len([b for b in boundaries if b > SEG_THRESHOLD])/float(len(boundaries))])

"""*************replace topical component of test sampling from signal corpus of 10000 articles***********************"""       
#dataset = [x[0] for a, x in enumerate(sorted(zip(temp, targets), key=lambda pair: pair[1], reverse=True)) if a < true_count] #move summary samples to top of the data set
#probs = [x[0] for a, x in enumerate(sorted(zip(temp2, targets), key=lambda pair: pair[1], reverse=True)) if a < true_count]
#targets.sort(key = lambda x: x, reverse=True)
#targets[true_count:] = [] #truncate target to to the same length (summaries only)
#
#for f, (file, softmax) in enumerate(zip(surplus, surplus2)):
#    dataset.append(file)
#    targets.append(0) #topical
#    probs.append(softmax)
#
#assert f == (true_count - 1)
#assert len(targets) == 2*true_count
#assert true_count == len([a for a in targets if a == 1])

for f, (file, label, softmax) in enumerate(zip(temp3, temp4, temp5)):
    dataset.append(file)
    targets.append(label) #topical
    probs.append(softmax)

assert f == (2*CONTROL - 1)
assert len(dataset) == 2*(CONTROL + true_count)
assert (true_count + CONTROL) == len([a for a in targets if a == 1])

"""****************************load Graphseg similarity output************************************************************"""    
#OUTPUTS = 'C:/Users/Fish/outputs/graphsim/'#graphseg/'#TEMP/'#
#GRAPHSEG_output = pkl.load(open(OUTPUTS+'GRAPHSEG_probs.pkl', 'rb'))
#sims = GRAPHSEG_output['probs']
#
#boundaries = []
#for a, article in enumerate(sims):
#    boundaries.append([])
#    for boundary in article:
#        boundaries[a].append(boundary[0]) #1st element is prob (2nd element, not needed here - is #contributing pairs)
#
#graphsim = {}
#with open(DATASETS+'signal_working/annotated/labelling_results.txt', 'r') as in_file: #sorted by file input order(OLD) - as per Graphseg output
##with open(OUTPUTS+'labels.txt', 'r') as in_file:
#    for a, annotation in enumerate(in_file.read().splitlines()):
#        article = annotation.split('\t')[0]
#        graphsim[article.strip()] = sum(boundaries[a])/float(len(boundaries[a])) #pred metrics are now accessible via index 
#    in_file.close()

print('loading google vectors...')
word2vec = gensim.models.KeyedVectors.load_word2vec_format(DATASETS+'word2vec/GoogleNews-vectors-negative300.bin', binary=True)
print('loaded')
nlp = en_core_web_sm.load()

### try restriucting to entities (BI) only
### try using W2V distnance from most prolific entity (this entity will have distance 0)
### try using distnance (subtraction) from most prolific entity
### try substituing C-Phrase vectors

files = []
entity_files = [] #abridged versions of files just containing entities
cleansed_articles = []
exceptions = [] #this isjust for auditing purposes
vectors = [] #all vectors across the whole corpus
ent_vectors = [] #all vectors across the whole corpus
unq_vectors = []
unq_ent_vectors = []
sources = [] #source document of each vector
ent_sources = []
vector_count = 0 #num vectors in the corpus
unq_ent_sources = []
unq_sources = []
doc_vectors = [] #set of vectors representing all words in each document
doc_ent_vectors = []
unq_doc_vectors = []
unq_doc_ent_vectors = []
rel_doc_vectors = []
rel_doc_ent_vectors = [] #set of vectors representing all entities in each document
rel_vectors = []
rel_ent_vectors = []
rel_sources = []
rel_ent_sources = []
cont_distances = []
ent_distances = []
cont_dist_matrix = []
ent_dist_matrix = [] #rel.distances as an array, to try with BOW classifier
num_clusters = []
num_vectors = []
num_entities = []
num_unq_vectors = []
num_unq_entities = []
similarities = []
SEGMENTER = segment_seperator + ',\d,.*?\.'

for s, sample in enumerate(dataset):#[:4*CONTROL]): #pull articles from JSON
    raw_content = re.sub(SEGMENTER, "", sample).strip()#strip any markups in the training set
    files.append(raw_content)
    #similarities.append([graphsim[index]])
    if float(s/10) == int(s/10):
        print('processing', s,'...') #there wont be any indexes!!
    tokens = [] #tokens.append([])
    entities = []
    contents = []
    size = 0 #number vectors in this article
    ent_size = 0
    unq_size = 0 #number unique vectors in this article
    unq_ent_size = 0 #number unique entities in this article
    doc_vectors.append(np.zeros(300))
    doc_ent_vectors.append(np.zeros(300))
    unq_doc_vectors.append(np.zeros(300))
    unq_doc_ent_vectors.append(np.zeros(300))
    exceptions.append([])
    #in_file = io.open(sample, 'r') 
    #files.append(in_file) #tfidf vectorizer says it accepts a list of file objects, but it FAILS 
    #with codecs.open(sample, 'r', 'ISO-8859-1') as in_file: #ISO encoding also works, but io is just as good if not better
    entity_files.append('') #start with empty string for this file
    cont_dist_matrix.append(np.zeros(20)) #similarities between 0 and 2 in 0.1 increments
    ent_dist_matrix.append(np.zeros(20)) #similarities between 0 and 2 in 0.1 increments
    #clean_txt = content.decode('utf-8').strip()
    #we could split paragraph clusters here, but we would need to re-synthesise/re-process all samples preserving multiple \n delimiters
    #clusters = [s.strip('\n') for s in re.split(SEPARATOR, raw_content) if len(s.strip('\n')) > 0]#content.decode('utf-8').strip())]
    #for c, cluster in enumerate(clusters):
    content = re.sub("[^a-zA-Z0-9\s,\.]+",'',re.sub('-',' ',raw_content)).strip()
    for token in nlp(content):
        word = token.text.lower()#re.sub('\W','',token.text.lower())#apostrophes (like don't) have been stripped from freq resource
        cleansed = re.sub('\d','#',re.sub("[^\w']",'',word))#retain commas as word2vec includes don't etc.
        if len(cleansed) > 0 and not word.isnumeric() and word not in stop_words and \
        token.lemma_ not in stop_words and not token.is_stop and (token.pos_ in CONTENT or token.ent_iob_ != 'O'):
            try:
                vectors.append(word2vec[cleansed])#cphrase[cleansed])#
            except:
                exceptions[s].append(cleansed)
            else:
                doc_vectors[s] += vectors[vector_count]#word2vec[cleansed] 
                tokens.append(cleansed)#tokens[s].append(cleansed)
                sources.append(s)
                cont = 0
                while cont < len(contents) and contents[cont][0] != cleansed:
                    cont += 1
                if cont < len(contents):
                    contents[cont][2] += 1
                else:
                    contents.append([cleansed, vectors[vector_count], 1])
                    unq_vectors.append(vectors[vector_count]) #only needed for kmeans clustering
                    unq_doc_vectors[s] += vectors[vector_count]#word2vec[cleansed]
                    unq_sources.append(s)
                    unq_size += 1
                if token.ent_iob_ != 'O':
                    ent_vectors.append(vectors[vector_count])
                    doc_ent_vectors[s] += vectors[vector_count]
                    ent_sources.append(s)
                    entity_files[s] = entity_files[s] + ' ' + cleansed
                    ent = 0
                    while ent < len(entities) and entities[ent][0] != cleansed:
                        ent += 1
                    if ent < len(entities):
                        entities[ent][2] += 1
                    else:
                        entities.append([cleansed, vectors[vector_count], 1])
                        unq_ent_vectors.append(vectors[vector_count]) #only needed for kmeans clustering
                        unq_doc_ent_vectors[s] += vectors[vector_count]
                        unq_ent_sources.append(s)
                        unq_ent_size += 1
                    ent_size += 1
                size += 1
                vector_count += 1
                    
    cleansed_articles.append(' '.join(tokens).strip())#(tokens[s]).strip())

    rel_doc_vectors.append(np.zeros(300))
    cont_distances.append([0, 0]) #start with zero distance & frequency
    #can try using similarity instead and sorting ascending instead
    contents.sort(key=lambda x: x[2], reverse=True)
    cont_dist_matrix[s][0] += contents[0][2] #add frequency for first most prominent content, but using its distance
    for content in contents[1:]:
        dist = word2vec.similarity(content[0], contents[0][0])
        cont_distances[s] += dist
        cont_dist_matrix[s][int(10*(1+dist))] += content[2] #add frequency for this content, but using its distance
        rel_vectors.append(content[1] - contents[0][1])
        rel_sources.append(s)
        rel_doc_vectors[s] += rel_vectors[-1]
    cont_distances[s][0] /= len(contents) #use the average distance as sample feature
    cont_distances[s][1] = len(contents)/size #use the quantity of unique contents as a feature (normalised by size of article)

##!!!!! this section must be restricted to entity tokens only --------
    rel_doc_ent_vectors.append(np.zeros(300))
    ent_distances.append([0, 0]) #start with zero distance & frequency
    if len(entities) > 0:
        #can try using similarity instead and sorting ascending instead
        entities.sort(key=lambda x: x[2], reverse=True)
        ent_dist_matrix[s][0] += entities[0][2] #add frequency for first most prominent entity, but using its distance
        for entity in entities[1:]:
            dist = word2vec.similarity(entity[0], entities[0][0])
            ent_distances[s] += dist
            ent_dist_matrix[s][int(10*(1+dist))] += entity[2] #add frequency for this entity, but using its distance
            rel_ent_vectors.append(entity[1] - entities[0][1])
            rel_ent_sources.append(s)
            rel_doc_ent_vectors[s] += rel_ent_vectors[-1]
        ent_distances[s][0] /= len(entities) #use the average distance as sample feature
        ent_distances[s][1] = len(entities)/size #use the quantity of unique entities as a feature (normalised by size of article)
        
    num_vectors.append(size)
    num_entities.append(ent_size)
    num_unq_vectors.append(unq_size)
    num_unq_entities.append(unq_ent_size)

names = ["Logistic Regression", "Linear SVM", "RBF SVM"]
#names = ["Multinomial NB", "Logistic Regression", "Linear SVM", "RBF SVM"]
sets = ["signal20K", "natural400"]

classifiers = [LogisticRegression(), SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1)]
#classifiers = [MultinomialNB(), LogisticRegression(),
#               SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1)]

y = np.asarray(targets)#[natural:]) #target is the same for all classifiers
natural = 2*true_count#2*CONTROL#y_test = np.asarray(targets[:2*true_count]) #natural Signal400 test set

indices = []
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True)
#ensure we are using the same folds for every experiment
for train, test in skf.split(files, y): # consider the first 30 examples
    indices.append([train, test])
for train, test in skf.split(files[:natural], y[:natural]): # consider the first 30 examples
    indices.append([train, test])
    
def evaluate(model, X, X_transf=None):
    score = [[],[],[],[],[],[]]
    for f, fold in enumerate(indices[:folds]): # consider the first 30 examples
        model.fit(X[fold[0]], y[fold[0]])
        score[0].append(model.score(X[fold[1]], y[fold[1]]))
        y_pred = model.predict(X[fold[1]])
        score[1].append(precision_score(y[fold[1]], y_pred))
        score[2].append(recall_score(y[fold[1]], y_pred))
        score[3].append(f1_score(y[fold[1]], y_pred))
        score[4].append(len([pred for p, pred in enumerate(y_pred) if pred == y[fold[1]][p] and pred == 0])/ \
             len([label for label in y[fold[1]] if label == 0]))
        score[5].append(len([pred for p, pred in enumerate(y_pred) if pred == y[fold[1]][p] and pred == 1])/ \
             len([label for label in y[fold[1]] if label == 1]))
    print(X.shape, sets[0],'Avg CV Acc:', round(np.mean(score[0]),4), end='')
    print(' precsn', round(np.mean(score[1]),4), end='')
    print(' recall', round(np.mean(score[2]),4), end='')
    print(' F1', round(np.mean(score[3]),4), end='')
    print(' topical', round(np.mean(score[4]),4), end='')
    print(' summary', round(np.mean(score[5]),4))
    model.fit(X[natural:], y[natural:])
    if X_transf==None:
        print(X[:natural].shape, sets[1], 'Test Acc:', round(model.score(X[:natural], y[:natural]),4), end='')
        y_pred = model.predict(X[:natural])
    else:
        print(X_transf.shape, sets[1], 'Test Acc:', round(model.score(X_transf, y[:natural]),4), end='')
        y_pred = model.predict(X_transf)
    print(' precsn', round(precision_score(y[:natural], y_pred),4), end='')
    print(' recall', round(recall_score(y[:natural], y_pred),4), end='')
    print(' F1', round(f1_score(y[:natural], y_pred),4), end='')
    print(' topical', round(len([pred for p, pred in enumerate(y_pred) if pred == y[p] and pred == 0])/true_count, 4), end='')
    print(' summary', round(len([pred for p, pred in enumerate(y_pred) if pred == y[p] and pred == 1])/true_count, 4))
#    score = []
#    for f, fold in enumerate(indices[folds:]): # consider the first 30 examples
#        model.fit(X[:natural][fold[0]], y[:natural][fold[0]])
#        score.append(model.score(X[:natural][fold[1]], y[:natural][fold[1]]))        
#    print(X[:natural].shape, sets[1],'Avg CV Train Acc:', round(np.mean(score),4))

#scope = [[s+1-natural, slice(0, len([v for v in sources if v > natural])),
#         slice(0, len([v for v in unq_ent_sources if v > natural])),
#         slice(0, len([v for v in rel_sources if v > natural]))],
#    [natural, slice(len([v for v in sources if not v > natural]), len(vectors)),
#     slice(len([v for v in unq_ent_sources if not v > natural]), len(ent_vectors)),
#     slice(len([v for v in rel_sources if not v > natural]), len(rel_vectors))]]

for model, name in zip(classifiers, names):
    
    print('using hashed count vectors -',name,'+++++++++++++++++++++++++++++')
    hasher = HashingVectorizer(n_features=1000, stop_words='english', norm = 'l2')
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    X = vectorizer.fit_transform(files)
    X_transf = vectorizer.transform(files[:natural])
    evaluate(model, X, X_transf)  

    print('using hashed entity count vectors -',name,'+++++++++++++++++++++++++++++')
    hasher = HashingVectorizer(n_features=1000, stop_words='english', norm = 'l2')
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    X = vectorizer.fit_transform(entity_files)
    X_transf = vectorizer.transform(entity_files[:natural])
    evaluate(model, X, X_transf)  

    print('using avg softmax -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray(probs)
    evaluate(model, X)
    
    print('now try with each feature in turn -',name)
    for feature in range(X.shape[1]):
        X = np.asarray([[f[feature]] for f in probs])
        evaluate(model, X)
    
#    print('using avg similarity -',name)
#    X = np.asarray(similarities)
#    evaluate(model, X)

    print('using aggregated doc vectors -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray(doc_vectors)
    evaluate(model, X)
    
    print('then normalised by vector count -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray([vector/size for vector, size in zip(doc_vectors, num_vectors)])
    evaluate(model, X)

    print('using aggregated unique vectors -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray(unq_doc_vectors)
    evaluate(model, X)
    
    print('then normalised by vector count -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray([vector/size for vector, size in zip(unq_doc_vectors, num_unq_vectors)])
    evaluate(model, X)

    print('using aggregated relative vectors -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray(rel_doc_vectors)
    evaluate(model, X)
    
    print('then normalised by vector count -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray([(vector if size == 0 else vector/size) for vector, size in zip(rel_doc_vectors, num_unq_vectors)])
    evaluate(model, X)
    
    print('using avg scalar distances -',name,'+++++++++++++++++++++++++++++')    
    X = np.asarray(cont_distances)
    evaluate(model, X)
    
    print('now try with each feature in turn -',name)
    for feature in range(X.shape[1]):
        X = np.asarray([[f[feature]] for f in cont_distances])
        evaluate(model, X)
        
    print('then with distance frequencies-', name,'+++++++++++++++++++++++++++++')
    X = np.asarray(cont_dist_matrix)
    evaluate(model, X)

    print('using aggregated doc entity vectors -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray(doc_ent_vectors)
    evaluate(model, X)
    
    print('then normalised by vector count -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray([(vector if size == 0 else vector/size) for vector, size in zip(doc_ent_vectors, num_entities)])
    evaluate(model, X)
    
    print('using aggregated unique entity vectors -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray(unq_doc_ent_vectors)
    evaluate(model, X)
    
    print('then normalised by vector count -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray([(vector if size == 0 else vector/size) for vector, size in zip(unq_doc_ent_vectors, num_unq_entities)])
    evaluate(model, X)
   
    print('using aggregated relative entity vectors -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray(rel_doc_ent_vectors)
    evaluate(model, X)
    
    print('then normalised by vector count -',name,'+++++++++++++++++++++++++++++')
    X = np.asarray([(vector if size == 0 else vector/size) for vector, size in zip(rel_doc_ent_vectors, num_unq_entities)])
    evaluate(model, X)

    print('using avg entity scalar distances -',name,'+++++++++++++++++++++++++++++')    
    X = np.asarray(ent_distances)
    evaluate(model, X)
    
    print('now try with each feature in turn -',name)
    for feature in range(X.shape[1]):
        X = np.asarray([[f[feature]] for f in ent_distances])
        evaluate(model, X)
    
    print('then with Entity distance frequencies-', name,'+++++++++++++++++++++++++++++')
    X = np.asarray(ent_dist_matrix)
    evaluate(model, X)

print('then with Entity docs','+++++++++++++++++++++++++++++')
vectorizer = TfidfVectorizer(max_df=0.6)
X = vectorizer.fit_transform(entity_files)
X_transf = vectorizer.transform(entity_files[:natural])
evaluate(MultinomialNB(), X, X_transf)

#use number of tfidf extracted features as basis for word vector clustering 
print('using counts of rel. entity cluster nodes','+++++++++++++++++++++++++++++')
clusters = int(X.shape[1]/(5*math.sqrt(s))) #use number of tfidf extracted features as basis for word vector clustering 
kmeans = KMeans(n_clusters = clusters)
nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(rel_ent_vectors, rel_ent_sources)]))# if not s < natural]))
X = np.zeros([s+1, clusters])#-natural, clusters])
for source, node in zip([v for v in rel_ent_sources], nodes):# if not v < natural], nodes):
    X[source, node] += 1# - natural, node] += 1
#kmeans = KMeans(n_clusters = clusters)
#nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(rel_ent_vectors, rel_ent_sources) if s < natural]))
#X_test = np.zeros([natural, clusters])
#for source, node in zip([v for v in rel_ent_sources if v < natural], nodes):
#    X_test[source, node] += 1
evaluate(MultinomialNB(), X)#, X_test)
for model, name in zip(classifiers, names):
    print('repeat with -',name)
    evaluate(model, X)#, X_test) 

print('then using counts of entity cluster nodes','+++++++++++++++++++++++++++++')
#clusters = int(X.shape[1]/(5*math.sqrt(s))) #use same dimensionality as full vector nodes above
kmeans = KMeans(n_clusters = clusters)
nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(ent_vectors, ent_sources)]))# if not s < natural]))
X = np.zeros([s+1, clusters])#-natural, clusters])
for source, node in zip([v for v in ent_sources], nodes):# if not v < natural], nodes):
    X[source, node] += 1# - natural, node] += 1
#kmeans = KMeans(n_clusters = clusters)
#nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(ent_vectors, ent_sources) if s < natural]))
#X_test = np.zeros([natural, clusters])
#for source, node in zip([v for v in ent_sources if v < natural], nodes):
#    X_test[source, node] += 1
evaluate(MultinomialNB(), X)#, X_test)
for model, name in zip(classifiers, names):
    print('repeat with -',name)
    evaluate(model, X)#, X_test) 
    
print('then using counts of unique entity cluster nodes','+++++++++++++++++++++++++++++')
#clusters = int(X.shape[1]/(10*math.sqrt(s))) #use number of tfidf extracted features as basis for word vector clustering 
kmeans = KMeans(n_clusters = clusters)
nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(unq_ent_vectors, unq_ent_sources)]))# if not s < natural]))
X = np.zeros([s+1, clusters])#-natural, clusters])
for source, node in zip([v for v in unq_ent_sources], nodes):# if not v < natural], nodes):
    X[source, node] += 1# - natural, node] += 1
#kmeans = KMeans(n_clusters = clusters)
#nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(unq_ent_vectors, unq_ent_sources) if s < natural]))
#X_test = np.zeros([natural, clusters])
#for source, node in zip([v for v in unq_ent_sources if v < natural], nodes):
#    X_test[source, node] += 1
evaluate(MultinomialNB(), X)#, X_test)
for model, name in zip(classifiers, names):
    print('repeat with -',name)
    evaluate(model, X)#, X_test) 

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.6)
X = vectorizer.fit_transform(files)
X_transf = vectorizer.transform(files[:natural])
print('using naive count vectors','+++++++++++++++++++++++++++++')
evaluate(MultinomialNB(), X, X_transf)
#I dont think I need to re-initialise the clf each time but I dont want to take the risk that its continuing training the previous one!

print('then with pre-processed docs','+++++++++++++++++++++++++++++')
vectorizer = TfidfVectorizer(max_df=0.6)
X = vectorizer.fit_transform(cleansed_articles)
X_transf = vectorizer.transform(cleansed_articles[:natural])
evaluate(MultinomialNB(), X, X_transf)

#use number of tfidf extracted features as basis for word vector clustering 
print('using counts of rel. cluster nodes','+++++++++++++++++++++++++++++')
clusters = int(X.shape[1]/(5*math.sqrt(s))) #use number of tfidf extracted features as basis for word vector clustering 
kmeans = KMeans(n_clusters = clusters)
nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(rel_vectors, rel_sources)]))# if not s < natural]))
X = np.zeros([s+1, clusters])#-natural, clusters])
for source, node in zip([v for v in rel_sources], nodes):# if not v < natural], nodes):
    X[source, node] += 1# - natural, node] += 1
#kmeans = KMeans(n_clusters = clusters)
#nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(rel_vectors, rel_sources) if s < natural]))
#X_test = np.zeros([natural, clusters])
#for source, node in zip([v for v in rel_sources if v < natural], nodes):
#    X_test[source, node] += 1`
evaluate(MultinomialNB(), X)#, X_test)
for model, name in zip(classifiers, names):
    print('repeat with -',name)
    evaluate(model, X)#, X_test) 

print('then using counts of cluster nodes','+++++++++++++++++++++++++++++')
#clusters = int(X.shape[1]/(10*math.sqrt(s))) #use number of tfidf extracted features as basis for word vector clustering 
kmeans = KMeans(n_clusters = clusters)
nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(vectors, sources)]))# if not s < natural]))
X = np.zeros([s+1, clusters])#-natural, clusters])
for source, node in zip([v for v in sources], nodes):# if not v < natural], nodes):
    X[source, node] += 1# - natural, node] += 1
#kmeans = KMeans(n_clusters = clusters)
#nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(vectors, sources) if s < natural]))
#X_test = np.zeros([natural, clusters])
#for source, node in zip([v for v in sources if v < natural], nodes):
#    X_test[source, node] += 1
evaluate(MultinomialNB(), X)#, X_test)
for model, name in zip(classifiers, names):
    print('repeat with -',name)
    evaluate(model, X)#, X_test) 
    
print('then using counts of unique cluster nodes','+++++++++++++++++++++++++++++')
#clusters = int(X.shape[1]/(10*math.sqrt(s))) #use number of tfidf extracted features as basis for word vector clustering 
kmeans = KMeans(n_clusters = clusters)
nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(unq_vectors, unq_sources)]))# if not s < natural]))
X = np.zeros([s+1, clusters])#-natural, clusters])
for source, node in zip([v for v in unq_sources], nodes):# if not v < natural], nodes):
    X[source, node] += 1# - natural, node] += 1
#kmeans = KMeans(n_clusters = clusters)
#nodes = kmeans.fit_predict(np.asarray([v for v, s in zip(unq_vectors, unq_sources) if s < natural]))
#X_test = np.zeros([natural, clusters])
#for source, node in zip([v for v in unq_sources if v < natural], nodes):
#    X_test[source, node] += 1
evaluate(MultinomialNB(), X)#, X_test)
for model, name in zip(classifiers, names):
    print('repeat with -',name)
    evaluate(model, X)#, X_test) 



