# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:46:38 2018
Uses softmax threshold for clustering, rather than Graphseg
NOTE - graphsim similarities have been removed - to evaluatae Graphsim use original clustering_evaln.py
@author: Fish
"""

import os
import json
import re
import io
import math
import random
import codecs
from pathlib2 import Path
import pickle as pkl
import gensim
import en_core_web_sm
import numpy as np

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from text_normaliser import normalise

PATH = 'C:/Users/Fish/Documents/GitHub/datasets/signal/'#testing/'#'#TEMP/'#
DATASETS = 'C:/Users/Fish/Documents/GitHub/datasets/'
VECTORS = 'C:/Users/Fish/big_datasets/cphrase/cphrase.txt'
SEPARATOR = "=========="
segment_seperator = "========"
NUM_ARTICLES = 1000 #number of articles we wish to synthesise
RESOURCES = 'C:/Users/Fish/Documents/GitHub/graphseg/source/res/'#stopwords.txt'
CONTENT = ['ADJ', 'VERB', 'ADV', 'NOUN', 'PROPN', 'PRON', 'INTJ']#these are POStags - high level
#CONTENT = ['NOUN', 'PROPN'] #need to re-set this to the line above (this is for testing ideas)
THRESHOLD = 0.4 #segmentation threshold

CONTROL = 338#1200 #num signal20K set being processed (total will be double)

samples = []
filenames = []
word_freqs = []
sum_freqs = 0
vocabulary = 0

def get_word(word):#unclear whether this is a list of LEMMAS or not????
    w = 0
    while word != word_freqs[w][0] and w < vocabulary:
        w += 1
    #if w == vocabulary:
        #return 1
    return w

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

#!!!!this should be callable functions in a util module
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

LABELS = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/TESTING/signal400/'
#with io.open(input_file, 'r', encoding ls = 'utf8') as json_file:
with codecs.open(LABELS+'signal400.jsonl', 'r', 'utf-8') as json_file:
    for l, line in enumerate(json_file):
        samples.append(json.loads(line))#dataset.append(json.loads(line))

"""****************************load LSTM softmax output************************************************************"""
NUM_FEATURES = 40
THRESHOLD = 0.05 # all boundaries above this will be preserved on normalisation
SEG_THRESHOLD = 0.4
MAX_PAD = 0.005 
OUTPUTS = 'C:/Users/Fish/docker/OUTPUTS/Testing/signal_1000K/'
#LSTM_output = pkl.load(open(OUTPUTS+'wiki700K_CPU_softmax_signal_400.pkl', 'rb'))
LSTM_output = pkl.load(open(OUTPUTS+'1000K_softmax_signal400.pkl', 'rb'))
probabilities = LSTM_output['probs']

boundaries = []
softmax = [] #for the dictionary lookup
for a, article in enumerate(probabilities):
    boundaries.append([abs(probability) for probability in article])#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)])
    softmax.append([sum(boundaries[a])/float(len(boundaries[a])),
                len([b for b in boundaries[a] if b > SEG_THRESHOLD])/float(len(boundaries[a]))]) #pred metrics are now accessible via index 

annotations = []
#with open(DATASETS+'signal_working/TESTING/signal400/labels_JSON.txt', 'r') as in_file: 
with open(DATASETS+'signal_working/TESTING/signal400/labels_JSON-no_reversions.txt', 'r') as in_file: #sorted by file input order(OLD) - as per Graphseg output
    for a, annotation in enumerate(in_file.read().splitlines()):
        article, label = annotation.split('\t')
        annotations.append(1 if label =='TRUE' else 0)
    in_file.close()

true_count = len([a for a in annotations if a == 1])
targets = annotations[:2*true_count] #balance the dataset - topical articles are all at the end of the dataset
temp = samples[:2*true_count]
temp2 = softmax[:2*true_count]
temp_b = boundaries[:2*true_count]
print(true_count, 'summaries in', len(targets), 'dataset')

"""***********************import synthetic and topical training set of 20000 articles***********************"""       
selected = [] #obtain random unique indexes, to ensure a randomised training set
while len(selected) < (CONTROL + true_count): #additional samples are needed to replace topical component of test set
    index = random.randint(0, 2152)#9999) #synth_top contains 10,000, final_top contains 2153
    i = 0
    while i < len(selected) and index != selected[i]:
        i += 1
    if i == len(selected):
        selected.append(index)

temp3 = []
temp4 = [] #temporary containers while we replace topical articles
surplus = []
#files = ['topical10K/synth_topical.jsonl', 'synthetic10K/test.jsonl']
#files = ['final/final_topical.jsonl', 'final/final.jsonl']#'synthetic10K/test.jsonl']
files = ['topical10K/synth_topical.jsonl', 'final/final.jsonl']#'synthetic10K/test.jsonl']
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
                temp3.append(json.loads(line))#dataset.append(json.loads(line))
                temp4.append(f)#targets.append(f) #0 for topical, 1 for summary
            #if i in selected[CONTROL:]:
                #surplus.append(json.loads(line))

"""***********************import synthetic and topical LSTM output of 20000 articles***********************"""       
#---these articles are drawn in the same sequence as above (as they were processed from the same JSON source)
temp5 = []
temp_b2 = []
surplus2 = []
surplus3 = []
PATH = 'C:/Users/Fish/docker/OUTPUTS/Testing/signal_1000K/'
#files = ['1000K_softmax_topical_10K.pkl', '1000K_softmax_synth_10K.pkl']
#files = ['1000K_softmax_FINAL_topical.pkl', '1000K_softmax_signalFINAL.pkl']
files = ['1000K_softmax_topical_10K.pkl', '1000K_softmax_signalFINAL.pkl']
for f, file in enumerate(files):
    LSTM_output = pkl.load(open(PATH + file, 'rb'))
    articles = LSTM_output['probs']
    count = 0
    for a, article in enumerate(articles):#[:CONTROL]):
        if f == 0:
            if a in selected[:CONTROL]: #articles are retrieved in the identical order as the JSON source above
                #NOTE - as this NOT a neural net, there is no need to [wrap] each sample
                boundaries = [abs(probability) for probability in article]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
                temp_b2.append(boundaries)
                temp5.append([sum(boundaries)/float(len(boundaries)),
                              len([b for b in boundaries if b > SEG_THRESHOLD])/float(len(boundaries))])
                #probs.append([sum(boundaries)/float(len(boundaries)), len([b for b in boundaries if b > THRESHOLD])])
            if a in selected[CONTROL:]:
                boundaries = [abs(probability) for probability in article]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
                surplus3.append(boundaries)
                surplus2.append([sum(boundaries)/float(len(boundaries)),
                             len([b for b in boundaries if b > SEG_THRESHOLD])/float(len(boundaries))])           
        else:
            boundaries = [abs(probability) for probability in article]#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)] #deos this need to be a numpy array?
            temp_b2.append(boundaries)
            temp5.append([sum(boundaries)/float(len(boundaries)),
                          len([b for b in boundaries if b > SEG_THRESHOLD])/float(len(boundaries))])

            #probs.append([sum(boundaries)/float(len(boundaries)), len([b for b in boundaries if b > THRESHOLD])])
            
"""*************replace topical component of test sampling from signal corpus of 10000 articles***********************"""       
dataset = [x[0] for a, x in enumerate(sorted(zip(temp, targets), key=lambda pair: pair[1], reverse=True)) if a < true_count] #move summary samples to top of the data set
probs = [x[0] for a, x in enumerate(sorted(zip(temp2, targets), key=lambda pair: pair[1], reverse=True)) if a < true_count]
bounds = [x[0] for a, x in enumerate(sorted(zip(temp_b, targets), key=lambda pair: pair[1], reverse=True)) if a < true_count]
targets.sort(key = lambda x: x, reverse=True)
targets[true_count:] = [] #truncate target to to the same length (summaries only)

for f, (file, softmax, boundaries) in enumerate(zip(surplus, surplus2, surplus3)):
    dataset.append(file)
    targets.append(0) #topical
    probs.append(softmax)
    bounds.append(boundaries)

assert f == (true_count - 1)
assert len(targets) == 2*true_count
assert true_count == len([a for a in targets if a == 1])

for f, (file, label, softmax, boundaries) in enumerate(zip(temp3, temp4, temp5, temp_b2)):
    dataset.append(file)
    targets.append(label) #topical
    probs.append(softmax)
    bounds.append(boundaries)

assert f == (2*CONTROL - 1)
assert len(dataset) == 2*(CONTROL + true_count)
assert (true_count + CONTROL) == len([a for a in targets if a == 1])

print('loading google vectors...')
#cphrase = load_vectors(VECTORS)
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
#tokens = [] #set of words in each document
vectors = [] #all vectors across the whole corpus
sources = [] #source document of each vector
ent_sources = []
ent_vectors = [] #all vectors across the whole corpus
doc_vectors = [] #set of vectors representing all words in each document
ent_doc_vectors = [] #set of vectors representing all entities in each document
rel_vectors = []
rel_sources = []
ent_distances = [] #this is the root > node distances
ent_dist_matrix = [] #rel.distances as an array, to try with BOW classifier
num_clusters = []
num_vectors = []
features = [] #unsupervised (cohesion & separation) / supervised (homogeneity)
vfeatures = [] #this one is for calinski harabaz
cfeatures = [] #non-linear clustering based features
efeatures = [] #and same again for entities only
div_factor = vocabulary + sum_freqs
SEGMENTER = segment_seperator + ',\d,.*?\.'

for s, sample in enumerate(dataset):#get_files(PATH)[:2*true_count]):
    if float(s/10) == int(s/10):
        print('processing', s,'...') #there wont be any indexes!!
    distances = [] #vector distances between each pair of words in the document
    ent_pairs = [] #the same again for entities only
    segments = [] #un-supervised linear clusters (0 > N, for each word in the document)
    tokens = [[],[]] #tokens.append([])
    ent_tokens = [[],[]]
    entities = [] #list of unique entities in this document
    sample_vectors = [] #original plan inidividual vector each word, but NB classifier requires single vector for each sample article
    ent_sample_vectors = []
    size = 0 #vectors in this article
    doc_vectors.append(np.zeros(300))
    exceptions.append([])
    features.append([])
    vfeatures.append([])
    cfeatures.append([])
    efeatures.append([])
    ent_dist_matrix.append(np.zeros(20)) #similarities between 0 and 2 in 0.1 increments
    #in_file = io.open(sample, 'r') 
    #files.append(in_file) #tfidf vectorizer says it accepts a list of file objects, but it FAILS 
    #with codecs.open(sample, 'r', 'ISO-8859-1') as in_file: #ISO encoding also works, but io is just as good if not better
    raw_content = re.sub('[=\s]+\.',"",re.sub(SEGMENTER, "", sample)).strip()#strip any markups in the training set
    #above inludes stripping extraneous content in some of the topical sources (mainly where accounting figures have been ghosted)
    files.append(raw_content)
    entity_files.append('') #start with empty string for this file
    #clean_txt = content.decode('utf-8').strip()
    sentences = [s.strip('\n') for s in raw_content.split('\n') if len(s.strip('\n')) > 0]#content.decode('utf-8').strip())]
    assert len(sentences) == 1 + len(bounds[s])
    clusters = ['']
    c = 0
    for b, boundary in enumerate(bounds[s]):
        clusters[c] = (clusters[c] + '\n' + sentences[b]).strip('\n')
        if boundary > SEG_THRESHOLD:
            clusters.append('') #start new cluster
            c += 1
    clusters[c] = (clusters[c] + '\n' + sentences[b]).strip('\n')
    for c, cluster in enumerate(clusters):
        content = re.sub("[^a-zA-Z0-9\s,\.']+",'',re.sub('-',' ',cluster)).strip()
        for token in nlp(content):
            word = re.sub('\W','',token.text.lower())#token.text.lower()#apostrophes (like don't) have been stripped from freq resource
            cleansed = re.sub('\d','#',re.sub("[^\w']",'',token.text.lower()))#retain commas as word2vec includes don't etc.
            if len(cleansed) > 0 and not word.isnumeric() and word not in stop_words and \
            token.lemma_ not in stop_words and not token.is_stop and (token.pos_ in CONTENT or token.ent_iob_ != 'O'):
                try:
                    sample_vectors.append(word2vec[cleansed])#cphrase[cleansed])#
                except:
                    exceptions[s].append(cleansed)
                else:
                    doc_vectors[s] += sample_vectors[size]#word2vec[cleansed]# 
                    tokens[0].append(cleansed)#tokens[s].append(cleansed)
                    vectors.append(sample_vectors[size]) #only needed for kmeans clustering
                    segments.append(c)
                    distances.append([])
                    sources.append(s)
                    if token.ent_iob_ == 'O':
                        factor = 1
                    else:
                        factor = 1
                        ent_sample_vectors.append(sample_vectors[size])
                        ent_tokens[0].append(cleansed)
                        ent_tokens[1].append(-math.log10((word_freqs[get_word(re.sub('\d','',word))][1] + 1)/div_factor))
                        ent_pairs.append([])
                        ent = 0
                        while ent < len(entities) and entities[ent][0] != cleansed:
                            ent += 1
                        if ent < len(entities):
                            entities[ent][2] += 1
                        else:
                            entities.append([cleansed, sample_vectors[size], 1])
                            ent_vectors.append(sample_vectors[size]) #only needed for kmeans clustering
                            ent_sources.append(s)
                    tokens[1].append(-factor*math.log10((word_freqs[get_word(re.sub('\d','',word))][1] + 1)/div_factor))
                    size += 1
                if token.ent_iob_ != 'O':
                    entity_files[s] = entity_files[s] + ' ' + cleansed
                        
    for t1, (token_1, icf_1) in enumerate(zip(tokens[0], tokens[1])):#(tokens[s]):
        for t2, (token_2, icf_2) in enumerate(zip(tokens[0], tokens[1])):#(tokens[s]):
            if t1 == t2:
                distances[t1].append(0)
            else:
                try:
                    ic_factor = min(icf_1, icf_2)
                    distances[t1].append(max(0, 4 - ic_factor*(word2vec.similarity(token_1, token_2))))
                except:
                    print('distance fail:',token_1 ,'/', token_2)
                 
    cleansed_articles.append(' '.join(tokens[0]).strip())#(tokens[s]).strip())

###!!!!! this section must be restricted to entity tokens only --------
    ent_doc_vectors.append(np.zeros(300))
    ent_distances.append([0, 0]) #start with zero distance & frequency
    if len(entities) == 0:
        efeatures[s].append(15) #not much other choice but to default to zero
        efeatures[s].append(2)
    else:
        #can try using similarity instead and sorting ascending instead
        entities.sort(key=lambda x: x[2], reverse=True)
        ent_dist_matrix[s][0] += entities[0][2] #add frequency for first most prominent entity, but using its distance
        for entity in entities[1:]:
            dist = word2vec.similarity(entity[0], entities[0][0])
            ent_distances[s] += dist
            ent_dist_matrix[s][int(10*(1+dist))] += entity[2] #add frequency for this entity, but using its distance
            rel_vectors.append(entity[1] - entities[0][1])
            rel_sources.append(s)
            ent_doc_vectors[s] += rel_vectors[-1]
        ent_distances[s][0] /= len(entities) #use the average distance as sample feature
        ent_distances[s][1] = len(entities) #use the quantity of unique entities as a feature 
        
        if c == 0 or len(entities) < 3: #at least 2 (#samples - 1) clusters is needed for the sillhouette and calinski metrics
            efeatures[s].append(10) #avg calinski score is 10 (anecdotally)
            efeatures[s].append(3) #avg. sillhouette score is 0.3 (anecodtally)
        else:
            X = np.asarray(ent_sample_vectors)
            kmeans = KMeans(n_clusters = c+1)
            nodes = kmeans.fit_predict(X)
            efeatures[s].append(metrics.calinski_harabaz_score(X, nodes))
            for t1, (token_1, icf_1) in enumerate(zip(ent_tokens[0], ent_tokens[1])):#(tokens[s]):
                for t2, (token_2, icf_2) in enumerate(zip(ent_tokens[0], ent_tokens[1])):#(tokens[s]):
                    if t1 == t2:
                        ent_pairs[t1].append(0)
                    else:
                        try:
                            ic_factor = min(icf_1, icf_2)
                            ent_pairs[t1].append(max(0, 4 - ic_factor*(word2vec.similarity(token_1, token_2))))
                        except:
                            print('ent distance fail:',token_1 ,'/', token_2)
            X = np.asarray(ent_pairs)
            efeatures[s].append(10*metrics.silhouette_score(X, nodes, metric='euclidean'))
                        
    num_vectors.append(size)
    num_clusters.append(c+1)
    X1 = np.asarray(sample_vectors)
    X2 = np.asarray(distances)       
    #NOTE the following linear clusters were only evaluated using all tokens
    #unlike the natural Kmeans clustering which were additionally evaluated using entities only
    #this is due to the linear constraint having been relaxed, which should permit better clustering
    if c == 0:
        vfeatures[s].append(2.4) #calinski score - linear clusters
        features[s].append(0) #sillhouettes score - linear clusters
        cfeatures[s].append(2.4) #calinsky score - natural clusters
        cfeatures[s].append(0) #sillhouettes score - natural 
    else:
        y = np.asarray(segments) #use LINEAR segmentation supplid by LSTM
        vfeatures[s].append(metrics.calinski_harabaz_score(X1, y))
        features[s].append(10*metrics.silhouette_score(X2, y, metric='euclidean'))
        kmeans = KMeans(n_clusters = c+1) #use NON-linear clustering
        nodes = kmeans.fit_predict(X1)
        cfeatures[s].append(metrics.calinski_harabaz_score(X1, nodes))
        cfeatures[s].append(10*metrics.silhouette_score(X2, nodes, metric='euclidean'))
    vfeatures[s].append((c+1)/(b+1)) 
    features[s].append((c+1)/(b+1)) #normalise #cluster by #sentences in sample
    #features[s].append(ent_distances[s]) #can also try including this in main feature array
    #homogeneity (and Pk) can only be measured on synthetic set
    #score[1].append(metrics.homogeneity_score(labels_true, predictions[s])

names = ["Logistic Regression", "Linear SVM", "RBF SVM"]
#names = ["Multinomial NB", "Logistic Regression", "Linear SVM", "RBF SVM"]
sets = ["signal20K", "natural400"]

classifiers = [LogisticRegression(), SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1)]
#classifiers = [MultinomialNB(), LogisticRegression(),
#               SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1)]

y = np.asarray(targets) #target is the same for all classifiers
natural = 2*true_count

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

for model, name in zip(classifiers, names):

    """******first few tests are just to audit against evaluation.py - shoiuld give dame result*********"""
    print('using hashed count vectors -',name)
    hasher = HashingVectorizer(n_features=1000, stop_words='english', norm = 'l2')
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    X = vectorizer.fit_transform(files)
    X_transf = vectorizer.transform(files[:natural])
    evaluate(model, X, X_transf)   
    
    print('using avg softmax -',name)
    X = np.asarray(probs)
    evaluate(model, X)
    
    print('now try with each feature in turn -',name)
    for feature in range(X.shape[1]):
        X = np.asarray([[f[feature]] for f in probs])
        evaluate(model, X)
    
    #    print('now try with each feature in turn -',name)
    #    for feature in range(X.shape[1]):
    #        X = np.asarray([[f[feature]] for f in similarities])
    #        evaluate(model, X)
    
    #******************these are already assessed on evaluation.py*****************
    #    print('using aggregated doc vectors -',name)
    #    X = np.asarray(doc_vectors)
    #    evaluate(model, X)
    #    
    #    print('then normalised by vector count -',name)
    #    X = np.asarray([vector/size for vector, size in zip(doc_vectors, num_vectors)])
    #    evaluate(model, X)
    #    
    #    print('using aggregated entity vectors -',name)
    #    X = np.asarray(ent_doc_vectors)
    #    evaluate(model, X)
    #    
    #    print('then normalised by vector count -',name)
    #    X = np.asarray([(vector if size[1] == 0 else vector/size[1]) for vector, size in zip(ent_doc_vectors, ent_distances)])
    #    evaluate(model, X)
    #    
    #    print('using avg entity scalar distances -',name)    
    #    X = np.asarray(ent_distances)
    #    evaluate(model, X) 
    #    
    #    print('then with Entity distance frequencies')
    #    X = np.asarray(ent_dist_matrix)
    #    evaluate(model, X)
        
    print('using sillhouette metrics -',name)
    X = np.asarray(features)
    evaluate(model, X)
    
    print('now try with each feature in turn -',name)
    for feature in range(X.shape[1]):
        X = np.asarray([[f[feature]] for f in features])
        evaluate(model, X)
        
    print('now try with normalised sill score -',name)
    X = np.asarray([[f[0] / count] for f, count in zip(features, num_clusters)])
    evaluate(model, X)
    
    print('using calinski metrics -',name)
    X = np.asarray(vfeatures)
    evaluate(model, X)
    
    print('now try with each feature in turn -',name)
    for feature in range(X.shape[1]):
        X = np.asarray([[f[feature]] for f in vfeatures])
        evaluate(model, X)
    
    print('now try with normalised calins score -',name)
    X = np.asarray([[f[0] / count] for f, count in zip(vfeatures, num_clusters)])
    evaluate(model, X)
    
    print('using natural clusterng metrics -',name)
    X = np.asarray(cfeatures)
    evaluate(model, X)
    
    quant = X.shape[1]
    print('now try with each metric in turn -',name)
    for feature in range(quant):
        X = np.asarray([[f[feature]] for f in cfeatures])
        evaluate(model, X)
    
    print('now try with nornalised feartures -',name)
    for feature in range(quant):
        X = np.asarray([[f[feature] / count] for f, count in zip(cfeatures, num_clusters)])
        evaluate(model, X)
    
    print('using natural entity clusterng metrics -',name)
    X = np.asarray(efeatures)
    evaluate(model, X)
    
    quant = X.shape[1]
    print('now try with each metric in turn -',name)
    for feature in range(quant):
        X = np.asarray([[f[feature]] for f in efeatures])
        evaluate(model, X)
    
    print('now try with nornalised feartures -',name)
    for feature in range(quant):
        X = np.asarray([[f[feature] / count] for f, count in zip(efeatures, num_clusters)])
        evaluate(model, X)
    
    #******************these are already assessed on evaluation.py*****************
    #print('using naive count vectors')
    #evaluate(MultinomialNB(), X)
    ##I dont think I need to re-initialise the clf each time but I dont want to take the risk that its continuing training the previous one!
    # 
    #print('then with pre-processed docs')
    #vectorizer = TfidfVectorizer(max_df=0.6)
    #X = vectorizer.fit_transform(cleansed_articles)
    #evaluate(MultinomialNB(), X)
    #
    #print('then using counts of cluster nodes')
    #clusters = int(X.shape[1]/(5*math.sqrt(s))) #use number of tfidf extracted features as basis for word vector clustering
    #kmeans = KMeans(n_clusters = clusters)
    #nodes = kmeans.fit_predict(np.asarray(vectors))
    #X = np.zeros([s+1, clusters])
    #for source, node in zip(sources, nodes):
    #    X[source, node] += 1
    #evaluate(MultinomialNB(), X)
    #print('repeat with -',names[0])
    #evaluate(classifiers[0], X)
    #
    #print('then with Entity docs')
    #vectorizer = TfidfVectorizer(max_df=0.6)
    #X = vectorizer.fit_transform(entity_files)
    #evaluate(MultinomialNB(), X)
    #
    #print('then using counts of entity cluster nodes')
    ##clusters = int(X.shape[1]/(5*math.sqrt(s))) #use same dimensionality as full vector nodes above
    #kmeans = KMeans(n_clusters = clusters)
    #nodes = kmeans.fit_predict(np.asarray(ent_vectors))
    #X = np.zeros([s+1, clusters])
    #for source, node in zip(ent_sources, nodes):
    #    X[source, node] += 1
    #evaluate(MultinomialNB(), X)
    #print('repeat with -',names[0])
    #evaluate(classifiers[0], X)
    #
    ##use number of tfidf extracted features as basis for word vector clustering 
    #print('using counts of rel. entity cluster nodes')
    ##clusters = int(X.shape[1]/(10*math.sqrt(s))) #use number of tfidf extracted features as basis for word vector clustering 
    #kmeans = KMeans(n_clusters = clusters)
    #nodes = kmeans.fit_predict(np.asarray(rel_vectors))
    #X = np.zeros([s+1, clusters])
    #for source, node in zip(rel_sources, nodes):
    #    X[source, node] += 1
    #evaluate(MultinomialNB(), X)
    #print('repeat with -',names[0])
    #evaluate(classifiers[0], X)


