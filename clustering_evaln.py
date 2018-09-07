# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:46:38 2018

@author: Fish
"""

import os
import json
import re
import io
import math
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

from text_normaliser import normalise

PATH = 'C:/Users/Fish/outputs/graphseg/'#testing/'#'#TEMP/'#
DATASETS = 'C:/Users/Fish/Documents/GitHub/datasets/'
SEPARATOR = "=========="
NUM_ARTICLES = 1000 #number of articles we wish to synthesise
RESOURCES = 'C:/Users/Fish/Documents/GitHub/graphseg/source/res/'#stopwords.txt'
CONTENT = ['ADJ', 'VERB', 'ADV', 'NOUN', 'PROPN', 'PRON', 'INTJ']#these are POStags - high level
#CONTENT = ['NOUN', 'PROPN'] #need to re-set this to the line above (this is for testing ideas)
THRESHOLD = 0.4 #segmentation threshold

samples = []
filenames = []

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

#!!!!this should be callable functions in a util module
with open(RESOURCES+'stopwords.txt', 'r') as in_file:
    stop_words = in_file.read().splitlines()
    in_file.close()

####!!!! latest pkls are all in a certain order as per labels/indexs in docker output file
#this section is only needed for boundary evaln (Pk or homogeneity score)
###!!!these could also be used to generate an average softmax feature which we could add to regressor
"""****************************load LSTM softmax output************************************************************"""
NUM_FEATURES = 40
THRESHOLD = 0.05 # all boundaries above this will be preserved on normalisation
SEG_THRESHOLD = 0.4
MAX_PAD = 0.005 
OUTPUTS = 'C:/Users/Fish/docker/OUTPUTS/Testing/wiki_700K/'
LSTM_output = pkl.load(open(OUTPUTS+'wiki700K_CPU_softmax_signal_400.pkl', 'rb'))
probabilities = LSTM_output['probs']

boundaries = []
for a, article in enumerate(probabilities):
    boundaries.append([abs(probability) for probability in article])#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)])

labels= {}
softmax = {}
#with open(DATASETS+'signal_working/TESTING/signal400/labels_JSON.txt', 'r') as in_file: 
with open(DATASETS+'signal_working/TESTING/signal400/labels_CPU.txt', 'r') as in_file: #sorted by file input order(OLD) - as per Graphseg output
    for a, annotation in enumerate(in_file.read().splitlines()):
        article, label = annotation.split('\t')
        labels[article.strip()] = 0 if (str(label) == 'FALSE') else 1
        softmax[article.strip()] = [sum(boundaries[a])/float(len(boundaries[a])),
                len([b for b in boundaries[a] if b > SEG_THRESHOLD])/float(len(boundaries[a]))] #pred metrics are now accessible via index 
    in_file.close()
"""****************************load Graphseg similarity output************************************************************"""    
NUM_FEATURES = 40
THRESHOLD = 0.3 # all boundaries above this will be preserved on normalisation
SEG_THRESHOLD = 0.4
MAX_PAD = 0.05  
OUTPUTS = 'C:/Users/Fish/outputs/graphsim/'#graphseg/'#TEMP/'#
GRAPHSEG_output = pkl.load(open(OUTPUTS+'GRAPHSIM_w6ent_FULL.pkl', 'rb'))
sims = GRAPHSEG_output['probs']

boundaries = []
for a, article in enumerate(sims):
    boundaries.append([abs(probability) for probability in article])#normalise(article, NUM_FEATURES, THRESHOLD, MAX_PAD)]) #deos this need to be a numpy array?

graphsim = {}
true_count = 0 
LABELS = 'C:/Users/Fish/Documents/GitHub/datasets/signal_working/TESTING/signal400/'
with open(LABELS+'labels_FILE.txt', 'r') as in_file: #NOTE - gold annotations must be in FILE order and discluding the problem articles
#with open(LABELS+'labels_FILE.txt', 'r') as in_file:#with open(OUTPUTS+'labels.txt', 'r') as in_file:
    for a, annotation in enumerate(in_file.read().splitlines()):
        article, label = annotation.split('\t')
        graphsim[article.strip()] = [sum(boundaries[a])/float(len(boundaries[a])),
                 len([b for b in boundaries[a] if b > SEG_THRESHOLD])/float(len(boundaries[a]))] #pred metrics are now accessible via index
        true_count += (1 if label =='TRUE' else 0)
    in_file.close()

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
#tokens = [] #set of words in each document
vectors = [] #all vectors across the whole corpus
sources = [] #source document of each vector
ent_sources = []
ent_vectors = [] #all vectors across the whole corpus
doc_vectors = [] #set of vectors representing all words in each document
ent_doc_vectors = [] #set of vectors representing all entities in each document
rel_vectors = []
rel_sources = []
ent_distances = []
ent_dist_matrix = [] #rel.distances as an array, to try with BOW classifier
num_clusters = []
num_vectors = []
features = [] #unsupervised (cohesion & separation) / supervised (homogeneity)
vfeatures = [] #this one is for calinski harabaz
targets = [] #gold annotations
probs = [] #boundary probs from LSTM softmax
similarities = []

for s, sample in enumerate(get_files(PATH)[:2*true_count]):
    index =  os.path.split(sample)[1]
    print('processing', index,'...')
    distances = [] #vector distances between each pair of words in the document
    segments = [] #un-supervised linear clusters (0 > N, for each word in the document)
    tokens = [] #tokens.append([])
    entities = []
    sample_vectors = [] #original plan inidividual vector each word, but NB classifier requires single vector for each sample article
    size = 0 #vectors in this article
    doc_vectors.append(np.zeros(300))
    exceptions.append([])
    features.append([])
    vfeatures.append([])
    ent_dist_matrix.append(np.zeros(20)) #similarities between 0 and 2 in 0.1 increments
    targets.append(labels[index])
    probs.append(softmax[index])
    similarities.append(graphsim[index])
    #in_file = io.open(sample, 'r') 
    #files.append(in_file) #tfidf vectorizer says it accepts a list of file objects, but it FAILS 
    #with codecs.open(sample, 'r', 'ISO-8859-1') as in_file: #ISO encoding also works, but io is just as good if not better
    with io.open(sample, 'r') as in_file: #utf-8 decoding doesnt work on outputs from the java engine
        raw_content = in_file.read().strip()
    files.append(raw_content)
    entity_files.append('') #start with empty string for this file
    #clean_txt = content.decode('utf-8').strip()
    clusters = [s.strip('\n') for s in re.split(SEPARATOR, raw_content) if len(s.strip('\n')) > 0]#content.decode('utf-8').strip())]
    boundaries = 0
    for c, cluster in enumerate(clusters):
        boundaries += len(cluster.splitlines())#this is just to get the #sentences for cluster normalisation
        #boundaries = raw_content.split('\n') #should return the identical list
        content = re.sub("[^a-zA-Z0-9\s,\.']+",'',re.sub('-',' ',cluster)).strip()
        for token in nlp(content):
            word = token.text.lower()#re.sub('\W','',token.text.lower())#apostrophes (like don't) have been stripped from freq resource
            cleansed = re.sub('\d','#',re.sub("[^\w']",'',token.text.lower()))#retain commas as word2vec includes don't etc.
            if len(cleansed) > 0 and not word.isnumeric() and word not in stop_words and \
            token.lemma_ not in stop_words and not token.is_stop and (token.pos_ in CONTENT or token.ent_iob_ != 'O'):
                try:
                    sample_vectors.append(word2vec[cleansed])#cphrase[cleansed])#
                except:
                    exceptions[s].append(cleansed)
                else:
                    doc_vectors[s] += sample_vectors[size]#word2vec[cleansed] 
                    tokens.append(cleansed)#tokens[s].append(cleansed)
                    vectors.append(sample_vectors[size]) #only needed for kmeans clustering
                    segments.append(c)
                    distances.append([])
                    sources.append(s)
                    if token.ent_iob_ != 'O':
                        entity_files[s] = entity_files[s] + ' ' + cleansed
                        ent = 0
                        while ent < len(entities) and entities[ent][0] != cleansed:
                            ent += 1
                        if ent < len(entities):
                            entities[ent][2] += 1
                        else:
                            entities.append([cleansed, sample_vectors[size], 1])
                            ent_vectors.append(sample_vectors[size]) #only needed for kmeans clustering
                            ent_sources.append(s)
                    size += 1
                        
    for t1, token_1 in enumerate(tokens):#(tokens[s]):
        for t2, token_2 in enumerate(tokens):#(tokens[s]):
            if t1 == t2:
                distances[t1].append(0)
            else:
                try:
                    distances[t1].append(word2vec.similarity(token_1, token_2))
                except:
                    print('distance fail:',token_1 ,'/', token_2)
                    
    cleansed_articles.append(' '.join(tokens).strip())#(tokens[s]).strip())

###!!!!! this section must be restricted to entity tokens only --------
    ent_doc_vectors.append(np.zeros(300))
    ent_distances.append([0, 0]) #start with zero distance & frequency
    if len(entities) > 0:
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
        
    num_vectors.append(size)
    num_clusters.append(c+1)                      
    if c == 0:
        features[s].append(0) #only one segment so use avg. sillhouette score by default??
        vfeatures[s].append(0)
    else:
        y = np.asarray(segments)
        X = np.asarray(sample_vectors)
        vfeatures[s].append(metrics.calinski_harabaz_score(X, y))
            #the ch_score accepts lists, silhouette (below) requires nump arrays
        X = np.asarray(distances)
        features[s].append(10*metrics.silhouette_score(X, y, metric='euclidean'))
    vfeatures[s].append((c+1) / boundaries) 
    features[s].append((c+1) / boundaries) #normalise #cluster by #sentences in sample
    #features[s].append(ent_distances[s]) #can also try including this in main feature array
                    
    #homogeneity (and Pk) can only be measured on synthetic set
    #score[1].append(metrics.homogeneity_score(labels_true, predictions[s])

names = ["Logistic Regression", "Linear SVM"]
#names = ["Multinomial NB", "Logistic Regression", "Linear SVM", "RBF SVM"]

classifiers = [LogisticRegression(), SVC(kernel="linear", C=0.025)]
#classifiers = [MultinomialNB(), LogisticRegression(),
#               SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1)]

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.6)
X = vectorizer.fit_transform(files)
y = np.asarray(targets) #target is the same for all classifiers

indices = []
skf = StratifiedKFold(n_splits=5, shuffle=True)
#ensure we are using the same folds for every experiment
for train, test in skf.split(X, y): # consider the first 30 examples
    indices.append([train, test])
    
def evaluate(model, X):
    score = []
    print(X.shape)
    for fold in indices: # consider the first 30 examples
        model.fit(X[fold[0]], y[fold[0]])
        score.append(model.score(X[fold[1]], y[fold[1]]))
    print('Avg Acc:', np.mean(score))

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

for model, name in zip(classifiers, names):

    print('using hashed count vectors -',name)
    hasher = HashingVectorizer(n_features=1000, stop_words='english', norm = 'l2')
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    X = vectorizer.fit_transform(files)
    evaluate(model, X)    
    
    print('using avg softmax -',name)
    X = np.asarray(probs)
    evaluate(model, X)
    
    print('now try with each feature in turn -',name)
    for feature in range(X.shape[1]):
        X = np.asarray([[f[feature]] for f in probs])
        evaluate(model, X)
        
    print('using avg similarity -',name)
    X = np.asarray(similarities)
    evaluate(model, X)

    print('now try with each feature in turn -',name)
    for feature in range(X.shape[1]):
        X = np.asarray([[f[feature]] for f in similarities])
        evaluate(model, X)

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

