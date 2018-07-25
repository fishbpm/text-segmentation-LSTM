# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:08:25 2018

@author: Fish
"""

import re
import os
import random
import numpy as np
import xlsxwriter as excel
import pickle as pkl
from text_normaliser import normalise

from keras.layers import Input, MaxPooling1D, Conv1D, Dense, Flatten, Dropout#, PReLU
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.utils import np_utils, to_categorical

LSTM_output = pkl.load(open('LSTM_probs.pkl', 'rb'))
articles = LSTM_output['probs']

#HEIGHT = 17
#WIDTH = 1
THRESHOLD = 0.05 # all boundaries above this will be preserved on normalisation
NUM_FEATURES = 40
dataset = []
#labels = []

for a, article in enumerate(articles):
#for article, sentences in enumerate(scaled_articles):
    if a < 129 or a > 141:
        boundaries = [a] #index
        for probability in normalise(article, NUM_FEATURES, THRESHOLD):
            boundaries.append([probability])
        #boundaries.append([random.randint(0, 1)])
        boundaries.append([1] if a < 129 else [0]) #label
        dataset.append(boundaries)
        
for step in range(0, 25):
    random.shuffle(dataset)
#labels.append(1 if a < 129 else 0)
 
X = np.asarray([boundaries[1:-1] for boundaries in dataset]) # excluding index and labels
labels = np.asarray([label[-1][0] for label in dataset])
y = to_categorical(labels, num_classes=2)
print('features have shape -', X.shape)
print('labels have shape -', y.shape)

BATCH_SIZE = 49
EPOCHS = 100

"""--------------construct and initialise CNN model----------------------------- """
inputs = Input(shape=(40, 1))
#x = Dense(64, activation='relu')(inputs)
x = Conv1D(24, kernel_size=5, activation='relu')(inputs)
#x = PReLU()(x) # Non-linearity
x = MaxPooling1D(pool_size=2, strides=1)(x)
x = Dropout(rate=0.2)(x)
#x = Conv2D(36, kernel_size=5, activation='relu')(x)
#x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(rate=0.2)(x)
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

history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    verbose=1, validation_split = 0.2)#validation_data=(mnist_test, mnist.test.labels))
#score = model.evaluate(mnist_test, mnist.test.labels, verbose=0)





















