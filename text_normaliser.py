# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:08:25 2018

@author: Fish
"""

import re
import os
import random
from PIL import Image
import numpy as np
import xlsxwriter as excel
import pickle as pkl

#section_delimiter = "========"
#path = 'C:\Users\Fish\Documents\GitHub\datasets\wiki_727\test\00\00_test'

#workbook = excel.Workbook('output_scaled.xlsx')
#worksheet = workbook.add_worksheet()
#
#LSTM_output = pkl.load(open('LSTM_probs.pkl', 'rb'))
#articles = LSTM_output['probs']

#HEIGHT = 17
#WIDTH = 1
#THRESHOLD = 0.05
#NUM_FEATURES = 40
#PAD = 0.0012
#scaled_articles = []
pads = []

def get_target_preds(source_preds, target_size, last):
    if target_size == len(source_preds):
        return source_preds
    else:
        if len(source_preds) == 1:
            if last == True:
                target_preds = [random.choice(pads) for boundary in range(0, target_size - 1)]
                target_preds.extend(source_preds)
                return target_preds
            else:
                source_preds.extend([random.choice(pads) for boundary in range(0, target_size - 1)])
                return source_preds
        else:
            imaged = Image.fromarray(np.asarray([source_preds]))
            rescaled = np.asarray(imaged.resize((target_size, 1), Image.BICUBIC))
            target_preds = rescaled.tolist()
            return target_preds[0]
            #for boundary in enumerate(sentences[0]):
                #target.append(boundary)        

def normalise(article, NUM_FEATURES, THRESHOLD):
    
    boundaries = []
    zeros = []
    segments = []
    audit = []
    num_boundaries = len(article)
    tot_pos_preds = len([boundary for boundary in article if boundary > THRESHOLD])
    neg_preds = 0
    num_segments = 0
    target_preds = 0
    start_pos = 0
    source_seg_size = 0
    target_seg_size = 0
    
    if num_boundaries == NUM_FEATURES:
        return article
    else: #first, we need to know number of segments (only relevant in case where we are expanding array)
        new_segment = True
        for boundary, probability in enumerate(article):
            if probability > THRESHOLD:
                new_segment = True
            else:
                if new_segment:
                    num_segments += 1
                    new_segment = False
                #the first and last segments are  not representative of the intra-doc probabilities
                if boundary > 0 and boundary < (len(article) - 1):
                    if probability < 0.005 and probability > 0.0001:
                        counter = 0
                        while counter < len(pads) and pads[counter] != probability:
                            counter += 1
                        if counter == len(pads):
                            pads.append(probability)
        
        if num_boundaries > (NUM_FEATURES + 10):  
            boundaries.extend(get_target_preds(article, NUM_FEATURES, last=False))
        else:
            if num_boundaries < NUM_FEATURES:
                extra_segments = min(tot_pos_preds + 1 - num_segments, NUM_FEATURES - num_boundaries)
                num_boundaries += extra_segments
            
            ratio = float((NUM_FEATURES - tot_pos_preds)/(num_boundaries - tot_pos_preds))
            surplus = float(ratio - int(ratio))
            num_segments = 0
          
            for boundary, probability in enumerate(article):
                if len(article) < NUM_FEATURES:
                    if probability < THRESHOLD:
                        neg_preds += 1
                        #source_seg_size += 1
                        if neg_preds > (num_boundaries - tot_pos_preds)*surplus:
                            target_seg_size += int(ratio)
                            zeros.append([target_preds + target_seg_size, probability])
                        else:
                            target_seg_size += (1 + int(ratio))
                    else:
                        if target_seg_size > 0:
                            boundaries.extend(get_target_preds(article[start_pos:boundary], target_seg_size, boundary==len(article)))
                            #audit.append([num_segments, boundary, target_preds + target_seg_size + 1, True])
                            #num_segments += 1
                        elif num_segments < extra_segments:
                            target_seg_size = int(ratio)
                            neg_preds += 1
                            zeros.append([target_preds + target_seg_size, random.choice(pads)])
                            boundaries.extend([random.choice(pads) for boundary in range(0, target_seg_size)])
                            #audit.append([num_segments, boundary, target_preds + target_seg_size + 1, False])
                            num_segments += 1
                        boundaries.append(probability)#.extend(append_target_preds(article[boundary], 1))
                        target_preds += (target_seg_size + 1)
                        start_pos = boundary + 1
                        #source_seg_size = 0
                        target_seg_size = 0
                else:
                    if probability < THRESHOLD:
                        #neg_preds += 1
                        source_seg_size += 1
                    else:
                        if start_pos < boundary: #or equivalently if source_seg_size > 0
                            if (num_segments + 1) > tot_pos_preds*surplus:# and boundary < len(article):
                                target_seg_size = int(source_seg_size*surplus)
                            else:
                                target_seg_size = 1 + int(source_seg_size*surplus)
                        if target_seg_size == 0:
                            zeros.append([target_preds, article[boundary-1]])
                        else:
                            segments.append([target_preds + target_seg_size, target_seg_size])
                            num_segments += 1
                            #neg_preds += target_seg_size
                            boundaries.extend(get_target_preds(article[start_pos:boundary], target_seg_size, boundary==len(article)))
                        boundaries.append(probability)#.extend(append_target_preds(article[boundary], 1))
                        target_preds += (target_seg_size + 1)
                        start_pos = boundary + 1 #start the next segmenet (if any)
                        source_seg_size = 0
                        target_seg_size = 0
            
            if target_seg_size > 0: #if a final segment was still building built
                boundaries.extend(get_target_preds(article[start_pos:len(article)], target_seg_size, last=True))
                target_preds += target_seg_size
            elif source_seg_size > 0:
                target_seg_size = max(1, int(source_seg_size*surplus))
                boundaries.extend(get_target_preds(article[start_pos:len(article)], target_seg_size, last=True))
                target_preds += target_seg_size
                
            assert target_preds == len(boundaries)
           
            if target_preds > NUM_FEATURES:
                audit.append([target_preds, 0])
                segments.sort(key=lambda x: int(x[1]))       
                counter = -1 
                while target_preds > NUM_FEATURES:
                    boundaries[segments[counter][0] - 1] = 0 #mark for deletion
                    target_preds -= 1
                    counter -= 1
                deletions = 0
                for step in range(0, len(boundaries)):
                    if boundaries[step - deletions] == 0:
                        del boundaries[step - deletions]
                        deletions += 1
                audit[-1][1] = deletions
            elif target_preds < NUM_FEATURES:
                counter = 0 
                while target_preds < NUM_FEATURES:
                    boundaries.insert(zeros[counter][0] + counter - 1, zeros[counter][1])
                    target_preds += 1
                    counter += 1  
                
        return boundaries


if __name__ == '__main__':
    
    workbook = excel.Workbook('output_test.xlsx')
    worksheet = workbook.add_worksheet()
    
    LSTM_output = pkl.load(open('LSTM_probs.pkl', 'rb'))
    articles = LSTM_output['probs']
    
    for a, article in enumerate(articles):
    #for article, sentences in enumerate(scaled_articles):
        for sentence, probability in enumerate(normalise(article, 40, 0.05)):
            worksheet.write(sentence, a, abs(probability))
    
    #for article in articles:
    #    
    #    imaged = Image.fromarray(np.asarray([article]))
    #    rescaled = np.asarray(imaged.resize((HEIGHT, WIDTH), Image.LANCZOS))
    #    boundaries.append(rescaled.tolist())
    #    
    #for article, sentences in enumerate(boundaries):
    #    for sentence, probability in enumerate(sentences[0]):
    #        worksheet.write(sentence, article, probability)
    
    workbook.close()



















