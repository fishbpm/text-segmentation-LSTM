# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 21:50:23 2018

@author: Fish
"""

import os 
import boto3

def pullBucketSamples(bucket, folder):
    root = '/samples'#'/data/signal' #remove the preceding  .  when pulling into docker container
    #s3 = boto3.resource('s3')
    #bucket = s3.Bucket('data.data-science.signal')
    #folder='summaries-segmentation'
    
    for obj in bucket.objects.filter(Prefix = folder):
        #do not write the top AWS folder object
        #(not sure why, but only this top folder resides as an additional object, other sub-folders do not)
        #(it has something to do with the S3 GUI?? - only the top folders are rendered on the GUI)
        if obj.key[-1] != '/':
            path, filename = os.path.split(obj.key)
            try:
                #first file (in each folder) initialises new local dir
                #after that we need to supress dir exists error
                os.makedirs(root + '/' + path[len(folder):])
            except OSError:#IOError:#FileExistsError:
                pass
            #now write the s3 file object into local folder, using the identical s3 filename
            bucket.Object(obj.key).download_file(root + obj.key[len(folder):])
    return
        
if __name__ == '__main__':
    s3 = boto3.resource('s3')
    mybucket = s3.Bucket('data.data-science.signal')
    myfolder = 'summaries-segmentation'
    pullBucketSamples(mybucket, myfolder+'/samples')