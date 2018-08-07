import json

#running from localhost
#jsondata = {
#    "word2vecfile": "/data/word2vec/GoogleNews-vectors-negative300.bin",
#    "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
#    "wikidataset": "/data/wiki_727", #we are not training with wiki, so this config is currently redundant
#}

#running from AWS - amazon web services
jsondata = {
    "word2vecfile": "/embeddings/GoogleNews-vectors-negative300.bin",
    "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
    "wikidataset": "/samples", #we are not training with wiki, so this config is currently redundant
}
    
#running from WSL - windows subsysetm for Linux
#jsondata = {
#    "word2vecfile": "/mnt/c/Users/Fish/Documents/GitHub/datasets/word2vec/GoogleNews-vectors-negative300.bin",
#    "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
#    "wikidataset": "/mnt/c/Users/Fish/Documents/GitHub/datasets/wiki_727",
#}
    
with open('config.json', 'w') as f:
    json.dump(jsondata, f)