import json

jsondata = {
    "word2vecfile": "/mnt/c/Users/Fish/Documents/GitHub/datasets/word2vec/GoogleNews-vectors-negative300.bin",
    "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
    "wikidataset": "/mnt/c/Users/Fish/Documents/GitHub/datasets/wiki_727",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)
