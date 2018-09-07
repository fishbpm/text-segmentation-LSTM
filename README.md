# Identifying Summary Articles

This repository contains code and supplementary materials which are required to first train and test an LSTM boundary predictor, then use the softmax test predictions as features to train a CNN article classifier.

The model has not yet been formally published, this content is a work-in-progress project for MSc dissertation. Full instructions will be provided upon publication
(following is just a draft o

## Download required resources

signal-1000K, signal-100K datasets:
>  To be provided

word2vec:
>  https://drive.google.com/a/audioburst.com/uc?export=download&confirm=zrin&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

Fill relevant paths in configgenerator.py to access these resources, and execute the script

## Creating an environment for LSTM Boundary Prediction:

    conda create -n boundpred python=2.7 numpy scipy gensim ipython 
    source activate boundpred
    pip install http://download.pytorch.org/whl/cu80/torch-0.3.0-cp27-cp27mu-linux_x86_64.whl 
    pip install tqdm pathlib2 segeval tensorboard_logger flask flask_wtf nltk
    pip install pandas xlrd xlsxwriter termcolor

## Creating an environment for CNN Article Prediction:

    conda create -n tensorflow python=3.5 numpy scipy gensim ipython 
    pip install --upgrade tensorflow
    etc. (to be provided)
    
## How to run training process?

    python run.py --help

Example:

    python run.py --cuda --model max_sentence_embedding --wiki

## How to run a test cycle to generate boundary predictions?

    python test_accuracy.py  --help

Example:

    python test_accuracy.py --cuda --model <path_to_model> --wiki

More to follow..   reiterate: above is a draft while MSc dissertation is still underway

