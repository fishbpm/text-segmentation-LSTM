import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure, log_value
import os
import sys
from pathlib2 import Path
from wiki_loader import WikipediaDataSet
import accuracy
import numpy as np
from termcolor import colored

from timeit import default_timer as timer
import boto3
import io

torch.multiprocessing.set_sharing_strategy('file_system')

preds_stats = utils.predictions_analysis()


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()


class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        current_idx = 0
        for k, t in enumerate(targets_np):
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)

            for threshold in self.thresholds:
                output = ((output_np[current_idx: to_idx, :])[:, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

            current_idx = to_idx

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold


def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break

                pbar.update()
                model.zero_grad()
                output = model(data)
                target_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                loss = model.criterion(output, target_var)
                loss.backward()

                optimizer.step()
                total_loss += loss.data[0]
                # logger.debug('Batch %s - Train error %7.4f', i, loss.data[0])
                pbar.set_description('Training, loss={:.4}'.format(loss.data[0]))
            # except Exception as e:
                # logger.info('Exception "%s" in batch %s', e, i)
                # logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
                # pass

    total_loss = total_loss / len(dataset)
    logger.debug('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))
    log_value('Training Loss', total_loss, epoch + 1)


def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validating', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)

                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                acc.update(output_softmax.data.cpu().numpy(), target)


            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass

        epoch_pk, epoch_windiff, threshold = acc.calc_accuracy()

        logger.info('Validating Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                            preds_stats.get_accuracy(),
                                                                                                            epoch_pk,
                                                                                                            epoch_windiff,
                                                                                                            preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk, threshold


def test(model, args, epoch, dataset, logger, threshold):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        acc = accuracy.Accuracy()
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                current_idx = 0

                for k, t in enumerate(target):
                    document_sentence_count = len(t)
                    to_idx = int(current_idx + document_sentence_count)

                    output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                    h = np.append(output, [1])
                    tt = np.append(t, [1])

                    acc.update(h, tt)

                    current_idx = to_idx

                    # acc.update(output_softmax.data.cpu().numpy(), target)

            #
            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)

        epoch_pk, epoch_windiff = acc.calc_accuracy()

        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk,
                                                                                                          epoch_windiff,
                                                                                                          preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk


def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)

    configure(os.path.join('runs', args.expname))

    # Let's use Amazon S3
    s3 = boto3.resource('s3') #s3 = boto3.client('s3', profile_name='signal-rnd')
    mybucket = s3.Bucket('data.data-science.signal')
    myfolder = 'summaries-segmentation'
    
    start = timer()
    print 'Loading word vectors....'
    if not args.test:
        #key = myfolder + utils.config['word2vecfile']
        
        #word2vec = gensim.models.KeyedVectors.load_word2vec_format(mybucket.Object(key).get()['Body'].read(), binary=True)
        #word2vec = gensim.models.KeyedVectors.load_word2vec_format(io.BytesIO(mybucket.Object(key).get()['Body'].read()), binary=True)
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
        #response = urllib2.urlopen('https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing')
        #word2vec = gensim.models.KeyedVectors.load_word2vec_format(response.read(), binary=True)
        
        #mybucket.Object(key).download_file('GoogleNews_vectors')  
        #word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews_vectors', binary=True)
    else:
        word2vec = None

    word2vec_done = timer()
    print 'Loading word2vec ellapsed: ' + str(word2vec_done - start) + ' seconds'
    
    print 'Loading samples....'
    if not args.infer:
        if args.wiki:
            if (args.wiki_folder):
                signal_training = True
                dataset_path = args.wiki_folder#Path(args.wiki_folder)
                os.makedirs('.'+args.wiki_folder) #to keep the container tidy
            else:
                signal_training = False
                dataset_path = Path(utils.config['wikidataset'])
                
            #dataset_path = Path(utils.config['wikidataset'])
            train_dataset = WikipediaDataSet(str(dataset_path)+'/train', word2vec=word2vec, folder=signal_training,
                                             high_granularity=args.high_granularity)
            dev_dataset = WikipediaDataSet(str(dataset_path)+'/dev', word2vec=word2vec, folder=signal_training,
                                           high_granularity=args.high_granularity)
            test_dataset = WikipediaDataSet(str(dataset_path)+'/test', word2vec=word2vec, folder=signal_training,
                                            high_granularity=args.high_granularity)

        else:
            dataset_path = utils.config['choidataset']
            train_dataset = ChoiDataset(dataset_path, word2vec)
            dev_dataset = ChoiDataset(dataset_path, word2vec)
            test_dataset = ChoiDataset(dataset_path, word2vec)

        train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=True,
                              num_workers=args.num_workers)
        dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                            num_workers=args.num_workers)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)

    samples_done = timer()
    print 'Samples pulled successfully into container in: ' + str(samples_done - word2vec_done) + ' seconds'

    assert bool(args.model) ^ bool(args.load_from)  # exactly one of them must be set

    if args.model:
        model = import_model(args.model)
    elif args.load_from:        
        key = myfolder+args.load_from
        #model = torch.load(mybucket.Object(key).get()['Body'].read())
        #fileobj = io.BytesIO()
        #mybucket.Object(key).download_fileobj(fileobj)
        mybucket.Object(key).download_file('trained_model')
    
        #with open(args.load_from, 'rb') as f:
        with open('trained_model', 'rb') as f:
            model = torch.load(f)

    model.train()
    model = maybe_cuda(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if not args.infer:
        best_val_pk = 1.0
        for j in range(args.epochs):
            train(model, args, j, train_dl, logger, optimizer)
            model_name = 'model{:03d}.t7'.format(j)
            with (checkpoint_path / 'model{:03d}.t7'.format(j)).open('wb') as f:
                torch.save(model, f)

            val_pk, threshold = validate(model, args, j, dev_dl, logger)
            if val_pk < best_val_pk:
                test_pk = test(model, args, j, test_dl, logger, threshold)
                logger.debug(
                    colored(
                        'Current best model from epoch {} with p_k {} and threshold {}'.format(j, test_pk, threshold),
                        'green'))
                best_val_pk = val_pk
                with (checkpoint_path / 'best_model.t7'.format(j)).open('wb') as f:
                    torch.save(model, f)
        
        key = myfolder + '/results/trained_model.t7'    
        mybucket.Object(key).upload_file(str(checkpoint_path)+'/'+model_name)
        key = myfolder + '/results/best_model.t7'   
        mybucket.Object(key).upload_file(str(checkpoint_path)+'/best_model.t7')
        key = myfolder + '/results/train.log'   
        mybucket.Object(key).upload_file(str(checkpoint_path)+'/train.log')
        
    else:
        test_dataset = WikipediaDataSet(args.infer, word2vec=word2vec,
                                        high_granularity=args.high_granularity)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        print test(model, args, 0, test_dl, logger, 0.4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--wiki_folder', help='path to folder which contains wiki documents')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--infer', help='inference_dir', type=str)

    main(parser.parse_args())
