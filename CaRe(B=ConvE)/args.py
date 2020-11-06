# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/11/2 15:21
@file: args.py
@desc: 
"""
import argparse

import torch

from utils import get_entity_len_and_num, get_rel_len_and_num, get_glove_entity_dim, get_elmo_entity_dim


def init_args(is_dataset = False):
    parser = argparse.ArgumentParser(description='CaRe: Canonicalization Infused Representations for Open KGs')

    ### Model and Dataset choice
    parser.add_argument('-CN', dest='CN', default='RGCN', choices=['Linear', 'GCN', 'LAN', 'RGCN', 'GAT'],
                        help='Choice of Canonical Cluster Encoder Network')
    parser.add_argument('-dataset', dest='dataset', default='Cockpit', choices=['Cockpit'],
                        help='Dataset Choice')

    ### Data Paths
    parser.add_argument('-data_path', dest='data_path', default='../Data', help='Data folder')
    parser.add_argument('-elmo_model_path', dest='elmo_model_path', default='../Elmo/zhs.model/',
                        help='Elmo model folder')
    parser.add_argument('-glove_path', dest='glove_path', default='../Glove/wiki.zh.glove.300d.txt',
                        help='glove file')

    #### Hyper-parameters
    parser.add_argument('-num_layers', dest='num_layers', default=1, type=int, help='No. of layers in encoder network')
    parser.add_argument('-nheads', dest='nheads', default=3, type=int, help='multi-head attantion in GAT')
    parser.add_argument('-bidirectional', dest='bidirectional', default=True, type=bool, help='type of encoder network')
    parser.add_argument('-relPoolType', dest='relPoolType', default='last', choices=['last', 'max', 'mean'],
                        help='pooling operation for encoder network')
    parser.add_argument('-entPoolType', dest='entPoolType', default='mean', choices=['max', 'mean'],
                        help='pooling operation for encoder network')
    parser.add_argument('-dropout', dest='dropout', default=0.5, type=float, help='Dropout')
    parser.add_argument('-lr', dest='lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-batch_size', dest='batch_size', default=1024, type=int, help='batch size for training')
    parser.add_argument('-n_epochs', dest='n_epochs', default=500, type=int, help='maximum no. of epochs')
    parser.add_argument('-grad_norm', dest='grad_norm', default=1.0, type=float, help='gradient clipping')
    parser.add_argument('-eval_epoch', dest='eval_epoch', default=5, type=int,
                        help='Interval for evaluating on validation dataset')
    parser.add_argument('-Hits', dest='Hits', default=[10, 30, 50], help='Choice of n in Hits@n')
    parser.add_argument('-early_stop', dest='early_stop', default=10, type=int,
                        help='Stopping training after validation performance stops improving')
    parser.add_argument('-use_glove', type=bool, default=True, help='Using Glove embedding or Elmo')
    parser.add_argument("-n_bases", type=int, default=4)
    parser.add_argument("-predict_num", type=int, default=10)

    args = parser.parse_args()

    args.data_files = {
        'dataset_path': args.data_path + '/' + args.dataset + '/dataset.txt',
        'origin_path': args.data_path + '/' + args.dataset + '/origin.txt',
        'ent2id_path': args.data_path + '/' + args.dataset + '/ent2id.txt',
        'rel2id_path': args.data_path + '/' + args.dataset + '/rel2id.txt',
        'train_trip_path': args.data_path + '/' + args.dataset + '/train_trip.txt',
        'test_trip_path': args.data_path + '/' + args.dataset + '/test_trip.txt',
        'valid_trip_path': args.data_path + '/' + args.dataset + '/valid_trip.txt',
        'origin_trip_path': args.data_path + '/' + args.dataset + '/origin_trip.txt',
        'elmo_embedding_path': args.data_path + '/' + args.dataset + '/elmo.preprocessed.pickle',
        'glove_embedding_path': args.data_path + '/' + args.dataset + '/glove.preprocessed.pickle',
        'word2id_path': args.data_path + '/' + args.dataset + '/word2id.txt',
        'word_glove_path': args.data_path + '/' + args.dataset + '/word.glove.pickle',
        'relation2word_path': args.data_path + '/' + args.dataset + '/relation2word.dict.pickle',
        'train_label_path': args.data_path + '/' + args.dataset + '/train.label.pickle',
        'graph_edges_path': args.data_path + '/' + args.dataset + '/edges.pickle',
        'edges_type_path': args.data_path + '/' + args.dataset + '/edges_type.pickle',
        'predict_path': args.data_path + '/' + args.dataset + '/Predict' + '/predict.txt',
        'result_path': args.data_path + '/' + args.dataset + '/Predict' + '/result.txt',
        'model_path': args.data_path + '/' + args.dataset + '/Model' + '/' + args.CN + "_modelpath.pth",
        'parameter_path': args.data_path + '/' + args.dataset + '/Model' + '/' + args.CN + "_parameter.txt",
    }

    if is_dataset == True:
        return args

    args.max_entity_length, args.ent_total = get_entity_len_and_num(args)
    args.max_rel_length, args.rel_total = get_rel_len_and_num(args)

    if args.use_glove:
        args.input_dim = get_glove_entity_dim(args)
    else:
        args.input_dim = get_elmo_entity_dim(args)

    args.nfeats = 300
    args.pad_id = 0

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    return args
