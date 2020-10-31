#### Import all the supporting classes

import argparse
import os
import pickle
import time

from model import ConvEParam
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

torch.backends.cudnn.enabled = False

from logger import config_logger

logger = config_logger('Model')


def get_glove_entity_dim(args):
    glove_embedding = pickle.load(open(args.data_files['glove_embedding_path'], 'rb'))
    dim = glove_embedding[0].shape[1]

    return int(dim)


def get_elmo_entity_dim(args):
    elmo_embedding = pickle.load(open(args.data_files['elmo_embedding_path'], 'rb'))

    temp = np.mean(elmo_embedding[0], 1)

    dim = temp.shape[1]

    return int(dim)


def get_entity_len_and_num(args):
    fin = open(args.data_files['ent2id_path'], 'r', encoding="utf8").readlines()
    for trip in fin[0:]:
        record = trip.strip().split()

    return int(record[1]), int(record[0])


def get_rel_len_and_num(args):
    fin = open(args.data_files['rel2id_path'], 'r', encoding="utf8").readlines()
    for trip in fin[0:]:
        record = trip.strip().split()

    return int(record[1]), int(record[0])


def pad_glove_all_entity_embedding(args):
    glove_embedding = pickle.load(open(args.data_files['glove_embedding_path'], 'rb'))
    pad_all_entity = []
    for _, entity_embedding in enumerate(glove_embedding):
        size = len(entity_embedding)
        pad_glove_entity = np.pad(entity_embedding, ((0, args.max_entity_length - size), (0, 0)), mode='constant')
        pad_all_entity.append(pad_glove_entity)

    return pad_all_entity


def pad_elmo_all_entity_embedding(args):
    elmo_embedding = pickle.load(open(args.data_files['elmo_embedding_path'], 'rb'))
    pad_all_entity = []
    for _, entity_embedding in enumerate(elmo_embedding):
        temp = np.mean(entity_embedding, 1)
        size = len(temp)
        pad_glove_entity = np.pad(temp, ((0, args.max_entity_length - size), (0, 0)), mode='constant')
        pad_all_entity.append(pad_glove_entity)

    return pad_all_entity


def init_args():
    parser = argparse.ArgumentParser(description='CaRe: Canonicalization Infused Representations for Open KGs')

    ### Model and Dataset choice
    parser.add_argument('-CN', dest='CN', default='RGCN', choices=['Linear', 'GCN', 'LAN', 'RGCN'],
                        help='Choice of Canonical Cluster Encoder Network')
    parser.add_argument('-GcnNum', dest='GcnNum', default=1, type=int, choices=[1, 2],
                        help='The number of Gcn Convolution')
    parser.add_argument('-dataset', dest='dataset', default='Cockpit', choices=['Cockpit'],
                        help='Dataset Choice')

    ### Data Paths
    parser.add_argument('-data_path', dest='data_path', default='../Data', help='Data folder')

    #### Hyper-parameters
    parser.add_argument('-num_layers', dest='num_layers', default=1, type=int, help='No. of layers in encoder network')
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
    parser.add_argument('--use_glove', type=bool, default=True, help='Using Glove embedding or Elmo')
    parser.add_argument("--n_bases", type=int, default=4)

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
    }

    args.max_entity_length, args.ent_total = get_entity_len_and_num(args)
    args.max_rel_length, args.rel_total = get_rel_len_and_num(args)

    if args.use_glove:
        args.input_dim = get_glove_entity_dim(args)
    else:
        args.input_dim = get_elmo_entity_dim(args)

    args.nfeats = 300
    args.pad_id = 0

    args.model_path = "ConvE" + "-" + args.CN + "_modelpath.pth"

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    args.use_cuda = False

    return args


def get_triple_data(triple_file, is_use_cuda):
    fin = open(triple_file, "r", encoding="utf8").readlines()
    trip_list = []
    for trip in fin[0:]:
        temp = trip.strip().split()
        e1, r, e2 = int(temp[0]), int(temp[1]), int(temp[2])
        trip_list.append([e1, r, e2])

    tripls = np.array(trip_list)

    tripls = Variable(torch.from_numpy(tripls))
    if is_use_cuda:
        tripls = tripls.cuda()

    return tripls


if __name__ == '__main__':
    args = init_args()

    if args.use_glove:
        pad_entity_embedding = pad_glove_all_entity_embedding(args)
    else:
        pad_entity_embedding = pad_elmo_all_entity_embedding(args)

    for item in vars(args).items():
        logger.info('%s : %s', item[0], str(item[1]))

    rel2words = np.load(args.data_files['relation2word_path'])
    embed_matrix = np.load(args.data_files["word_glove_path"])
    entity_embedding = torch.Tensor(pad_entity_embedding)
    edge_index = torch.tensor(np.load(args.data_files['graph_edges_path']), dtype=torch.long)
    edge_type = torch.tensor(np.load(args.data_files['edges_type_path']), dtype=torch.long)
    edge_norm = edge_normalization(edge_type, edge_index, args.ent_total, args.rel_total)
    edge_norm = torch.tensor(edge_norm)

    model = ConvEParam(args, embed_matrix, rel2words)

    logger.info(model)

    if args.use_cuda:
        model.cuda()
        entity_embedding = entity_embedding.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()

    model_state_file = args.model_path

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_triple = get_triple_data(args.data_files['train_trip_path'], args.use_cuda)
    valid_triple = get_triple_data(args.data_files['valid_trip_path'], args.use_cuda)
    test_triple = get_triple_data(args.data_files['test_trip_path'], args.use_cuda)

    train_id = np.arange(len(train_triple))

    best_MR = 20000
    best_MRR = 0
    best_epoch = 0
    count = 0
    label_graph = np.load(args.data_files['train_label_path'])
    for epoch in range(args.n_epochs):
        model.train()
        if count >= args.early_stop: break
        epoch_loss = 0
        permute = np.random.permutation(train_id)
        train_id = train_id[permute]
        n_batches = train_id.shape[0] // args.batch_size
        t1 = time.time()
        for i in range(n_batches):
            id_list = train_id[i * args.batch_size:(i + 1) * args.batch_size]
            samples, labels = get_next_batch(id_list, label_graph, args, train_triple)

            optimizer.zero_grad()
            loss = model.get_loss(samples, labels, entity_embedding, edge_index, edge_type, edge_norm)
            loss.backward()
            logger.info("batch {}/{} batches, batch_loss: {}".format(i, n_batches, (loss.data).cpu().numpy()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            epoch_loss += (loss.data).cpu().numpy()
        logger.info("epoch {}/{} total epochs, epoch_loss: {}".format(epoch + 1, args.n_epochs, epoch_loss / n_batches))

        if (epoch + 1) % args.eval_epoch == 0:
            model.eval()
            MR, MRR = evaluate(model, args.ent_total, valid_triple, args, entity_embedding, label_graph, edge_index, edge_type, edge_norm)
            if MRR > best_MRR or MR < best_MR:
                count = 0
                if MRR > best_MRR: best_MRR = MRR
                if MR < best_MR: best_MR = MR
                best_epoch = epoch + 1
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            else:
                count += 1
            logger.info(
                "Best Valid MRR: {}, Best Valid MR: {}, Best Epoch(MR/MRR): {}".format(best_MRR, best_MR, best_epoch))
            scheduler.step(best_epoch)

    ### Get Embeddings
    logger.info("Test Set Evaluation ---")
    checkpoint = torch.load(model_state_file)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    _, _ = evaluate(model, args.ent_total, test_triple, args, entity_embedding, label_graph, edge_index, edge_type, edge_norm)
