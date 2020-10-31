import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch_scatter import scatter_add

warnings.filterwarnings("ignore")

from logger import config_logger

logger = config_logger('Model')


def seq_batch(phrase_id, args, phrase2word):
    phrase_batch = np.ones((len(phrase_id), args.max_rel_length), dtype=int) * args.pad_id
    phrase_len = torch.LongTensor(len(phrase_id))

    for i, ID in enumerate(phrase_id):
        phrase_batch[i, 0:len(phrase2word[int(ID)])] = np.array(phrase2word[int(ID)])
        phrase_len[i] = len(phrase2word[int(ID)])

    phrase_batch = torch.from_numpy(phrase_batch)
    phrase_batch = Variable(torch.LongTensor(phrase_batch))
    phrase_len = Variable(phrase_len)

    if args.use_cuda:
        phrase_batch = phrase_batch.cuda()
        phrase_len = phrase_len.cuda()

    return phrase_batch, phrase_len


def get_next_batch(id_list, label_graph, args, train):
    entTotal = args.ent_total
    samples = []
    labels = np.zeros((len(id_list), entTotal))
    for i in range(len(id_list)):
        trip = train[id_list[i]]
        head = int(trip[0])
        rel = int(trip[1])
        samples.append([head, rel])
        pos_ids = list(label_graph[(head, rel)])
        labels[i][pos_ids] = 1

    samples = Variable(torch.from_numpy(np.array(samples)))
    labels = Variable(torch.from_numpy(labels).float())

    if args.use_cuda:
        samples = samples.cuda()
        labels = labels.cuda()

    return samples, labels


def get_rank(scores, clust, Hits, pos_ids):
    hits = np.ones((len(Hits)))
    scores = np.argsort(scores)

    rank = 1
    for i in range(scores.shape[0]):
        if scores[i] in clust:
            break
        elif scores[i] in pos_ids:
            continue
        else:
            rank += 1
    for i, r in enumerate(Hits):
        if rank > r:
            hits[i] = 0
        else:
            break
    return rank, hits


def evaluate(model, entTotal, test_trips, args, entity_embedding, label_graph, edge_index, edge_type, edge_norm):
    H_Rank = []
    H_inv_Rank = []
    H_Hits = np.zeros((len(args.Hits)))
    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))
    head = test_trips[:, 0]
    rel = test_trips[:, 1]
    tail = test_trips[:, 2]
    bs = args.batch_size

    test_scores = np.zeros((test_trips.shape[0], entTotal))
    n_batches = int(test_trips.shape[0] / bs) + 1

    for i in range(n_batches):
        ent = head[i * bs:min((i + 1) * bs, test_trips.shape[0])]
        r = rel[i * bs:min((i + 1) * bs, test_trips.shape[0])]

        ent = ent.view(len(ent), -1)
        r = r.view(len(r), -1)

        samples = torch.cat([ent, r], dim=1)

        ent_embed, r_embed, np_embed = model.get_embed(samples, entity_embedding, edge_index, edge_type, edge_norm)

        scores = model.get_scores(ent_embed, r_embed, np_embed, ent.shape[0]).cpu().data.numpy()

        test_scores[i * bs:min((i + 1) * bs, test_trips.shape[0]), :] = scores

    for j in range(test_trips.shape[0]):

        sample_scores = -test_scores[j, :]

        true_tail = tail[j]

        temp_head = int(test_trips[j, 0])
        temp_rel = int(test_trips[j, 1])
        pos_ids = list(label_graph[(temp_head, temp_rel)])
        if j % 2 == 1:
            H_r, H_h = get_rank(sample_scores, true_tail, args.Hits, pos_ids)
            H_Rank.append(H_r)
            H_inv_Rank.append(1 / H_r)
            H_Hits += H_h
        else:
            T_r, T_h = get_rank(sample_scores, true_tail, args.Hits, pos_ids)
            T_Rank.append(T_r)
            T_inv_Rank.append(1 / T_r)
            T_Hits += T_h

    logger.info("Mean Rank: Head = {}  Tail = {}  Avg = {}"
                .format(np.mean(np.array(H_Rank)), np.mean(np.array(T_Rank)),
                        (np.mean(np.array(H_Rank)) + np.mean(np.array(T_Rank))) / 2))
    logger.info("MRR: Head = {}  Tail = {}  Avg = {}"
                .format(np.mean(np.array(H_inv_Rank)), np.mean(np.array(T_inv_Rank)),
                        (np.mean(np.array(H_inv_Rank)) + np.mean(np.array(T_inv_Rank))) / 2))

    for i, hits in enumerate(args.Hits):
        logger.info("Hits@{}: Head = {}  Tail={}  Avg = {}"
                    .format(hits, H_Hits[i] / len(H_Rank), T_Hits[i] / len(T_Rank),
                            (H_Hits[i] + T_Hits[i]) / (len(H_Rank) + len(T_Rank))))
    return (np.mean(np.array(H_Rank)) + np.mean(np.array(T_Rank))) / 2, (
            np.mean(np.array(H_inv_Rank)) + np.mean(np.array(T_inv_Rank))) / 2


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes=num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[1], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[1])) * (num_relation)
    edge_norm = 1 / deg[edge_index[1]].view(-1)[index]

    return edge_norm