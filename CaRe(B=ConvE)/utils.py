import math
import warnings
import pickle

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


def evaluate(model, test_trips, args, entity_embedding, label_graph, edge_index, edge_type, edge_norm, src_index):
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

    test_scores = np.zeros((test_trips.shape[0], args.ent_total))
    n_batches = int(test_trips.shape[0] / bs) + 1

    for i in range(n_batches):
        ent = head[i * bs:min((i + 1) * bs, test_trips.shape[0])]
        r = rel[i * bs:min((i + 1) * bs, test_trips.shape[0])]

        ent = ent.view(len(ent), -1)
        r = r.view(len(r), -1)

        samples = torch.cat([ent, r], dim=1)

        ent_embed, r_embed, np_embed = model.get_embed(samples, entity_embedding, edge_index, edge_type, edge_norm, src_index)

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


def get_list_from_file(_file, is_contain_last=True):
    fin = open(_file, "r", encoding="utf8").readlines()
    if is_contain_last != True:
        records = fin[0:-1]
    else:
        records = fin[0:]

    conttent_list = []
    for trip in records:
        record = trip.strip().split()
        conttent_list.append(record[0])

    return conttent_list

def get_list_from_triple_file(triple_file):
    fin = open(triple_file, "r", encoding="utf8").readlines()
    conttent_list = []
    for trip in fin[0:]:
        record = trip.strip().split()
        entity_id = int(record[0])
        if entity_id not in conttent_list:
            conttent_list.append(entity_id)

        entity_id = int(record[2])
        if entity_id not in conttent_list:
            conttent_list.append(entity_id)

    return conttent_list


def get_dict_from_file(file, is_contain_last=True):
    fin = open(file, "r", encoding="utf8").readlines()
    if is_contain_last != True:
        records = fin[0:-1]
    else:
        records = fin[0:]

    content_dict = {}
    content_inv_dict = {}
    for trip in records:
        record = trip.strip().split()
        content_dict[record[0]] = record[1]
        content_inv_dict[record[1]] = record[0]

    return content_dict, content_inv_dict

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


def get_glove_entity_dim(args):
    glove_embedding = pickle.load(open(args.data_files['glove_embedding_path'], 'rb'))
    dim = glove_embedding[0].shape[1]

    return int(dim)


def get_elmo_entity_dim(args):
    elmo_embedding = pickle.load(open(args.data_files['elmo_embedding_path'], 'rb'))

    temp = np.mean(elmo_embedding[0], 1)

    dim = temp.shape[1]

    return int(dim)


def pad_elmo_all_entity_embedding(args):
    elmo_embedding = pickle.load(open(args.data_files['elmo_embedding_path'], 'rb'))
    pad_all_entity = []
    for _, entity_embedding in enumerate(elmo_embedding):
        temp = np.mean(entity_embedding, 1)
        size = len(temp)
        pad_glove_entity = np.pad(temp, ((0, args.max_entity_length - size), (0, 0)), mode='constant')
        pad_all_entity.append(pad_glove_entity)

    return pad_all_entity


def pad_glove_all_entity_embedding(args):
    glove_embedding = pickle.load(open(args.data_files['glove_embedding_path'], 'rb'))
    pad_all_entity = []
    for _, entity_embedding in enumerate(glove_embedding):
        size = len(entity_embedding)
        pad_glove_entity = np.pad(entity_embedding, ((0, args.max_entity_length - size), (0, 0)), mode='constant')
        pad_all_entity.append(pad_glove_entity)

    return pad_all_entity

def str_to_int_from_dict(trans_dict, is_key_trans = True, is_value_trans = True, key_type = None, value_type = None):
    temp_dict = {}
    for key, value in trans_dict.items():
        temp_key = key
        temp_value = value
        if is_key_trans == True:
            if key_type == None:
               temp_key = int(key)
            else:
               temp_key = key_type(map(int, key))

        if is_value_trans == True:
            if value_type == None:
               temp_value = int(value)
            else:
               temp_value = value_type(map(int, value))

        temp_dict[temp_key] = temp_value

    return temp_dict


def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点

    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表

    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        results.append(res)

    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样

    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射

    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]

    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result





