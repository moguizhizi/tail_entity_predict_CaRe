# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/11/1 16:43
@file: test.py
@desc: 
"""

import os

from args import init_args
from model import ConvEParam
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

torch.backends.cudnn.enabled = False

from logger import config_logger

logger = config_logger('Test')


def check_valid(entity_list, relation_list, predict_file):
    fin = open(predict_file, "r", encoding="utf8").readlines()
    for trip in fin[0:-1]:
        record = trip.strip().split()
        if record[0] not in entity_list:
            logger.error('Entity %s does not exist', record[0])
            return False

        if record[1] == 'all':
            continue
        elif record[1] not in relation_list:
            logger.error('Relation %s does not exist', record[0])
            return False

    return True


def get_predict_trip(entity_dict, relation_dict, pridict_file, relation_list, is_use_cuda):
    fin = open(pridict_file, "r", encoding="utf8").readlines()
    trip_list = []
    for trip in fin[0:]:
        temp = trip.strip().split()
        e1 = entity_dict[temp[0]]
        if temp[1] == 'all':
            for i, rel_type in enumerate(relation_list):
                rel = relation_dict[rel_type]
                trip_list.append([e1, rel])
        else:
            rel = relation_dict[temp[1]]
            trip_list.append([e1, rel])

    tripls = np.array(trip_list)

    tripls = Variable(torch.from_numpy(tripls))
    if is_use_cuda:
        tripls = tripls.cuda()

    return tripls


def predict(model, predict_trips, args, entity_embedding, label_graph, edge_index, edge_type, edge_norm, entity_dict,
            rel_dict):
    head = predict_trips[:, 0]
    rel = predict_trips[:, 1]
    bs = args.batch_size
    predict_num = args.predict_num
    result_file = args.data_files['result_path']

    test_scores = np.zeros((predict_trips.shape[0], args.ent_total))
    n_batches = int(predict_trips.shape[0] / bs) + 1

    for i in range(n_batches):
        ent = head[i * bs:min((i + 1) * bs, predict_trips.shape[0])]
        r = rel[i * bs:min((i + 1) * bs, predict_trips.shape[0])]

        ent = ent.view(len(ent), -1)
        r = r.view(len(r), -1)

        samples = torch.cat([ent, r], dim=1)

        ent_embed, r_embed, np_embed = model.get_embed(samples, entity_embedding, edge_index, edge_type, edge_norm)

        scores = model.get_scores(ent_embed, r_embed, np_embed, ent.shape[0]).cpu().data.numpy()

        test_scores[i * bs:min((i + 1) * bs, predict_trips.shape[0]), :] = scores

    for j in range(predict_trips.shape[0]):

        sample_scores = -test_scores[j, :]

        temp_head = int(predict_trips[j, 0])
        temp_rel = int(predict_trips[j, 1])

        try:
            pos_ids = list(label_graph[(temp_head, temp_rel)])
        except KeyError as e:
            pos_ids = []

        scores = np.argsort(sample_scores)

        rank = 0
        predict_list = []
        for i in range(scores.shape[0]):
            if scores[i] in pos_ids:
                continue
            else:
                rank = rank + 1
                predict_list.append(scores[i])
                if rank >= predict_num:
                    break

        for i, predict_index in enumerate(predict_list):
            pre_head = entity_dict[temp_head]
            pre_rel = rel_dict[temp_rel]
            pre_tail = entity_dict[predict_index]

            with open(result_file, 'a+', encoding='utf8') as f:
                temp_record = '{}{}{}{}{}'.format(pre_head, ' ', pre_rel, ' ', pre_tail)
                f.write(temp_record)
                f.write('\n')

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
    label_graph = np.load(args.data_files['train_label_path'])

    model = ConvEParam(args, embed_matrix, rel2words)

    logger.info(model)

    if args.use_cuda:
        model.cuda()
        entity_embedding = entity_embedding.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()

    entity_list = get_list_from_file(args.data_files['ent2id_path'], is_contain_last=False)
    relation_list = get_list_from_file(args.data_files['rel2id_path'], is_contain_last=False)

    if check_valid(entity_list, relation_list, args.data_files['predict_path']) != True:
        logger.error("Incorrect content or format")
        exit(0)

    entity_dict, entity_inv_dict = get_dict_from_file(args.data_files['ent2id_path'], is_contain_last=False)
    entity_dict = str_to_int_from_dict(entity_dict, is_key_trans=False, is_value_trans=True)
    entity_inv_dict = str_to_int_from_dict(entity_inv_dict, is_key_trans=True, is_value_trans=False)

    rel_dict, rel_inv_dict = get_dict_from_file(args.data_files['rel2id_path'], is_contain_last=False)
    rel_dict = str_to_int_from_dict(rel_dict, is_key_trans=False, is_value_trans=True)
    rel_inv_dict = str_to_int_from_dict(rel_inv_dict, is_key_trans=True, is_value_trans=False)
    predict_trip = get_predict_trip(entity_dict, rel_dict, args.data_files['predict_path'], relation_list,
                                    args.use_cuda)

    model_state_file = args.data_files['model_path']

    checkpoint = torch.load(model_state_file)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])

    if os.path.exists(args.data_files['result_path']):
        os.remove(args.data_files['result_path'])

    predict(model, predict_trip, args, entity_embedding, label_graph, edge_index, edge_type, edge_norm, entity_inv_dict,
            rel_inv_dict)

    logger.info('Successful')
