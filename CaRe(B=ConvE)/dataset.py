# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/10/20 13:51
@file: dataset.py
@desc: 
"""
from args import init_args
import os
import pickle
import random

import numpy as np
import unicodedata
from elmoformanylangs import Embedder
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA

from logger import config_logger
from utils import get_list_from_file

logger = config_logger('DataSet')


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def preprocessForGlove(data_elmo, gloveEmbMap, vocab2index):
    token_glove = []
    for token in data_elmo:
        token_glove.append(gloveEmbMap[vocab2index[token]])
    data = np.array(token_glove).astype(np.float32)
    return data


def preprocessForElmo(text_data, ee):
    results = ee.sents2elmo(text_data)

    return results


def doPreprocess(data_mb, mode, ee=None, gloveEmbMap=None, vocab2index=None, bert_model=None):
    data_gen = []
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=len(data_mb)).start()

    for index, data in enumerate(data_mb):
        if mode == 'elmo':
            preprocessElmoData = preprocessForElmo(data, ee)
            data_gen.append(preprocessElmoData)
        elif mode == 'glove':
            preprocessGloveData = preprocessForGlove(data, gloveEmbMap, vocab2index)
            data_gen.append(preprocessGloveData)
    pbar.finish()

    return data_gen


def buildVocabMap(data_elmo):
    vocab2index, index2vocab = {}, {}
    count = 1
    vocab2index['unk'] = 0
    index2vocab[0] = 'unk'
    for data_mb in data_elmo:
        for character in data_mb:
            if not vocab2index.__contains__(character):
                vocab2index[character] = count
                index2vocab[count] = character
                count += 1
    return vocab2index, index2vocab


def buildGloveEmbMap(vocab2index, glove_file, dim=300):
    vocab_size = len(vocab2index)
    emb = np.zeros((vocab_size, dim))
    emb[0] = 0
    unknown_mask = np.zeros(vocab_size, dtype=bool)
    with open(glove_file, encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line_count += 1
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-dim]))
            if token in vocab2index:
                emb[vocab2index[token]] = [float(v) for v in elems[-dim:]]
                unknown_mask[vocab2index[token]] = True
    for index, mask in enumerate(unknown_mask):
        if not mask:
            emb[index] = emb[0]
    return emb


def doPreprocessForGlove(entity_list, file_glove_preprocessed, glove_path):
    if not os.path.isfile(file_glove_preprocessed):
        logger.info('Building vocabulary dictionary....')
        vocab2index, index2vocab = buildVocabMap(entity_list)
        logger.info('Building GloVe Embedding Map....')
        gloveEmbMap = buildGloveEmbMap(vocab2index, glove_path)
        logger.info('Prerpocessing Json data for Glove....')
        data_glove = doPreprocess(entity_list, mode='glove', gloveEmbMap=gloveEmbMap,
                                  vocab2index=vocab2index)
        with open(file_glove_preprocessed, 'wb') as f:
            pickle.dump(data_glove, f)
            logger.info('Successfully save preprocessed Glove data file %s',
                        file_glove_preprocessed)
    else:
        logger.info('Preprocessed Glove data is already existed, no preprocessing will be executed.')


def doPreprocessForElmo(text, file_elmo_preprocessed, model_dir):
    if not os.path.isfile(file_elmo_preprocessed):
        embedder = Embedder(model_dir)
        logger.info('Preprocsssing txt data for Elmo....')
        data = doPreprocess(text, mode='elmo', ee=embedder)
        logger.info('Preprocessing Elmo data finished')
        with open(file_elmo_preprocessed, 'wb') as f:
            pickle.dump(data, f)
            logger.info('Successfully save preprocessed Elmo data file %s',
                        file_elmo_preprocessed)
    else:
        logger.info('Preprocessed Elmo data is already existed, no preprocessing will be executed.')


def gen_id_file(origin_file, entity_file, relation_file):
    fin = open(origin_file, "r", encoding="utf8").readlines()
    entity_out = open(entity_file, "w", encoding="utf8")
    relation_out = open(relation_file, "w", encoding="utf8")
    entity_list = []
    relation_list = []
    entity_num = 0
    relation_num = 0
    max_entity_length = 0
    max_re_length = 0
    for trip in fin[0:]:
        record = trip.strip().split()
        head_entity = record[0]
        tail_entity = record[1]
        relation = record[2]

        if len(head_entity) > max_entity_length:
            max_entity_length = len(head_entity)

        if len(tail_entity) > max_entity_length:
            max_entity_length = len(tail_entity)

        if len(relation) > max_re_length:
            max_re_length = len(relation)

        if head_entity not in entity_list:
            temp = '{}{}{}'.format(head_entity, ' ', str(entity_num))
            entity_num = entity_num + 1
            entity_out.write(temp)
            entity_out.write("\n")
            entity_list.append(head_entity)

        if tail_entity not in entity_list:
            temp = '{}{}{}'.format(tail_entity, ' ', str(entity_num))
            entity_num = entity_num + 1
            entity_out.write(temp)
            entity_out.write("\n")
            entity_list.append(tail_entity)

        if relation not in relation_list:
            temp = '{}{}{}'.format(relation, ' ', str(relation_num))
            relation_num = relation_num + 1
            relation_out.write(temp)
            relation_out.write("\n")
            relation_list.append(relation)
    entity_out.write(str(entity_num) + " " + str(max_entity_length))
    relation_out.write(str(relation_num) + " " + str(max_re_length))

def gen_triple_file(origin_file, entity_list, relation_list, train_file, test_file, valid_file, origin_triple_file):
    entity_dict = {}
    for index, entity in enumerate(entity_list):
        entity_dict[entity] = index

    relation_dict = {}
    for index, entity in enumerate(relation_list):
        relation_dict[entity] = index

    fin = open(origin_file, "r", encoding="utf8").readlines()
    record = []
    for trip in fin[0:]:
        temp = trip.strip()
        record.append(temp)

    head_rel_list = []
    full_traversal_train_list = []
    for i, value in enumerate(record):
        temp = value.strip().split()
        temp_tuple = (temp[0], temp[2])
        if temp_tuple not in head_rel_list:
            head_rel_list.append(temp_tuple)
            full_traversal_train_list.append(i)

    train_num = int(len(record) * 0.7)
    test_num = int(len(record) * 0.2)

    remain_num = train_num - len(full_traversal_train_list)

    sub_list = list(set(range(0, len(record))).difference(set(full_traversal_train_list)))

    train_list = random.sample(sub_list, remain_num)

    train_index = train_list + full_traversal_train_list
    index = []
    for i in range(0, len(record)):
        if i not in train_index:
            index.append(i)

    test_index = random.sample(index, test_num)
    valid_index = []
    for i in index:
        if i not in test_index:
            valid_index.append(i)

    f_origin = open(origin_triple_file, "w", encoding="utf8")
    for i, value in enumerate(record):
        temp = value.strip().split()
        head_id = entity_dict[temp[0]]
        tail_id = entity_dict[temp[1]]
        relation_id = relation_dict[temp[2]]
        f_origin.write(str(head_id) + " " + str(relation_id) + " " + str(tail_id))
        f_origin.write('\n')

    logger.info('Successfully save origin file %s',
                origin_triple_file)

    f_train = open(train_file, "w", encoding="utf8")
    for i in train_index:
        temp = record[i].strip().split()
        head_id = entity_dict[temp[0]]
        tail_id = entity_dict[temp[1]]
        relation_id = relation_dict[temp[2]]
        f_train.write(str(head_id) + " " + str(relation_id) + " " + str(tail_id))
        f_train.write('\n')

    logger.info('Successfully save train file %s',
                train_file)

    f_test = open(test_file, "w", encoding="utf8")
    for i in test_index:
        temp = record[i].strip().split()
        head_id = entity_dict[temp[0]]
        tail_id = entity_dict[temp[1]]
        relation_id = relation_dict[temp[2]]
        f_test.write(str(head_id) + " " + str(relation_id) + " " + str(tail_id))
        f_test.write('\n')

    logger.info('Successfully save valid file %s',
                test_file)

    f_valid = open(valid_file, "w", encoding="utf8")
    for i in valid_index:
        temp = record[i].strip().split()
        head_id = entity_dict[temp[0]]
        tail_id = entity_dict[temp[1]]
        relation_id = relation_dict[temp[2]]
        f_valid.write(str(head_id) + " " + str(relation_id) + " " + str(tail_id))
        f_valid.write('\n')

    logger.info('Successfully save test file %s',
                valid_file)


def gen_word2id_file(origin_file, word2id_file):
    fin = open(origin_file, "r", encoding="utf8").readlines()
    word_dict = {}
    word_dict['PAD'] = 0
    for trip in fin[0:]:
        record = trip.strip().split()
        head_entity = record[0]
        tail_entity = record[1]
        relation = record[2]
        for word in head_entity:
            if word not in word_dict.keys():
                word_dict[word] = len(word_dict.keys())

        for word in tail_entity:
            if word not in word_dict.keys():
                word_dict[word] = len(word_dict.keys())

        for word in relation:
            if word not in word_dict.keys():
                word_dict[word] = len(word_dict.keys())

    word2id_out = open(word2id_file, "w", encoding="utf8")
    for key, value in word_dict.items():
        record = '{} {}\n'.format(key, value)
        word2id_out.write(record)


def gen_word_glove_file(word2id_file, word_glove_path, glove_path):
    fin = open(word2id_file, "r", encoding="utf8").readlines()
    word2index = {}
    for trip in fin[0:]:
        record = trip.strip().split()
        word2index[record[0]] = int(record[1])

    word_embedding = buildGloveEmbMap(word2index, glove_path)
    with open(word_glove_path, 'wb') as f:
        pickle.dump(word_embedding, f)
        logger.info('Successfully save preprocessed Glove data file %s',
                    word_glove_path)


def get_relation2word_file(rel2id_file, word2id_file, relation2word_file):
    fin = open(word2id_file, "r", encoding="utf8").readlines()
    word2index = {}
    for trip in fin[0:]:
        record = trip.strip().split()
        word2index[record[0]] = int(record[1])

    fr2id = open(rel2id_file, "r", encoding="utf8").readlines()
    r2w_dict = {}
    for trip in fr2id[0:-1]:
        record = trip.strip().split()
        temp = []
        for char in record[0]:
            if char in word2index:
                temp.append(word2index[char])
            else:
                temp.append(0)
        r2w_dict[int(record[1])] = temp

    with open(relation2word_file, 'wb') as f:
        pickle.dump(r2w_dict, f)
        logger.info('Successfully save relation to word file %s',
                    relation2word_file)


def gen_label(triples_path, label_file):
    label_graph = {}
    f = open(triples_path, "r").readlines()
    for trip in f[0:]:
        trip = trip.strip().split()
        e1, r, e2 = int(trip[0]), int(trip[1]), int(trip[2])

        if (e1, r) not in label_graph:
            label_graph[(e1, r)] = set()
        label_graph[(e1, r)].add(e2)

    with open(label_file, 'wb') as f:
        pickle.dump(label_graph, f)
        logger.info('Successfully save train label file %s',
                    label_file)


def convert_to_stander(dataset_file, origin_file):
    fin = open(dataset_file, "r", encoding="utf8").readlines()
    fout = open(origin_file, "w", encoding="utf8")
    for trip in fin[0:]:
        temp = trip.strip().split('\t')
        temp[0] = temp[0].strip().replace(' ', '_')
        temp[1] = temp[1].strip().replace(' ', '_')
        temp[2] = temp[2].strip().replace(' ', '_')
        fout.write(str(temp[0]) + " " + str(temp[1]) + " " + str(temp[2]))
        fout.write("\n")


def get_edges_type(triple_file, edges_file, edages_type_file):
    fin = open(triple_file, "r", encoding="utf8").readlines()
    head_list = []
    tail_list = []
    relation_list = []
    for trip in fin[0:]:
        temp = trip.strip().split()
        temp[0] = temp[0].strip()
        temp[2] = temp[2].strip()
        head_id = int(temp[0])
        rel_id = int(temp[1])
        tail_id = int(temp[2])
        head_list.append(head_id)
        tail_list.append(tail_id)
        relation_list.append(rel_id)

    head_list = np.array(head_list).reshape(1, -1)
    tail_list = np.array(tail_list).reshape(1, -1)
    relation_list = np.array(relation_list)

    edges = np.concatenate((np.array(head_list), np.array(tail_list)), axis=0)

    with open(edges_file, 'wb') as f:
        pickle.dump(edges, f)

    with open(edages_type_file, 'wb') as f:
        pickle.dump(relation_list, f)

    logger.info('Successfully save graph edages and type file %s, %s',
                edges_file, edages_type_file)


if __name__ == '__main__':
    args = init_args(is_dataset=True)

    for item in vars(args).items():
        logger.info('%s : %s', item[0], str(item[1]))

    data_files = args.data_files

    convert_to_stander(data_files['dataset_path'], data_files['origin_path'])

    gen_id_file(data_files['origin_path'], data_files['ent2id_path'], data_files['rel2id_path'])

    entity_list = get_list_from_file(data_files['ent2id_path'], is_contain_last=False)
    doPreprocessForElmo(entity_list, data_files['elmo_embedding_path'], args.elmo_model_path)
    doPreprocessForGlove(entity_list, data_files['glove_embedding_path'], args.glove_path)

    relation_list = get_list_from_file(data_files['rel2id_path'], is_contain_last=False)

    gen_triple_file(data_files['origin_path'], entity_list, relation_list,
                    data_files['train_trip_path'], data_files['test_trip_path'],
                    data_files['valid_trip_path'], data_files['origin_trip_path'])

    gen_word2id_file(data_files['origin_path'], data_files['word2id_path'])

    gen_word_glove_file(data_files['word2id_path'], data_files['word_glove_path'], args.glove_path)

    get_relation2word_file(data_files['rel2id_path'], data_files['word2id_path'], data_files['relation2word_path'])

    gen_label(data_files['train_trip_path'], data_files['train_label_path'])

    get_edges_type(data_files['origin_trip_path'], data_files['graph_edges_path'], data_files['edges_type_path'])
