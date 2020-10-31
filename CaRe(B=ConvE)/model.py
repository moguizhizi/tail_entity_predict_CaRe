# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/10/23 10:56
@file: model.py
@desc: 
"""
import os

from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

from encoder import GRUEncoder, LinearEncoder, GcnNet, GCNCov1, LAN, RGCNNet
from logger import config_logger

logger = config_logger('Model')


class ConvEParam(nn.Module):
    def __init__(self, args, embed_matrix, rel2words):
        super(ConvEParam, self).__init__()
        self.args = args

        self.rel2words = rel2words
        self.phrase_embed_model = GRUEncoder(embed_matrix, self.args)

        if self.args.CN == 'Linear':
            self.cn = LinearEncoder(args)
        elif self.args.CN == 'GCN':
            if self.args.GcnNum == 1:
                self.cn = GCNCov1(args.input_dim, args.nfeats, args.entPoolType)
            elif self.args.GcnNum == 2:
                self.cn = GcnNet(args.input_dim, args.nfeats, args.entPoolType)
            else:
                logger.error("The gcn number is error, value = %s", self.args.GcnNum)
                exit(0)
        elif self.args.CN == 'LAN':
            self.cn = LAN(args.input_dim, args.nfeats, args.entPoolType)
        elif self.args.CN == 'RGCN':
            self.cn = RGCNNet(args.input_dim, args.nfeats, args.entPoolType, args.rel_total, args.n_bases)
        else:
            logger.error("The module does not support so CN type, CN = %s", self.args.CN)
            exit(0)

        self.np_embeddings = nn.Embedding(self.args.ent_total, self.args.nfeats)
        nn.init.xavier_normal_(self.np_embeddings.weight.data)

        self.inp_drop = torch.nn.Dropout(self.args.dropout)
        self.hidden_drop = torch.nn.Dropout(self.args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(self.args.dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.args.nfeats)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.args.ent_total)))
        self.fc = torch.nn.Linear(16128, self.args.nfeats)

    def forward(self, x, edge, edge_type, edge_norm):
        if self.args.CN == 'RGCN':
            return self.cn(x, edge, edge_type, edge_norm)
        else:
            return self.cn(x, edge)

    def get_scores(self, ent, rel, ent_embed, batch_size):

        ent = ent.view(-1, 1, 15, 20)
        rel = rel.view(-1, 1, 15, 20)

        stacked_inputs = torch.cat([ent, rel], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_embed.transpose(1, 0))
        x += self.b.expand_as(x)

        return x

    def get_embed(self, samples, entity_embedding, edge_index, edge_type, edge_norm):

        np_embed = self.forward(entity_embedding, edge_index, edge_type, edge_norm)

        sub_embed = np_embed[samples[:, 0]]
        r = samples[:, 1]

        r_batch, r_len = seq_batch(r.cpu().numpy(), self.args, self.rel2words)

        rel_embed = self.phrase_embed_model(r_batch, r_len)

        return sub_embed, rel_embed, np_embed

    def get_loss(self, samples, labels, entity_embedding, edge_index, edge_type, edge_norm):

        sub_embed, rel_embed, np_embed = self.get_embed(samples, entity_embedding, edge_index, edge_type, edge_norm)
        scores = self.get_scores(sub_embed, rel_embed, np_embed, self.args.batch_size)
        pred = F.sigmoid(scores)

        predict_loss = self.loss(pred, labels)

        return predict_loss