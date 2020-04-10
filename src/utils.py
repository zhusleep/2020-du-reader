import sys
import time
import socket
from argparse import ArgumentParser
from collections import OrderedDict
from json import dumps, loads, load
from pathlib import Path

import numpy as np
import torch
import config
from redis import StrictRedis
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AlbertConfig, AlbertModel, BertTokenizer


def convert_one_line(item, tokenizer=None, train=True):
    q_id = item[0]
    doc_tokens = item[1]
    query_tokens = item[2]
    doc_tokens = doc_tokens.replace(u"“", u"\"")
    # 起始位置偏离
    delta_pos = 0
    if len(query_tokens) > config.max_query_length:
        query_tokens = query_tokens[0:config.max_query_length]
    if len(doc_tokens) > 450:
        if train:
            answer = item[3]
            start_pos = item[4]
            if start_pos+len(answer) < 450:
                doc_tokens = doc_tokens[0:450]
            else:
                delta_pos = len(doc_tokens)-450
                doc_tokens = doc_tokens[-450::]
        else:
            doc_tokens = doc_tokens[0:450]
    one_token = tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + list(query_tokens) + ["[SEP]"] + list(doc_tokens) + ["[SEP]"])
    token_type = np.zeros(len(one_token))
    token_type[-len(doc_tokens)-1:] = 1
    # start and end position
    if train:
        answer = item[3]
        start_pos = item[4]
        start_id = int(start_pos)
        start_id += len(query_tokens)+2
        start_id -= delta_pos
        end_id = start_id + len(answer)
        if end_id > len(one_token):
            print(end_id, len(one_token))
            raise Exception('数据长度被截取')
        #assert one_token[start_id: end_id] == answer
        return q_id, one_token, token_type, start_id,end_id
    else:
        return q_id, one_token, token_type


class ReaderDataset(Dataset):

    def __init__(self, data, vocab_path=None, do_lower=True, tokenizer=None,
                 train=True):
        super(ReaderDataset, self).__init__()
        self._data = data
        self.train = train
        self._tokenizer = tokenizer
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.train:
            q_id, one_token, token_type, start_id, end_id = convert_one_line(self._data[idx],
                                                                             tokenizer=self._tokenizer,
                                                                             train=self.train)
            return (q_id, torch.LongTensor(one_token), torch.LongTensor(token_type), \
                   start_id, end_id)
        else:
            q_id, one_token, token_type = convert_one_line(self._data[idx],
                                                           tokenizer=self._tokenizer,
                                                           train=self.train)
            return q_id, torch.LongTensor(one_token), torch.LongTensor(token_type)


def collate_fn_train(batch):
    q_id, token, token_type, start_id, end_id = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    start_id = torch.tensor(start_id)
    end_id = torch.tensor(end_id)
    return q_id, token, token_type, start_id, end_id


def collate_fn_test(batch):
    q_id, token, token_type = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    return q_id, token, token_type


def find_best_answer_for_passage(start_probs, end_probs):
    best_start, best_end, max_prob = -1, -1, 0

    start_probs, end_probs = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
    prob_start, best_start = torch.max(start_probs, 1)
    prob_end, best_end = torch.max(end_probs, 1)
    num = 0
    while True:
        if num > 3:
            break
        if best_end >= best_start:
            break
        else:
            start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0
            prob_start, best_start = torch.max(start_probs, 1)
            prob_end, best_end = torch.max(end_probs, 1)
        num += 1
    max_prob = prob_start * prob_end

    if best_start <= best_end:
        return (best_start, best_end), max_prob
    else:
        return (best_end, best_start), max_prob