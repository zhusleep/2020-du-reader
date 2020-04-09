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
    answer = item[3]
    doc_tokens = doc_tokens.replace(u"“", u"\"")
    if len(query_tokens) > config.max_query_length:
        query_tokens = query_tokens[0:config.max_query_length]
    one_token = tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + list(query_tokens) + ["[SEP]"] + list(doc_tokens) + ["[SEP]"])
    token_type = np.zeros(len(one_token))
    token_type[-len(doc_tokens)-1:] = 1
    # start and end position
    if train:
        start_pos = item[4]
        start_id = int(start_pos)
        start_id += len(query_tokens)+2
        end_id = start_id + len(answer)
        assert doc_tokens[start_id: end_id] == answer
        return q_id, one_token, token_type, start_id,end_id
    else:
        return q_id, one_token, token_type


class ReaderDataset(Dataset):

    def __init__(self, data, vocab_path=None, do_lower=True, train=True):
        super(ReaderDataset, self).__init__()
        self._data = data
        self.train = train
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.train:
            q_id, one_token, token_type, start_id,end_id = convert_one_line(self._data[idx], tokenizer=self._tokenizer)
            return q_id, torch.LongTensor(one_token), torch.LongTensor(token_type), start_id, end_id
        else:
            q_id, one_token, token_type = convert_one_line(self._data[idx], tokenizer=self._tokenizer)
            return q_id, torch.LongTensor(one_token), torch.LongTensor(token_type)


def collate_fn(batch):
    token, token_type = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    return token, token_type