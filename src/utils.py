import sys
import time
import socket
from argparse import ArgumentParser
from collections import OrderedDict
from json import dumps, loads, load
from pathlib import Path

import numpy as np
import torch
from redis import StrictRedis
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AlbertConfig, AlbertModel, BertTokenizer


def convert_one_line(text_a, text_b, tokenizer=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)
    one_token = tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"])
    token_type = np.zeros(len(one_token))
    token_type[-len(tokens_b):] = 1
    return one_token, token_type


class TestDataset(Dataset):

    def __init__(self, data, vocab_path=None, do_lower=True):
        super(TestDataset, self).__init__()
        self._data = data
        self._tokenizer = BertTokenizer.from_pretrained(
            vocab_path, cache_dir=None, do_lower_case=do_lower)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        token, token_type = convert_one_line(self._data[idx][0], self._data[idx][1], tokenizer=self._tokenizer)
        return torch.LongTensor(token), torch.LongTensor(token_type)


def collate_test_fn(batch):
    token, token_type = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    return token, token_type