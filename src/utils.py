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


def convert_one_line(item, tokenizer=None, mode='train'):
    q_id = item[0]
    doc_tokens = item[1]
    query_tokens = item[2]
    # doc_tokens = doc_tokens.replace(u"“", u"\"")
    # 起始位置偏离
    delta_pos = 0
    if len(query_tokens) > config.max_query_length:
        query_tokens = query_tokens[0:config.max_query_length]
    if len(doc_tokens) > 450:
        if mode=='train' or mode=='dev':
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
    if mode == 'train' or mode == 'dev':
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


def random_crop(sentence, start, text, n=2):
    """
    :param sentence: 答案所在段落
    :param start: 开始位置
    :param text: 答案
    :param max_len: 数据最大长度
    :param n: 数据增强成几份
    :return:[（sentence, start, text）]
    """
    # 随机切分成好几句话
    # 计算分割点
    results = []
    split_symbol = set(['，', '。', ' '])
    left_split_pos = []
    right_split_pos = []
    for index, char in enumerate(sentence):
        if start <= index < start+len(text):
            continue
        if char in split_symbol:
            if index < start:
                left_split_pos.append(index)
            else:
                right_split_pos.append(index)
    for r in range(2):
        temp_start, temp_end = 0, len(sentence)
        text_start = start
        if left_split_pos:
            temp_start = np.random.choice(left_split_pos, 1)[0]
            temp_start += 1
            text_start = text_start - temp_start
        if right_split_pos:
            temp_end = np.random.choice(right_split_pos, 1)[0]
            temp_end += 1
        results.append([sentence[temp_start:temp_end], text_start, text])
    # 保留原先输入
    results.append([sentence, start, text])
    return results


def limit_len(crop_result, max_len=450, mode='train'):
    new_results = []
    for sentence, start, text in crop_result:
        # print(sentence, start, text)
        assert sentence[start:start+len(text)] == text
        if len(sentence) < max_len:
            new_results.append([sentence, start, text])
        elif mode=='train':
            sentence_split = []
            if start+len(text) < max_len:
                sentence_split.append([sentence[0:max_len], start, text])
            if len(sentence)-max_len < start:
                cut = sentence[-max_len::]
                if cut[0:2]=='0 ':
                    pass
                sentence_split.append([cut, start-len(sentence)+len(cut), text])
            if not sentence_split:
                # 如果答案刚好在中间，则在答案左右进行切割
                sentence_split.append([sentence[start-150:start+300], 150, text])
            new_results.extend(sentence_split)
        elif mode=='dev':
            for i in range(int(len(sentence)/450)+1):
                new_results.append([sentence[i*450:(i+1)*450], -1, ''])

    for sentence, start, text in new_results:
        # print(sentence, start, text)
        if mode == 'train' and sentence[start:start+len(text)] != text:
            print(sentence, start, text)

    return new_results


def convert_one_line_new(item, tokenizer=None, mode='train'):
    q_id = item[0]
    doc_tokens = item[1]
    query_tokens = item[2]
    if mode == 'test':
        answer = ''
        start_pos = -1
    else:
        answer = item[3]
        start_pos = item[4]

    # doc_tokens = doc_tokens.replace(u"“", u"\"")
    # 起始位置偏离
    delta_pos = 0
    if len(query_tokens) > config.max_query_length:
        query_tokens = query_tokens[0:config.max_query_length]
    # 随机切割
    # crop_results = random_crop(doc_tokens, start_pos, answer)
    crop_results = [[doc_tokens, start_pos, answer]]
    # 限制长度
    results_limit_len = limit_len(crop_results, mode=mode)
    tokenizer_results = []
    for sentence, start, text in results_limit_len:
        if mode == 'train':
            assert sentence[start:start+len(text)] == text
        one_token = tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + list(query_tokens) + ["[SEP]"] + list(sentence) + ["[SEP]"])
        token_type = np.zeros(len(one_token))
        token_type[-len(doc_tokens)-1:] = 1
        # start and end position
        start_pos = start
        start_id = int(start_pos)
        start_id += len(query_tokens)+2
        end_id = start_id + len(answer)
        if end_id > len(one_token):
            print(end_id, len(one_token))
            # raise Exception('数据长度被截取')
        #assert one_token[start_id: end_id] == answer
        tokenizer_results.append([q_id, one_token, token_type, start_id, end_id])
    return tokenizer_results


class ReaderDataset(Dataset):

    def __init__(self, data, vocab_path=None, do_lower=True, tokenizer=None,
                 mode='train'):
        super(ReaderDataset, self).__init__()
        self._data = data
        self.mode = mode
        self._tokenizer = tokenizer
        self.q_id, self.one_token, self.token_type, self.start_id, self.end_id = [],[],[],[],[]
        self.collect_data()

    def __len__(self):
        return len(self.q_id)

    def collect_data(self):
        for item in self._data:
            results = convert_one_line_new(item, tokenizer=self._tokenizer, mode=self.mode)
            for q_id, one_token, token_type, start_id, end_id in results:
                self.q_id.append(q_id)
                self.one_token.append(one_token)
                self.token_type.append(token_type)
                self.start_id.append(start_id)
                self.end_id.append(end_id)

    def __getitem__(self, idx):
        if self.mode in['train','dev']:
            # q_id, one_token, token_type, start_id, end_id = convert_one_line(self._data[idx],
            #                                                                  tokenizer=self._tokenizer,
            #                                                                  train=self.train)
            q_id, one_token, token_type, start_id, end_id = self.q_id[idx], self.one_token[idx], self.token_type[idx], self.start_id[idx], self.end_id[idx]
            return (q_id, torch.LongTensor(one_token), torch.LongTensor(token_type), start_id, end_id)

        elif self.mode == 'test':
            q_id, one_token, token_type = self.q_id[idx], self.one_token[idx], self.token_type[idx]
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