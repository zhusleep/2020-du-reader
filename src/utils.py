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
import unicodedata

from redis import StrictRedis
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AlbertConfig, AlbertModel, BertTokenizer

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


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def limit_len(crop_result, max_len=450, mode='train'):
    new_results = []
    for sentence, start, text in crop_result:
        # print(sentence, start, text)
        assert sentence[start:start+len(text)] == text
        if len(sentence) < max_len:
            new_results.append([sentence, start, text])
        elif mode == 'train':
            sentence_split = []
            if start+len(text) < max_len:
                sentence_split.append([sentence[0:max_len], start, text])
            elif len(sentence)-max_len < start:
                cut = sentence[-max_len::]
                sentence_split.append([cut, start-len(sentence)+len(cut), text])
            if not sentence_split:
                # 如果答案刚好在中间，则在答案左右进行切割
                k = np.random.randint(0, 100)
                sentence_split.append([sentence[start-k:start+max_len-k], k, text])
            new_results.extend(sentence_split)
        elif mode in ['dev', 'test']:
            for i in range(int(len(sentence)/max_len)+1):
                new_results.append([sentence[i*max_len:(i+1)*max_len], -1, ''])
                # 避免答案在分割线中
                new_results.append([sentence[int((i+0.5)*max_len):int((i+1.5)*max_len)], -1, ''])


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
    results_limit_len = limit_len(crop_results,
                                  max_len=config.max_seq_length-len(query_tokens),
                                  mode=mode)
    tokenizer_results = []
    for sentence, start, text in results_limit_len:
        if len(sentence)==0:continue
        if mode == 'train':
            assert sentence[start:start+len(text)] == text
            question = ["[CLS]"] + tokenizer.tokenize(query_tokens.lower()) + ["[SEP]"]
            left_doc = tokenizer.tokenize(sentence.lower()[0:start])
            middle_doc = tokenizer.tokenize(text.lower())
            right_doc = tokenizer.tokenize(sentence.lower()[(start+len(text)):])+["[SEP]"]
            one_token = tokenizer.convert_tokens_to_ids(question+left_doc+middle_doc+right_doc)
            token_type = np.zeros(len(one_token))
            token_type[len(question):] = 1
            # start and end position
            start_id = len(question+left_doc)
            # start_id = int(start_pos)
            # start_id += len(query_tokens)+2
            end_id = len(question+left_doc+middle_doc)-1
            # raw_sentence = '.' + query_tokens + '。' + sentence
            tokenizer_results.append([q_id, (sentence,None,None), one_token, token_type, start_id, end_id])
        else:
            question = ["[CLS]"] + tokenizer.tokenize(query_tokens.lower()) + ["[SEP]"]
            first_token_to_orig_index = {}
            first_token_text = []
            last_not_chinese_index = -1
            seprate = set([',','.','?',':','!','<','>',"：","，","。","？","《","》",'、','-',
                           '（','）','(',')','|','/','(','〕','.','~','”','①',';','＝'])
            # 相当于空格填充后split() ------------
            for index, char in enumerate(sentence):
                cp = ord(char)
                if _is_chinese_char(cp) or char in seprate or _is_punctuation(char):
                    if last_not_chinese_index>=0:
                        first_token_text.append(sentence[last_not_chinese_index:index])
                        first_token_to_orig_index[len(first_token_text) - 1] = [last_not_chinese_index, index]
                        last_not_chinese_index = -1
                    first_token_text.append(char)
                    first_token_to_orig_index[len(first_token_text)-1] = [index, index+1]
                    last_not_chinese_index = -1
                elif char == " ":
                    if last_not_chinese_index>=0:
                        first_token_text.append(sentence[last_not_chinese_index:index])
                        first_token_to_orig_index[len(first_token_text)-1] = [last_not_chinese_index, index]
                        last_not_chinese_index = -1
                else:
                    if last_not_chinese_index==-1:
                        last_not_chinese_index = index
            if sentence[-1] != " ":
                first_token_text.append(sentence[last_not_chinese_index:index+1])
                first_token_to_orig_index[len(first_token_text)-1] = [last_not_chinese_index, index+1]
                last_not_chinese_index = -1
            # split 结束 -----------------------
            # 校验：
            for index,item in enumerate(first_token_text):
                assert item ==sentence[first_token_to_orig_index[index][0]:first_token_to_orig_index[index][1]]
            # 开始tokenize
            original_to_tokenized_index = []
            tokenized_to_original_index = []
            all_doc_tokens = []  # tokenized document text
            for i , word in enumerate(first_token_text):
                original_to_tokenized_index.append(len(all_doc_tokens))
                word = word.lower()
                for sub_token in word:
                    tokenized_to_original_index.append(i)
                    all_doc_tokens.append(sub_token)
            one_token = tokenizer.convert_tokens_to_ids(question+all_doc_tokens)
            if len(one_token)>512:
                raise
                continue
            token_type = np.zeros(len(one_token))
            token_type[len(question):] = 1
            tokenizer_results.append([q_id, (sentence,tokenized_to_original_index,first_token_to_orig_index), one_token, token_type, 0, 0])

            # todo
    return tokenizer_results


class ReaderDataset(Dataset):

    def __init__(self, data, vocab_path=None, do_lower=True, tokenizer=None,
                 mode='train'):
        super(ReaderDataset, self).__init__()
        self._data = data
        self.mode = mode
        self._tokenizer = tokenizer
        self.q_id, self.raw_sentence, self.one_token, self.token_type, self.start_id, self.end_id = [],[],[],[],[],[]
        self.collect_data()

    def __len__(self):
        return len(self.q_id)

    def collect_data(self):
        for item in self._data:
            if item[0]=='a2cdf8f87575527755075a687fb55939':
                j = 1
            results = convert_one_line_new(item, tokenizer=self._tokenizer, mode=self.mode)
            for q_id, raw_sentence, one_token, token_type, start_id, end_id in results:
                self.q_id.append(q_id)
                self.raw_sentence.append(raw_sentence)
                self.one_token.append(one_token)
                self.token_type.append(token_type)
                self.start_id.append(start_id)
                self.end_id.append(end_id)

    def __getitem__(self, idx):
        if self.mode in['train','dev']:
            # q_id, one_token, token_type, start_id, end_id = convert_one_line(self._data[idx],
            #                                                                  tokenizer=self._tokenizer,
            #                                                                  train=self.train)
            q_id, raw_sentence, one_token, token_type, start_id, end_id = self.q_id[idx], self.raw_sentence[idx], self.one_token[idx], self.token_type[idx], self.start_id[idx], self.end_id[idx]
            return (q_id, raw_sentence, torch.LongTensor(one_token), torch.LongTensor(token_type), start_id, end_id)

        elif self.mode == 'test':
            q_id, raw_sentence, one_token, token_type = self.q_id[idx], self.raw_sentence[idx], self.one_token[idx], self.token_type[idx]
            return q_id, raw_sentence, torch.LongTensor(one_token), torch.LongTensor(token_type)


def collate_fn_train(batch):
    q_id, raw_sentence, token, token_type, start_id, end_id = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    start_id = torch.tensor(start_id)
    end_id = torch.tensor(end_id)
    return q_id, raw_sentence, token, token_type, start_id, end_id


def collate_fn_test(batch):
    q_id, raw_sentence, token, token_type = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    return q_id, raw_sentence, token, token_type


def find_best_answer_for_passage(start_probs, end_probs, min_start, max_end):
    start_probs, end_probs = start_probs[min_start:max_end], end_probs[min_start:max_end]
    (best_start, best_end), max_prob = find_best_answer(start_probs, end_probs)
    maxlen = 30
    if best_end-best_start < maxlen:
        return (best_start+min_start, best_end+min_start), max_prob
    # 限制最大长度为n
    n = min(maxlen,len(start_probs))
    results = []
    # start_probs = start_probs-min(start_probs)
    # end_probs = end_probs-min(end_probs)

    for i in range(0, len(start_probs)-n+1, 5):

        (best_start, best_end), max_prob = find_best_answer(start_probs[i:i+n], end_probs[i:i+n])
        results.append([(best_start+i, best_end+i), max_prob])
    results_submit = sorted(results,key=lambda x:x[1], reverse=True)[0]
    (best_start, best_end), max_prob = results_submit
    return (best_start + min_start, best_end + min_start), max_prob


def find_best_answer(start_probs, end_probs):
    best_start, best_end, max_prob = -1, -1, 0

    start_probs, end_probs = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
    prob_start, best_start = torch.max(start_probs, 1)
    prob_end, best_end = torch.max(end_probs, 1)
    num = 0
    results = []
    while True:
        if num > 3:
            break
        if best_end >= best_start:
            break
        else:
            # be
            # start_probs[0][best_start], end_probs[0][best_end] = start_probs[0][best_start], -10
            # 最大最小位置颠倒，则分开计算
            prob_start_1, best_start_1 = prob_start,best_start
            prob_end_1, best_end_1 = torch.max(end_probs[:, best_start:], 1)
            max_prob_1 = prob_start_1 + prob_end_1
            if best_end==0:
                best_start,best_end,prob_start,prob_end = best_start,best_end_1+best_start,prob_start,prob_end_1
                break
            prob_start_2, best_start_2 = torch.max(start_probs[:, :best_end], 1)
            prob_end_2, best_end_2 = prob_end, best_end
            max_prob_2 = prob_start_2 + prob_end_2
            if max_prob_1>max_prob_2:
                best_start,best_end,prob_start,prob_end = best_start,best_end_1+best_start,prob_start,prob_end_1
            else:
                best_start,best_end,prob_start,prob_end = best_start_2,best_end,prob_start_2,prob_end
            break

    max_prob = prob_start + prob_end

    if best_start <= best_end:
        return (best_start, best_end), max_prob
    else:
        return (best_end, best_start), max_prob