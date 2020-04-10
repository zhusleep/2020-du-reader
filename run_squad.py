import json
import config
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

random.seed(config.seed)


def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],qa['answers'][0]['text'],qa['answers'][0]['answer_start']
                # [a['text'] for a in qa.get('answers', [])]
            ])
    return D


def read_squad_examples(input_file, train=True):
    total, error = 0, 0
    examples = []
    raw_data = load_data(input_file)
    for index, item in enumerate(raw_data):
        q_id = item[0]
        context = item[1]
        question = item[2]
        answer = item[3]
        if train:
            start_pos = item[4]
        else:
            start_pos = context.index(answer)

        start_id = int(start_pos)
        end_id = start_id+len(answer)
        assert context[start_id: end_id] == answer
        # 异常数据则丢弃
        if start_id >= end_id or end_id > len(context):
            continue
        # 收集信息
        example = {
            "qas_id": q_id,
            "question_text": question.strip(),
            "doc_tokens": context.strip(),
            "start_position": start_id,
            "end_position": end_id}

        examples.append(example)
    print("len(examples):", len(examples))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length, train=True):

    features = []

    for example in tqdm(examples):
        query_tokens = list(example['question_text'])
        doc_tokens = example['doc_tokens']
        doc_tokens = doc_tokens.replace(u"“", u"\"")
        doc_tokens = doc_tokens.replace(u"”", u"\"")
        start_position = example['start_position']
        end_position = example['end_position']

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
            start_position = start_position + 1
            end_position = end_position + 1

        tokens.append("[SEP]")
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for i in doc_tokens:
            tokens.append(i)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if end_position >= max_seq_length:
            continue

        if len(tokens) > max_seq_length:
            tokens[max_seq_length-1] = "[SEP]"
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])      ## !!! SEP
            segment_ids = segment_ids[:max_seq_length]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) == len(segment_ids)

        features.append(
                        {"input_ids":input_ids,
                         "input_mask":input_mask,
                         "segment_ids":segment_ids,
                         "start_position":start_position,
                         "end_position":end_position,
                         'qas_id':example['qas_id']})
    if train:
        with open("../featured_data/train.data", 'w', encoding="utf-8") as fout:
            for feature in features:
                fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
        print("train len(features):", len(features))
    else:
        with open("../featured_data/dev.data", 'w', encoding="utf-8") as fout:
            for feature in features:
                fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
        print("dev len(features):", len(features))
    return features


if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    # 生成训练数据， train.data
    examples = read_squad_examples(input_file=config.train_input_file,train=True)
    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                            max_seq_length=config.max_seq_length,
                                            max_query_length=config.max_query_length)
    # 生成验证数据
    examples = read_squad_examples(input_file=config.dev_input_file,train=False)
    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                            max_seq_length=config.max_seq_length,
                                            max_query_length=config.max_query_length,
                                            train=False)

    # 生成验证数据， dev.data。记得注释掉生成训练数据的代码，并在196行将train.data改为dev.data
    # examples = read_squad_examples(zhidao_input_file=config.dev_zhidao_input_file,
    #                               input_file=config.dev_input_file)
    # features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
    #                                        max_seq_length=config.max_seq_length, max_query_length=config.max_query_length)
