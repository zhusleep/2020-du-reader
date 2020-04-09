import os
import config
import torch
import random
import pickle
from tqdm import tqdm
from torch import nn, optim

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AdamW, get_constant_schedule_with_warmup
# from dataset.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertForQuestionAnswering, BertConfig
from utils import ReaderDataset
# 随机种子
random.seed(config.seed)
torch.manual_seed(config.seed)
import json


def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],qa['answers'][0]['text'],qa['answers'][0]['answer_start']
                # [a['text'] for a in qa.get('answers', [])]
            ])
    return D


def train():
    # 1 载入数据
    train_data = load_data('../data/train.json')
    dev_data = load_data('../data/dev.json')
    train_set = ReaderDataset(train_data,train=True)
    train_dataloader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 2 载入模型
    # 加载预训练bert
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
    device = config.device
    model.to(device)

    # 3 准备 optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_steps = config.num_train_optimization_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    # 4 开始训练
    for i in range(config.num_train_epochs):
        for step , batch in enumerate(tqdm(train_dataloader, desc="Epoch")):
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            input_ids, input_mask, segment_ids, start_positions, end_positions = \
                                        input_ids.to(device), input_mask.to(device), segment_ids.to(device), start_positions.to(device), end_positions.to(device)

            # 计算loss
            loss, _, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, start_positions=start_positions, end_positions=end_positions)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            # 更新梯度
            if (step+1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step(step)
                optimizer.zero_grad()
                print(loss.item())

    # # make prediction
    # test_predicts = []
    #
    # for inputs, token_types in test_loader:
    #     inputs, token_types = inputs.to(device), token_types.to(device)
    #     masks = (inputs > 0).to(device)
    #     v_pred = model(inputs, token_types, masks)
    #     v_pred = torch.sigmoid(v_pred)
    #     # predict
    #     test_predicts.extend(v_pred.cpu().numpy())
    #
    # # 准备数据
    # data = Dureader()
    # train_dataloader, dev_dataloader = data.train_iter, data.dev_iter
    #
    #
    torch.save(model.state_dict(), "final.pt")


def evaluate(model, dev_data):
    total, losses = 0.0, []
    device = config.device

    with torch.no_grad():
        model.eval()
        for batch in dev_data:

            input_ids, input_mask, segment_ids, start_positions, end_positions = batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            loss, _, _ = model(input_ids.to(device), token_type_ids=segment_ids.to(device), attention_mask=input_mask.to(device),
                               start_positions=start_positions.to(device), end_positions=end_positions.to(device))
            loss = loss / config.gradient_accumulation_steps
            losses.append(loss.item())

        for i in losses:
            total += i
        with open("./log", 'a') as f:
            f.write("eval_loss: " + str(total / len(losses)) + "\n")

        return total / len(losses)


if __name__ == "__main__":
    train()
