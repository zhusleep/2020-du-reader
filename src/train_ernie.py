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
from utils import *
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from pathlib import Path

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
    # model = RobertaForQuestionAnswering.from_pretrained('../lm_pretrained/RoBERTa_zh_L12_PyTorch/pytorch_model.bin')

    VOCAB_PATH = '../lm_pretrained/ernie/vocab.txt'
    #VOCAB_PATH = Path(VOCAB_PATH)
    tokenizer = BertTokenizer.from_pretrained(
                    VOCAB_PATH, cache_dir=None, do_lower_case=True)
    train_set = ReaderDataset(train_data,train=True, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_set, batch_size=config.batch_size,
                                  shuffle=True, num_workers=0, collate_fn=collate_fn_train)
    # 开发集用于验证
    dev_set = ReaderDataset(dev_data, train=True, tokenizer=tokenizer)
    dev_dataloader = DataLoader(dev_set, batch_size=config.batch_size,
                                  shuffle=False, num_workers=0, collate_fn=collate_fn_train)
    # 2 载入模型
    # 加载预训练bert
    # model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
    MODEL_PATH = '../lm_pretrained/ernie/'
    model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
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
            q_ids, input_ids, segment_ids, start_positions, end_positions = batch
            input_mask = (input_ids > 0).to(device)
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
                # print(loss.item())

            if (step + 1) % 1000 == 0:
                # 5 在开发集上验证
                validate_dev(model, dev_dataloader)
        metrics = validate_dev(model, dev_dataloader)
        torch.save(model.state_dict(), "../model/ernie_epoch_%s_f1_%s.pt" % (str(i), str(metrics['F1'])))


# 开发集上预测验证函数
def validate_dev(model, dev_data_loader):
    total, losses = 0.0, []
    device = config.device

    with torch.no_grad():
        model.eval()
        pred_results = {}
        for batch in dev_data_loader:

            q_ids, input_ids, segment_ids, start_positions, end_positions = batch
            input_ids, segment_ids, start_positions, end_positions = \
            input_ids.to(device), segment_ids.to(device), start_positions.to(
                device), end_positions.to(device)
            input_mask = (input_ids > 0).to(device)
            start_prob, end_prob = model(input_ids.to(device),
                                         token_type_ids=segment_ids.to(device),
                                         attention_mask=input_mask.to(device)
                                        )
            # start_prob = start_prob.squeeze(0)
            # end_prob = end_prob.squeeze(0)
            for i in range(len(batch[0])):
                try:
                    (best_start, best_end), max_prob = find_best_answer_for_passage(start_prob[i], end_prob[i])
                except:
                    pass
                pred_results[q_ids[i]] = (best_start.cpu().numpy()[0], best_end.cpu().numpy()[0])
        submit = {}
        for item in dev_data:
            q_id = item[0]
            context = item[1]
            question = item[2]
            new_sentence = '.'+question+'。'+context
            submit[q_id] = new_sentence[pred_results[q_id][0]:pred_results[q_id][1]]
            print(question, new_sentence[pred_results[q_id][0]:pred_results[q_id][1]])

        submit_path = '../submit/submit.json'
        metrics = evaluate(submit, submit_path)
        print(metrics)
        return metrics
        # return total / len(losses)


def evaluate(submit_dict, submit_path):
    """评测函数（官方提供评测脚本evaluate.py）
    """
    predict_to_file(submit_dict, submit_path)
    metrics = json.loads(
        os.popen(
            'python3 ../evaluate.py %s %s'
            % ('../data/dev.json', submit_path)
        ).read().strip()
    )
    return metrics


def predict_to_file(submit_dict, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    R = json.dumps(submit_dict, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()


if __name__ == "__main__":
    # 1 载入数据
    train_data = load_data('../data/train.json')
    dev_data = load_data('../data/dev.json')
    train()
