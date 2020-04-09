import os
import config
import torch
import random
from dataloader import Dureader
# from dataset.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertForQuestionAnswering, BertConfig
# 随机种子
random.seed(config.seed)
torch.manual_seed(config.seed)


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


def evaluate():
    # 加载预训练bert
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
    device = config.device
    model.load_state_dict(torch.load("final.pt"))
    model.to(device)

    # 准备数据
    data = Dureader()
    train_dataloader, dev_dataloader = data.train_iter, data.dev_iter

    total, losses = 0.0, []
    device = config.device
    start_probs, end_probs = [], []

    with torch.no_grad():
        model.eval()
        for batch in dev_dataloader:
            print('ok')
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            start_prob, end_prob = model(input_ids.to(device), token_type_ids=segment_ids.to(device),
                                         attention_mask=input_mask.to(device)
                                         )
            start_probs.append(start_prob.squeeze(0))
            end_probs.append(end_prob.squeeze(0))
            for i in range(4):
                (best_start, best_end), max_prob = find_best_answer_for_passage(start_probs[0][0], end_probs[0][0])


if __name__ == "__main__":
    evaluate()
