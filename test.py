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


data = load_data("data/dev.json")
print(len(data))