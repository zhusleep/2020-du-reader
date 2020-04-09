import config
import torchtext
from torchtext import data
from torch.utils.data import DataLoader


def x_tokenize(ids):
    return [int(i) for i in ids]


def y_tokenize(y):
    return int(y)


def str_tokenize(s):
    return str(s)


class Dureader():
    def __init__(self, path='../featured_data'):

        self.WORD = torchtext.data.Field(batch_first=True, sequential=True, tokenize=x_tokenize,
                                         use_vocab=False, pad_token=0)
        self.LABEL = torchtext.data.Field(sequential=False, tokenize=y_tokenize, use_vocab=False)

        self.ID = torchtext.data.Field(sequential=False, tokenize=str_tokenize, use_vocab=False)

        dict_fields = {'input_ids': ('input_ids', self.WORD),
                       'input_mask': ('input_mask', self.WORD),
                       'segment_ids': ('segment_ids', self.WORD),
                       'start_position': ('start_position', self.LABEL),
                       'end_position': ('end_position', self.LABEL),
                       'qas_id': ('qas_id', self.ID)}

        self.train, self.dev = torchtext.data.TabularDataset.splits(
                path=path,
                train="train.data",
                validation="dev.data",
                format='json',
                fields=dict_fields)
        self.train_iter, self.dev_iter = torchtext.data.BucketIterator.splits(
                                                                    [self.train, self.dev],  batch_size=config.batch_size,
                                                                    sort_key=lambda x: len(x.input_ids), sort_within_batch=True, shuffle=True)
