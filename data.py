import torch
from torch.utils.data import Dataset
PAD = '<PAD>'
UNK = '<UNK>'


def read_data(filename, with_labels):
    X = []
    y = []
    unique_x_toks = set()
    unique_y_toks = set()
    with open(filename, 'r', encoding='utf8') as f:
        x_sent = []
        y_sent = []
        for word in f:
            word = word.rstrip('\n')

            if '\t' in word:
                word = word.replace("\t", " ")
            if word:
                if with_labels:
                    x_y = word.split(" ")
                    x_sent.append(x_y[0])
                    y_sent.append(x_y[1])
                    unique_x_toks.add(x_y[0])
                    unique_y_toks.add(x_y[1])
                else:
                    x_sent.append(word)
                    unique_x_toks.add(word)
            else:
                X.append(x_sent)
                x_sent = []
                if with_labels:
                    y.append(y_sent)
                    y_sent = []

    if with_labels:
        return X, y, unique_x_toks, unique_y_toks
    else:
        return X, unique_x_toks


class DatasetA(Dataset):
    def __init__(self, filename, return_y, tokens2ids=None, tags2ids=None):
        self.return_y = return_y
        data = read_data(filename, with_labels=return_y)
        if return_y:
            self.X, self.y, self.unique_x_toks, self.unique_y_toks = data
        else:
            self.X, self.unique_x_toks = data

        if tokens2ids is None:
            self.tokens2ids = {t: i for i, t in enumerate(self.unique_x_toks)}
            self.tokens2ids[UNK] = len(self.tokens2ids)
            self.tokens2ids[PAD] = len(self.tokens2ids)
        else:
            self.tokens2ids = tokens2ids

        if tags2ids is None and return_y:
            self.tags2ids = {t: i for i, t in enumerate(self.unique_y_toks)}
        else:
            self.tags2ids = tags2ids

        self.vocab_size = len(self.tokens2ids.keys())
        self.tagset_size = len(self.tags2ids.keys())

        self.X = [self._tensorize_x(x) for x in self.X]
        if return_y:
            self.y = [self._tensorize_y(y) for y in self.y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if self.return_y:
            return self.X[item], self.y[item]
        return self.X[item]

    def _tensorize_x(self, x):
        idxs = [self.tokens2ids[t] if t in self.tokens2ids else self.tokens2ids[UNK] for t in x]
        return torch.tensor(idxs, dtype=torch.long)

    def _tensorize_y(self, y):
        idxs = [self.tags2ids[t] for t in y]
        return torch.tensor(idxs)


# class BDataset(Dataset):
#     def __init__(self, filename, return_y, tokens2ids=None, tags2ids=None):
#         self.return_y = return_y
#         data = read_data(filename, with_labels=return_y)
#         if return_y:
#             self.X, self.y, self.unique_x_toks, self.unique_y_toks = data
#         else:
#             self.X, self.unique_x_toks = data
#
#         if tokens2ids is None:
#             self.tokens2ids = {t: i for i, t in enumerate(self.unique_x_toks)}
#             self.tokens2ids[UNK] = len(self.tokens2ids)
#             self.tokens2ids[PAD] = len(self.tokens2ids)
#         else:
#             self.tokens2ids = tokens2ids
#
#         if tags2ids is None and return_y:
#             self.tags2ids = {t: i for i, t in enumerate(self.unique_y_toks)}
#         else:
#             self.tags2ids = tags2ids
#
#         self.char2ids = {}
#         for token in self.tokens2ids:
#             for c in token:
#                 if c not in self.char2ids:
#                     self.char2ids[c] = len(self.char2ids)
#
#         self.vocab_size = len(self.tokens2ids.keys())
#         self.alphabet_size = len(self.char2ids.keys())
#         self.tagset_size = len(self.tags2ids.keys())
#
#         self.X = [self._tensorize_x(x) for x in self.X]
#         if return_y:
#             self.y = [self._tensorize_y(y) for y in self.y]
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, item):
#         if self.return_y:
#             return self.X[item], self.y[item]
#         return self.X[item]
#
#     def _tensorize_x(self, x):
#         idxs = [self.tokens2ids[t] if t in self.tokens2ids else self.tokens2ids[UNK] for t in x]
#         return torch.tensor(idxs, dtype=torch.long)
#
#     def _tensorize_y(self, y):
#         idxs = [self.tags2ids[t] for t in y]
#         return torch.tensor(idxs)