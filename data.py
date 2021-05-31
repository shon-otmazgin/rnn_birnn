import torch
import numpy as np
from torch.utils.data import Dataset
PAD = '<PAD>'
UNK = 'UUUNKKK'
UNK_PRE = 'UUUNKKK_PRE'
UNK_SUF = 'UUUNKKK_SUF'


def load_pretrained_embeds(vec_path, vocab_path):
    tokens2ids = {}
    with open(vocab_path, 'r') as f:
        for line in f.readlines():
            tokens2ids[line.strip()] = len(tokens2ids)
    tokens2ids[PAD] = len(tokens2ids)

    tmp = np.loadtxt(vec_path, dtype=np.float32)
    vecs = np.zeros((tmp.shape[0]+1, tmp.shape[1]), dtype=np.float32)
    vecs[:-1, :] = tmp

    return tokens2ids, torch.from_numpy(vecs)


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


def get_prefix_suffix(token):
    if len(token) > 2:
        return token[0:3], token[-3:]
    return UNK, UNK


class TagDataset(Dataset):
    def __init__(self, filename, return_y, tokens2ids=None, tags2ids=None, char2ids=None, pre2ids=None, suf2ids=None):
        self.tokens2ids = tokens2ids
        self.char2ids = char2ids
        self.pre2ids = pre2ids
        self.suf2ids = suf2ids
        self.tags2ids = tags2ids
        self.return_y = return_y

        data = read_data(filename, with_labels=return_y)
        if return_y:
            self.X, self.y, self.unique_x_toks, self.unique_y_toks = data
        else:
            self.X, self.unique_x_toks = data

        if self.tokens2ids is None:
            self.tokens2ids = {t: i for i, t in enumerate(self.unique_x_toks)}
            self.tokens2ids[UNK] = len(self.tokens2ids)
            self.tokens2ids[PAD] = len(self.tokens2ids)

        if self.tags2ids is None and return_y:
            self.tags2ids = {t: i for i, t in enumerate(self.unique_y_toks)}

        if self.char2ids is None:
            self.char2ids = {}
            for token in self.unique_x_toks:
                for c in token:
                    if c not in self.char2ids:
                        self.char2ids[c] = len(self.char2ids)
            self.char2ids[UNK] = len(self.char2ids)
            self.char2ids[PAD] = len(self.char2ids)

        if self.pre2ids is None and self.suf2ids is None:
            self.pre2ids, self.suf2ids = {}, {}
            for token in self.unique_x_toks:
                prefix, suffix = get_prefix_suffix(token)
                if prefix not in self.pre2ids:
                    self.pre2ids[prefix] = len(self.pre2ids)
                if suffix not in self.suf2ids:
                    self.suf2ids[suffix] = len(self.suf2ids)

            if UNK not in self.pre2ids:
                self.pre2ids[UNK] = len(self.pre2ids)
            self.pre2ids[PAD] = len(self.pre2ids)

            if UNK not in self.suf2ids:
                self.suf2ids[UNK] = len(self.suf2ids)
            self.suf2ids[PAD] = len(self.suf2ids)

        self.vocab_size = len(self.tokens2ids.keys())
        self.alphabet_size = len(self.char2ids.keys())
        self.pre_vocab_size = len(self.pre2ids.keys())
        self.suf_vocab_size = len(self.suf2ids.keys())
        self.tagset_size = len(self.tags2ids.keys())

        self.sentences = self.X
        self.X = [self._get_x_ids(x) for x in self.X]

        if return_y:
            self.y = [self._get_y_ids(y) for y in self.y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        tokens_ids, pre_ids, suf_ids, char_ids = self.X[item]
        if self.return_y:
            y_ids = self.y[item]
            return tokens_ids, pre_ids, suf_ids, char_ids, y_ids
        return tokens_ids, pre_ids, suf_ids, char_ids

    def _get_x_ids(self, x):
        char_ids = []
        pre_ids = []
        suf_ids = []
        for t in x:
            prefix, suffix = get_prefix_suffix(t)
            pre_ids.append(self.pre2ids[prefix] if prefix in self.pre2ids else self.pre2ids[UNK])
            suf_ids.append(self.suf2ids[suffix] if suffix in self.suf2ids else self.suf2ids[UNK])
            ids = torch.zeros(len(t), dtype=torch.long)
            for i, c in enumerate(t):
                ids[i] = self.char2ids[c] if c in self.char2ids else self.char2ids[UNK]
            char_ids.append(ids)

        tokens_ids = [self.tokens2ids[t] if t in self.tokens2ids else self.tokens2ids[UNK] for t in x]
        tokens_ids = torch.tensor(tokens_ids, dtype=torch.long)
        pre_ids = torch.tensor(pre_ids, dtype=torch.long)
        suf_ids = torch.tensor(suf_ids, dtype=torch.long)

        return tokens_ids, pre_ids, suf_ids, char_ids

    def _get_y_ids(self, y):
        y_ids = [self.tags2ids[t] for t in y]
        return torch.tensor(y_ids)