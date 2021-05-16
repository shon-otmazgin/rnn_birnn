import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader


def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  return xx_pad, yy_pad, x_lens, y_lens

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


class TagDataset(Dataset):
    def __init__(self, filename, return_y, tokens2ids=None, tags2ids=None):
        self.return_y = return_y
        data = read_data(filename, with_labels=return_y)
        if return_y:
            self.X, self.y, self.unique_x_toks, self.unique_y_toks = data
        else:
            self.X, self.unique_x_toks = data

        if tokens2ids is None:
            self.tokens2ids = {t: (i+1) for i, t in enumerate(self.unique_x_toks)}
            self.tokens2ids['<PAD>'] = 0
            self.tokens2ids['<UNK>'] = len(self.tokens2ids)
        else:
            self.tokens2ids = tokens2ids

        if tags2ids is None and return_y:
            self.tags2ids = {t: (i+1) for i, t in enumerate(self.unique_y_toks)}
            self.tags2ids['<PAD>'] = 0
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
        idxs = [self.tokens2ids[t] if t in self.tokens2ids else self.tokens2ids['<UNK>'] for t in x]
        return torch.tensor(idxs, dtype=torch.long)

    def _tensorize_y(self, y):
        idxs = [self.tags2ids[t] for t in y]
        return torch.tensor(idxs, dtype=torch.long)


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super(BiLSTMTagger, self).__init__()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.emb_size = 300
        self.s_dim = 768

        self.word_emb = nn.Embedding(self.vocab_size, self.emb_size)
        self.bilstm1 = nn.LSTM(bidirectional=True, input_size=self.emb_size, hidden_size=self.s_dim, num_layers=1, batch_first=True)
        self.bilstm2 = nn.LSTM(bidirectional=True, input_size=2*self.s_dim, hidden_size=self.s_dim, num_layers=1, batch_first=True)

        self.linear = nn.Linear(2*self.s_dim, tagset_size)

    def forward(self, input_ids, input_lens):
        embds = self.word_emb(input_ids)        # [batch, sequence_len, emb]

        x_packed = pack_padded_sequence(embds, input_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm1(x_packed)          # [batch, sequence_len, out_dim]
        lstm_out, _ = self.bilstm2(lstm_out)
        output_padded, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        output = self.linear(output_padded)
        return output


train_dataset = TagDataset('data/pos/train', return_y=True)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=pad_collate)

model = BiLSTMTagger(vocab_size=train_dataset.vocab_size, tagset_size=train_dataset.tagset_size)

for xx_pad, yy_pad, x_lens, y_lens in train_loader:
    out = model(xx_pad, x_lens)
    break

#
# dev_dataset = TagDataset('data/pos/dev', return_y=True,
#                          tokens2ids=train_dataset.tokens2ids,
#                          tags2ids=train_dataset.tags2ids)
# dev_ex = dev_dataset[0]
# print(dev_ex)
#
# test_dataset = TagDataset('data/pos/test', return_y=False,
#                           tokens2ids=train_dataset.tokens2ids,
#                           tags2ids=train_dataset.tags2ids)
# test_ex = test_dataset[0]
# print(test_ex)
