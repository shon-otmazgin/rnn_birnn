import random
import torch
from torch import nn
from torch.utils.data import Dataset


class LangDataset(Dataset):
    def __init__(self, pos_path, neg_path):
        self.c2i = {'a': 0, 'b': 1, 'c': 2, 'd': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12}

        self.examples = []
        with open(pos_path, 'r') as f:
            for ex in f.readlines():
                self.examples.append((ex.strip(), 1))
        with open(neg_path, 'r') as f:
            for ex in f.readlines():
                self.examples.append((ex.strip(), 0))
        self.examples = [self._tensorize_example(e) for e in self.examples]
        random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def _tensorize_example(self, example):
        x, y = example
        idxs = [self.c2i[c] for c in x]
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LangRNN(nn.Module):
    def __init__(self):
        super(LangRNN, self).__init__()
        self.dropout = 0.3
        self.s_dim = 768
        self.emb_size = 300
        self.vocab_size = 13

        self.char_emb = nn.Embedding(self.vocab_size, self.emb_size)

        self.rnn = nn.LSTM(input_size=self.emb_size, hidden_size=self.s_dim, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.s_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids):
        print(input_ids.shape)
        embds = self.char_emb(input_ids) #[sequence, emb]
        print(embds.shape)
        embds = embds.unsqueeze(0) #[1, sequence, emb]
        print(embds.shape)
        _, (h, c) = self.rnn(embds) #[1, sequence, out_dim]
        print(h.shape)
        h = h.squeeze(1)
        print(h.shape)
        probs = self.mlp(h)
        print(probs.shape)
        print(probs)





train_dataset = LangDataset('train_pos', 'train_neg')
print(len(train_dataset))

model = LangRNN()
x, y = train_dataset[4]
model(x)