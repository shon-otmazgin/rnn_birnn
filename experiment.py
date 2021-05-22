import random
import time
import os
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader


def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=13)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=13)

  return xx_pad, yy_pad, x_lens, y_lens

class LangDataset(Dataset):
    def __init__(self, pos_path, neg_path):
        self.c2i = {'a': 0, 'b': 1, 'c': 2, 'd': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12, '[PAD]': 13}

        self.examples = []
        with open(pos_path, 'r') as f:
            for ex in f.readlines():
                self.examples.append((ex.strip(), 1))
        with open(neg_path, 'r') as f:
            for ex in f.readlines():
                self.examples.append((ex.strip(), 0))
        self.examples = [self._tensorize_example(e) for e in self.examples]
        # random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def _tensorize_example(self, example):
        x, y = example
        ids = [self.c2i[c] for c in x]
        y = torch.zeros(1) if y == 0 else torch.ones(1)
        return torch.tensor(ids, dtype=torch.long), y


class LangRNN(nn.Module):
    def __init__(self):
        super(LangRNN, self).__init__()
        self.dropout = 0.3
        self.s_dim = 768
        self.emb_size = 300
        self.vocab_size = 14

        self.char_emb = nn.Embedding(self.vocab_size, self.emb_size)

        self.rnn = nn.LSTM(input_size=self.emb_size, hidden_size=self.s_dim, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.s_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 1)
        )

    def forward(self, input_ids, input_lens):
        embds = self.char_emb(input_ids)        #[batch, sequence_len, emb]
        x_packed = pack_padded_sequence(embds, input_lens, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.rnn(x_packed)          #[batch, 1, out_dim]
        h = h.squeeze(0)                        #[batch, out_dim]
        return self.mlp(h)                      #[batch, 1]


def train(model, train_loader, test_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    seen_examples = 0
    steps = []
    train_loss = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    train_iterator = trange(0, 5, desc="Epoch", position=0)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration", position=0)
        for step, (xx_pad, yy_pad, x_lens, y_lens) in enumerate(epoch_iterator):
            model.train()
            model.zero_grad()
            input_ids, y = xx_pad.to(device), yy_pad.to(device)

            logits = model(input_ids, x_lens)   #[batch, 1]
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            seen_examples += input_ids.shape[0]
            if seen_examples % 100 == 0:
                train_acc = predict(model, train_loader, device)
                test_acc = predict(model, test_loader, device)
                print()
                print(f'Train loss: {(loss / 100):.8f}')
                print(f'Train acc:{train_acc:.8f}')
                print(f'Test acc:{test_acc:.8f}')
                steps.append(seen_examples)
                train_accuracies.append(train_acc)
                train_losses.append(train_loss)
                test_accuracies.append(test_acc)

                train_loss = 0

    return steps, train_losses, train_accuracies, test_accuracies


def predict(model, loader, device):
    correct = 0

    model.eval()
    with torch.no_grad():
        for xx_pad, yy_pad, x_lens, y_lens in loader:
            input_ids, y = xx_pad.to(device), yy_pad.to(device)

            logits = model(input_ids, x_lens)  # [batch, 1]
            preds = logits.sigmoid().round()   # get the index of the max log-probability/logits
            correct += preds.eq(y).sum().item()

    return correct / len(loader.dataset)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running device: {device}')

    train_dataset = LangDataset('pos_train', 'neg_train')
    test_dataset = LangDataset('pos_test', 'neg_test')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)

    model = LangRNN()
    model.to(device)
    steps, train_losses, train_accuracies, test_accuracies = train(model, train_loader, test_loader, device)

    print(f'steps = {steps}')
    print(f'losses = {train_losses}')
    print(f'train_accuracies = {train_accuracies}')
    print(f'test_accuracies = {test_accuracies}')
