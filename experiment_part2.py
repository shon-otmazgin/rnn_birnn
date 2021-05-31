import random
import time
import os
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader


def is_prime(n):
    for i in range(3, n):
        if n % i == 0:
            return False
    return True


def gen_primes_dataset(n):
    examples = []

    for i in range(n):
        if is_prime(i):
            examples.append((str(i), 1))

    size = len(examples)
    for i in range(size):
        r = random.randint(1, n)
        while is_prime(r):
            r = random.randint(1, n)
        examples.append((str(r), 0))

    random.shuffle(examples)
    t = round(len(examples) * 0.7)
    return examples[0:t], examples[t:]



L = 100
def gen_01():
    s = ''
    l = random.randint(1, L)
    for i in range(l):
        s += str(random.randint(0, 1))
    return s

L = 100
def gen_19():
    s = ''
    l = random.randint(1, L)
    for i in range(l):
        s += str(random.randint(1, 9))
    return s


def gen_palindrome():
    s = ''
    l = random.randint(1, L)
    for i in range(round(l / 2)):
        s += str(random.randint(0, 1))
    odd = random.randint(0, 1)
    if odd:
        return s + ''.join(reversed(s))[1:]
    return s + ''.join(reversed(s))


def gen_palindrome_dataset(n):
    examples = []

    for i in range(n):
        s = gen_01()
        while isPalindrome(s) or len(s) <= 0:
            s = gen_01()
        examples.append((s, 0))

    size = len(examples)
    for i in range(size):
        p = gen_palindrome()
        while not isPalindrome(p) or len(p) <= 0:
            p = gen_palindrome()
        examples.append((p, 1))

    random.shuffle(examples)
    t = round(len(examples) * 0.7)
    return examples[0:t], examples[t:]


def isPalindrome(s):
    ''' check if a string is a Palindrome '''
    s = str(s)
    return s == s[::-1]


def gen_start_end_same_digit_dataset(n):
    examples = []

    for i in range(n):
        s = gen_19()
        while isDivided_by3(s) or len(s) <= 0:
            s = gen_19()
        examples.append((s, 0))

    size = len(examples)
    for i in range(size):
        p = gen_19()
        while not isDivided_by3(p) or len(p) <= 0:
            p = gen_19()
        examples.append((p, 1))

    random.shuffle(examples)
    t = round(len(examples) * 0.7)
    return examples[0:t], examples[t:]


def isDivided_by3(s):
    ''' check if a string start and end with the same digit'''
    return int(s) % 3 == 0


def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=10)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=10)

  return xx_pad, yy_pad, x_lens, y_lens


class LangDataset(Dataset):
    def __init__(self, examples):

        self.c2i = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '[PAD]': 10}

        self.examples = examples
        self.examples = [self._tensorize_example(e) for e in self.examples]

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
    def __init__(self, vocab_size, pad_idx):
        super(LangRNN, self).__init__()
        self.dropout = 0.3
        self.s_dim = 768
        self.emb_size = 50

        self.char_emb = nn.Embedding(vocab_size, self.emb_size, padding_idx=pad_idx)

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
        _, (h, c) = self.rnn(x_packed)          #[batch, sequence_len, out_dim]
        h = h.squeeze(0)                        #[batch, out_dim]
        return self.mlp(h)                      #[batch, 1]


def train(model, train_loader, test_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    eval = 100
    eval_time = 1
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
            if seen_examples >= (eval_time * eval):
                eval_time += 1
                train_acc = predict(model, train_loader, device)
                # train_acc = 0
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

    # trainset, testset = gen_primes_dataset(n=50000)
    # trainset, testset = gen_palindrome_dataset(n=1500)
    trainset, testset = gen_start_end_same_digit_dataset(n=2500)

    train_pos = sum([1 for x, y in trainset if y == 1])
    test_pos = sum([1 for x, y in testset if y == 1])

    train_min = min([len(x) for x, y in trainset])
    print(train_min)
    train_max = max([len(x) for x, y in trainset])
    print(train_max)

    print(f'Train: size: {len(trainset)}, pos: {train_pos}, neg: {len(trainset) - train_pos}')
    print(f'Test: size: {len(testset)}, pos: {test_pos}, neg: {len(testset) - test_pos}')
    print(testset)

    train_dataset = LangDataset(trainset)
    test_dataset = LangDataset(testset)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)

    model = LangRNN(vocab_size=len(train_dataset.c2i.keys()), pad_idx=train_dataset.c2i['[PAD]'])
    model.to(device)
    steps, train_losses, train_accuracies, test_accuracies = train(model, train_loader, test_loader, device)

    print(f'steps = {steps}')
    print(f'losses = {train_losses}')
    print(f'train_accuracies = {train_accuracies}')
    print(f'test_accuracies = {test_accuracies}')
