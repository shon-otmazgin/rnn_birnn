import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm import tqdm

from data import PAD, TagDataset
from models import BiLSTMTaggerA, BiLSTMTagger


def pad_collate_chars(xx, max_pad, pad_value):
    l = []
    for i in xx:
        pad = i.new_zeros(max_pad - i.size(0))
        pad += pad_value
        l.append(torch.cat([i, pad]))

    return torch.stack(l, 0)


def pad_collate(batch, token_pad, char_pad, y_pad):
  (xx_tokens, xx_chars, yy) = zip(*batch)
  x_tokens_lens = [len(x) for x in xx_tokens]
  x_chars_lens = [[len(x) for x in tokens] for tokens in xx_chars]
  for l in x_chars_lens:
      for i in range(max(x_tokens_lens)-len(l)):
          l.append(0)
  y_lens = [len(y) for y in yy]

  xx_tokens_pad = pad_sequence(xx_tokens, batch_first=True, padding_value=token_pad)

  max_token_len = max(max(lens) for lens in x_chars_lens)
  xx_chars_inside_pad = [pad_collate_chars(xx, max_token_len, char_pad) for xx in xx_chars]
  xx_chars_pad = pad_sequence(xx_chars_inside_pad, batch_first=True, padding_value=char_pad)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=y_pad)

  return xx_tokens_pad, xx_chars_pad, yy_pad, x_tokens_lens, x_chars_lens, y_lens


def train(model, train_loader, dev_loader, device, y_pad):
    criterion = nn.CrossEntropyLoss(ignore_index=y_pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss = 0
    seen_sents = 0
    best_acc = 0

    accuracies = []
    steps = []

    train_iterator = trange(0, 5, desc="Epoch", position=0)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration", position=0)
        for step, (xx_tokens_pad, xx_chars_pad, yy_pad, x_tokens_lens, x_chars_lens, y_lens) in enumerate(epoch_iterator):
            model.train()
            model.zero_grad()
            tokens_input_ids = xx_tokens_pad.to(device)
            char_input_ids = xx_chars_pad.to(device)
            y = yy_pad.to(device)

            logits = model(tokens_input_ids, char_input_ids, x_tokens_lens, x_chars_lens)  # [batch, seq_len, tagset]
            logits = logits.permute(0, 2, 1)  # [batch, tagset, seq_len]
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            seen_sents += tokens_input_ids.shape[0]

            if seen_sents % 500 == 0:
                acc = predict(model, dev_loader, device, y_pad)
                print(f'Train loss: {(loss / 500):.8f}')
                print(f'Dev acc:{acc:.8f}')
                if acc > best_acc:
                    best_acc = acc
                print(f'Best Dev acc:{best_acc:.8f}')
                accuracies.append(acc)
                steps.append(seen_sents)
                train_loss = 0

    return best_acc, accuracies, steps
    # torch.save(model.state_dict(), 'bilstm.pt')


def predict(model, loader, device, y_pad):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for step, (xx_tokens_pad, xx_chars_pad, yy_pad, x_tokens_lens, x_chars_lens, y_lens) in enumerate(loader):
            tokens_input_ids = xx_tokens_pad.to(device)
            char_input_ids = xx_chars_pad.to(device)
            y = yy_pad.to(device)

            logits = model(tokens_input_ids, char_input_ids, x_tokens_lens, x_chars_lens)  # [batch, seq_len, tagset]
            probs = logits.softmax(dim=2).argmax(dim=2)
            mask = (y != y_pad)

            correct += probs[mask].eq(y[mask]).sum()
            total += sum(y_lens)
    return correct.item() / total


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running device: {device}')

    train_dataset = TagDataset('data/pos/train', return_y=True)
    token_pad, char_pad, y_pad = train_dataset.tokens2ids[PAD], train_dataset.char2ids[PAD], len(train_dataset.tags2ids)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              collate_fn=lambda b: pad_collate(b, token_pad, char_pad, y_pad))

    dev_dataset = TagDataset('data/pos/dev', return_y=True,
                             tokens2ids=train_dataset.tokens2ids,
                             tags2ids=train_dataset.tags2ids)
    dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False,
                            collate_fn=lambda b: pad_collate(b, token_pad, char_pad, y_pad))

    model = BiLSTMTagger(vocab_size=train_dataset.vocab_size,
                         alphabet_size=train_dataset.alphabet_size,
                         tagset_size=train_dataset.tagset_size,
                         token_padding_idx=token_pad,
                         char_padding_idx=char_pad,
                         char_level=False,
                         token_level=True)
    model.to(device)
    a_best_acc, a_accuracies, a_steps = train(model, train_loader, dev_loader, device, y_pad)

    model = BiLSTMTagger(vocab_size=train_dataset.vocab_size,
                         alphabet_size=train_dataset.alphabet_size,
                         tagset_size=train_dataset.tagset_size,
                         token_padding_idx=token_pad,
                         char_padding_idx=char_pad,
                         char_level=True,
                         token_level=False)
    model.to(device)
    b_best_acc, b_accuracies, b_steps = train(model, train_loader, dev_loader, device, y_pad)

    model = BiLSTMTagger(vocab_size=train_dataset.vocab_size,
                         alphabet_size=train_dataset.alphabet_size,
                         tagset_size=train_dataset.tagset_size,
                         token_padding_idx=token_pad,
                         char_padding_idx=char_pad,
                         char_level=True,
                         token_level=True)
    model.to(device)
    d_best_acc, d_accuracies, d_steps = train(model, train_loader, dev_loader, device, y_pad)

    print(f'steps = {a_steps}')
    print(f'a = {a_accuracies}')
    print(f'b = {b_accuracies}')
    print(f'c = {d_accuracies}')

    print(f'a: {a_best_acc}\nb: {b_best_acc}\na: {d_best_acc}')