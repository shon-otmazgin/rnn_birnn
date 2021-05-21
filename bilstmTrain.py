import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm import tqdm

from data import PAD, TagDataset
from models import BiLSTMTaggerA, BiLSTMTaggerB


def pad_collate_chars(xx, max_pad):
    xx_pad = torch.stack([torch.cat([i, i.new_zeros(max_pad - i.size(0))], 0) for i in xx], 0)
    return xx_pad

def pad_collate(batch, token_pad, char_pad, y_pad):
  (xx_tokens, xx_chars, yy) = zip(*batch)
  x_tokens_lens = [len(x) for x in xx_tokens]
  x_chars_lens = [[len(x) for x in tokens] for tokens in xx_chars]
  y_lens = [len(y) for y in yy]

  xx_tokens_pad = pad_sequence(xx_tokens, batch_first=True, padding_value=token_pad)

  max_token_len = max(max(lens) for lens in x_chars_lens)
  xx_chars_inside_pad = [pad_collate_chars(xx, max_token_len) for xx in xx_chars]
  xx_chars_pad = pad_sequence(xx_chars_inside_pad, batch_first=True, padding_value=char_pad)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=y_pad)

  return xx_tokens_pad, xx_chars_pad, yy_pad, x_tokens_lens, x_chars_lens, y_lens


def train(model, train_loader, dev_loader, device, y_pad):
    criterion = nn.CrossEntropyLoss(ignore_index=y_pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss = 0
    seen_sents = 0
    best_acc = 0

    train_iterator = trange(0, 5, desc="Epoch", position=0)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration", position=0)
        for step, (xx_pad, yy_pad, x_lens, y_lens) in enumerate(epoch_iterator):
            model.train()
            model.zero_grad()
            input_ids, y = xx_pad.to(device), yy_pad.to(device)

            logits = model(input_ids, x_lens)  # [batch, seq_len, tagset]
            logits = logits.permute(0, 2, 1)  # [batch, tagset, seq_len]
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            seen_sents += input_ids.shape[0]

            if seen_sents % 500 == 0:
                acc = predict(model, dev_loader, device, y_pad)
                print(f'Train loss: {(loss / 500):.8f}')
                print(f'Dev acc:{acc:.8f}')
                if acc > best_acc:
                    best_acc = acc
                print(f'Best Dev acc:{best_acc:.8f}')
                train_loss = 0

    torch.save(model.state_dict(), 'bilstm.pt')


def predict(model, loader, device, y_pad):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for step, (xx_pad, yy_pad, x_lens, y_lens) in enumerate(loader):
            input_ids, y = xx_pad.to(device), yy_pad.to(device)

            logits = model(input_ids, x_lens)  # [batch, seq_len, tagset]
            probs = logits.softmax(dim=2).argmax(dim=2)
            mask = (y != y_pad)

            correct += probs[mask].eq(y[mask]).sum()
            total += sum(y_lens)
    return correct.item() / total.item()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running device: {device}')


    # section A
    # train_dataset = DatasetA('data/pos/train', return_y=True)
    # x_pad, y_pad = train_dataset.tokens2ids[PAD], len(train_dataset.tags2ids)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
    #                           collate_fn=lambda b: pad_collate(b, x_pad, y_pad))
    #
    # dev_dataset = DatasetA('data/pos/dev', return_y=True,
    #                        tokens2ids=train_dataset.tokens2ids,
    #                        tags2ids=train_dataset.tags2ids)
    # dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False,
    #                         collate_fn=lambda b: pad_collate(b, x_pad, y_pad))
    #
    # model = BiLSTMTaggerA(vocab_size=train_dataset.vocab_size,
    #                       tagset_size=train_dataset.tagset_size,
    #                       padding_idx=x_pad)
    # model.to(device)

    # section B
    train_dataset = TagDataset('data/pos/dev', return_y=True)
    token_pad, char_pad, y_pad = train_dataset.tokens2ids[PAD], train_dataset.char2ids[PAD], len(train_dataset.tags2ids)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              collate_fn=lambda b: pad_collate(b, token_pad, char_pad, y_pad))
    for step, (xx_tokens_pad, xx_chars_pad, yy_pad, x_tokens_lens, x_chars_lens, y_lens) in enumerate(train_loader):
        print(xx_tokens_pad.shape)
        print(xx_chars_pad.shape)
        print(yy_pad.shape)
        print(x_tokens_lens)
        print(x_chars_lens)
        print(y_lens)
    # dev_dataset = TagDataset('data/pos/dev', return_y=True,
    #                          tokens2ids=train_dataset.tokens2ids,
    #                          tags2ids=train_dataset.tags2ids)
    # dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    #
    # model = BiLSTMTaggerB(vocab_size=train_dataset.alphabet_size,
    #                       tagset_size=train_dataset.tagset_size,
    #                       padding_idx=token_pad)
    # model.to(device)
    #
    # train(model, train_loader, dev_loader, device, y_pad)