import argparse
import random
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pickle

from data import PAD, TagDataset, load_pretrained_embeds
from models import BiLSTMTagger


def pad_collate_chars(xx, max_pad, pad_value):
    l = []
    for i in xx:
        pad = i.new_zeros(max_pad - i.size(0))
        pad += pad_value
        l.append(torch.cat([i, pad]))

    return torch.stack(l, 0)


def pad_collate(batch, token_pad, pre_pad, suf_pad, char_pad, y_pad):
    (xx_tokens, xx_pre, xx_suf, xx_chars, yy) = zip(*batch)
    x_tokens_lens = [len(x) for x in xx_tokens]
    x_chars_lens = [[len(x) for x in tokens] for tokens in xx_chars]
    for l in x_chars_lens:
      for i in range(max(x_tokens_lens)-len(l)):
          l.append(0)
    y_lens = [len(y) for y in yy]

    xx_tokens_pad = pad_sequence(xx_tokens, batch_first=True, padding_value=token_pad)
    xx_pre_pad = pad_sequence(xx_pre, batch_first=True, padding_value=pre_pad)
    xx_suf_pad = pad_sequence(xx_suf, batch_first=True, padding_value=suf_pad)

    assert xx_pre_pad.size() == xx_suf_pad.size() == xx_tokens_pad.size()

    max_token_len = max(max(lens) for lens in x_chars_lens)
    xx_chars_inside_pad = [pad_collate_chars(xx, max_token_len, char_pad) for xx in xx_chars]
    xx_chars_pad = pad_sequence(xx_chars_inside_pad, batch_first=True, padding_value=char_pad)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=y_pad)

    return xx_tokens_pad, xx_pre_pad, xx_suf_pad, xx_chars_pad, yy_pad, x_tokens_lens, x_chars_lens, y_lens


def train(model, train_loader, dev_loader, device, y_pad, o_id, model_path):
    criterion = nn.CrossEntropyLoss(ignore_index=y_pad)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=1e-3)

    train_loss = 0
    seen_sents = 0
    best_acc = 0
    eval = 500
    eval_time = 1

    accuracies = []
    steps = []
    state_dict = None

    train_iterator = range(5)
    for epoch in train_iterator:
        for step, (xx_tokens_pad, xx_pre_pad, xx_suf_pad, xx_chars_pad, yy_pad, x_tokens_lens, x_chars_lens, y_lens) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            tokens_input_ids = xx_tokens_pad.to(device)
            pre_input_ids = xx_pre_pad.to(device)
            suf_input_ids = xx_suf_pad.to(device)
            char_input_ids = xx_chars_pad.to(device)
            y = yy_pad.to(device)

            logits = model(tokens_input_ids, pre_input_ids, suf_input_ids, char_input_ids, x_tokens_lens, x_chars_lens)  # [batch, seq_len, tagset]
            logits = logits.permute(0, 2, 1)  # [batch, tagset, seq_len]
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            seen_sents += tokens_input_ids.shape[0]

            if seen_sents >= (eval_time * eval):
                eval_time += 1
                print()
                if dev_loader:
                    acc = predict(model, dev_loader, device, y_pad, o_id)
                    print(f'Dev acc:{acc:.8f}')
                    if acc > best_acc:
                        best_acc = acc
                        state_dict = model.state_dict()
                    print(f'Best Dev acc:{best_acc:.8f}')
                    accuracies.append(acc)
                    steps.append(seen_sents)

                print(f'Train loss: {(loss / 500):.8f}')
                train_loss = 0
    if dev_loader:
        acc = predict(model, dev_loader, device, y_pad, o_id)
        print(f'Dev acc:{acc:.8f}')
        if acc > best_acc:
            best_acc = acc
            state_dict = model.state_dict()
        print(f'Best Dev acc:{best_acc:.8f}')
        accuracies.append(acc)
        steps.append(seen_sents)
    if state_dict is None:
        state_dict = model.state_dict()
    return best_acc, accuracies, steps, state_dict


def predict(model, loader, device, y_pad, o_id):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for step, (xx_tokens_pad, xx_pre_pad, xx_suf_pad, xx_chars_pad, yy_pad, x_tokens_lens, x_chars_lens, y_lens) in enumerate(loader):
            tokens_input_ids = xx_tokens_pad.to(device)
            pre_input_ids = xx_pre_pad.to(device)
            suf_input_ids = xx_suf_pad.to(device)
            char_input_ids = xx_chars_pad.to(device)
            y = yy_pad.to(device)

            logits = model(tokens_input_ids, pre_input_ids, suf_input_ids, char_input_ids, x_tokens_lens, x_chars_lens)  # [batch, seq_len, tagset]
            probs = logits.softmax(dim=2).argmax(dim=2)

            mask = (y != y_pad)
            probs = probs[mask]
            y = y[mask]

            equals = probs.eq(y)

            both_o_scores = torch.logical_and(equals, y == o_id)

            correct += equals.sum() - both_o_scores.sum()
            total += equals.shape[0] - both_o_scores.sum()

    return (correct / total).item()


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='LSTM Tagger')
    parser.add_argument('repr', metavar='repr', type=str, help='one of a,b,c,d')
    parser.add_argument('trainFile', type=str, help='input file to train on')
    parser.add_argument('modelFile', type=str, help='file to save the model')
    parser.add_argument("--devFile", dest='dev_path', type=str, help='dev file to calc acc during train')
    parser.add_argument("--vecFile", dest='vec_path', type=str, help='file to pretrained vectors')
    parser.add_argument("--vocabFile", dest='vocab_path', type=str, help='file to the vocab of pretrained vectors')
    parser.add_argument("--batchSize", dest='batch_size', type=int, help='batch size to train. Default:5', default=5)

    args = parser.parse_args()

    method = args.repr
    train_path = args.trainFile
    dev_path = args.dev_path
    model_path = args.modelFile
    vec_path = args.vec_path
    vocab_path = args.vocab_path
    batch_size = args.batch_size

    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running device: {device}')

    tokens2ids, pretrained_vecs = None, None
    if method == 'a':
        token_level, pre_suf_level, char_level = True, False, False
    elif method == 'b':
        token_level, pre_suf_level, char_level = False, False, True
    elif method == 'c':
        token_level, pre_suf_level, char_level = True, True, False
        if vec_path is not None and vocab_path is not None:
            tokens2ids, pretrained_vecs = load_pretrained_embeds(vec_path, vocab_path)
    elif method == 'd':
        token_level, pre_suf_level, char_level = True, False, True

    train_dataset = TagDataset(train_path, return_y=True, tokens2ids=tokens2ids)
    token_pad = train_dataset.tokens2ids[PAD]
    pre_pad = train_dataset.pre2ids[PAD]
    suf_pad = train_dataset.suf2ids[PAD]
    char_pad = train_dataset.char2ids[PAD]
    y_pad = len(train_dataset.tags2ids)
    o_id = train_dataset.tags2ids['O'] if 'O' in train_dataset.tags2ids else y_pad
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: pad_collate(b, token_pad, pre_pad, suf_pad, char_pad, y_pad))

    if dev_path:
        dev_dataset = TagDataset(dev_path, return_y=True,
                                 tokens2ids=train_dataset.tokens2ids,
                                 char2ids=train_dataset.char2ids,
                                 pre2ids=train_dataset.pre2ids,
                                 suf2ids=train_dataset.suf2ids,
                                 tags2ids=train_dataset.tags2ids)
        dev_loader = DataLoader(dev_dataset, batch_size=500, shuffle=False,
                                collate_fn=lambda b: pad_collate(b, token_pad, pre_pad, suf_pad, char_pad, y_pad))
    else:
        dev_loader = None

    model = BiLSTMTagger(vocab_size=train_dataset.vocab_size,
                         pre_vocab_size=train_dataset.pre_vocab_size,
                         suf_vocab_size=train_dataset.suf_vocab_size,
                         alphabet_size=train_dataset.alphabet_size,
                         tagset_size=train_dataset.tagset_size,
                         token_padding_idx=token_pad,
                         pre_padding_idx=pre_pad,
                         suf_padding_idx=suf_pad,
                         char_padding_idx=char_pad,
                         token_level=token_level,
                         pre_suf_level=pre_suf_level,
                         char_level=char_level,
                         pretrained_vecs=pretrained_vecs)
    model.to(device)
    best_acc, accuracies, steps, state_dict = train(model, train_loader, dev_loader, device, y_pad, o_id, model_path)

    print(f'steps = {steps}')
    print(f'method_{method} = {accuracies}')
    print(f'method_{method} best_acc_dev: {best_acc}')

    print(f'Saving model and train dataset to: {model_path} is a checkpoint dict')
    torch.save({
        'model_state_dict': state_dict,
        'tokens2ids': train_dataset.tokens2ids,
        'char2ids': train_dataset.char2ids,
        'pre2ids': train_dataset.pre2ids,
        'suf2ids': train_dataset.suf2ids,
        'tag2ids': train_dataset.tags2ids,
    }, model_path)
