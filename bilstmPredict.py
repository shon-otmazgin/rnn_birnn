import torch
import argparse
import random
import pickle
from torch.utils.data import DataLoader
from data import PAD, load_pretrained_embeds, TagDataset
from models import BiLSTMTagger
from torch.nn.utils.rnn import pad_sequence


def predict(model, loader, device, tag2ids):
    ids2tags = {v: k for k, v in tag2ids.items()}
    preds = []

    model.eval()
    with torch.no_grad():
        for step, (xx_tokens_pad, xx_pre_pad, xx_suf_pad, xx_chars_pad, x_tokens_lens, x_chars_lens) in enumerate(loader):
            tokens_input_ids = xx_tokens_pad.to(device)
            pre_input_ids = xx_pre_pad.to(device)
            suf_input_ids = xx_suf_pad.to(device)
            char_input_ids = xx_chars_pad.to(device)

            logits = model(tokens_input_ids, pre_input_ids, suf_input_ids, char_input_ids, x_tokens_lens, x_chars_lens)  # [batch, seq_len, tagset]
            probs = logits.softmax(dim=2).argmax(dim=2) # [batch, seq_len]
            for b in range(probs.shape[0]):
                sentence = probs[b]
                sentence_len = x_tokens_lens[b]
                tag_ids = sentence[0:sentence_len]
                preds.append([ids2tags[t_id.item()] for t_id in tag_ids])

    return preds



def pad_collate_chars(xx, max_pad, pad_value):
    l = []
    for i in xx:
        pad = i.new_zeros(max_pad - i.size(0))
        pad += pad_value
        l.append(torch.cat([i, pad]))

    return torch.stack(l, 0)


def pad_collate(batch, token_pad, pre_pad, suf_pad, char_pad):
    (xx_tokens, xx_pre, xx_suf, xx_chars) = zip(*batch)
    x_tokens_lens = [len(x) for x in xx_tokens]
    x_chars_lens = [[len(x) for x in tokens] for tokens in xx_chars]
    for l in x_chars_lens:
      for i in range(max(x_tokens_lens)-len(l)):
          l.append(0)

    xx_tokens_pad = pad_sequence(xx_tokens, batch_first=True, padding_value=token_pad)
    xx_pre_pad = pad_sequence(xx_pre, batch_first=True, padding_value=pre_pad)
    xx_suf_pad = pad_sequence(xx_suf, batch_first=True, padding_value=suf_pad)

    assert xx_pre_pad.size() == xx_suf_pad.size() == xx_tokens_pad.size()

    max_token_len = max(max(lens) for lens in x_chars_lens)
    xx_chars_inside_pad = [pad_collate_chars(xx, max_token_len, char_pad) for xx in xx_chars]
    xx_chars_pad = pad_sequence(xx_chars_inside_pad, batch_first=True, padding_value=char_pad)

    return xx_tokens_pad, xx_pre_pad, xx_suf_pad, xx_chars_pad, x_tokens_lens, x_chars_lens


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='LSTM Tagger')
    parser.add_argument('repr', metavar='repr', type=str, help='one of a,b,c,d')
    parser.add_argument('modelFile', type=str, help='file to save the model')
    parser.add_argument('inputFile', type=str, help='the blind input file to tag')
    # parser.add_argument('--task', dest='task', type=str, help='task to run - this way you can save the predictions')

    args = parser.parse_args()

    method = args.repr
    model_path = args.modelFile
    input_file = args.inputFile
    task = "ner" if "ner" in input_file else "pos"

    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running device: {device}')
    checkpoint = torch.load(model_path, map_location=device)

    pretrained_vecs = None
    if method == 'a':
        token_level, pre_suf_level, char_level = True, False, False
    elif method == 'b':
        token_level, pre_suf_level, char_level = False, False, True
    elif method == 'c':
        token_level, pre_suf_level, char_level = True, True, False
    elif method == 'd':
        token_level, pre_suf_level, char_level = True, False, True

    model_state_dict = checkpoint['model_state_dict']
    tokens2ids = checkpoint['tokens2ids']
    char2ids = checkpoint['char2ids']
    pre2ids = checkpoint['pre2ids']
    suf2ids = checkpoint['suf2ids']
    tags2ids = checkpoint['tag2ids']

    token_pad = tokens2ids[PAD]
    pre_pad = pre2ids[PAD]
    suf_pad = suf2ids[PAD]
    char_pad = char2ids[PAD]

    test_dataset = TagDataset(input_file, return_y=False,
                                tokens2ids=tokens2ids,
                                char2ids=char2ids,
                                pre2ids=pre2ids,
                                suf2ids=suf2ids,
                              tags2ids=tags2ids
                              )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, collate_fn=lambda b: pad_collate(b, token_pad, pre_pad, suf_pad, char_pad))

    model = BiLSTMTagger(vocab_size=len(tokens2ids.keys()),
                         pre_vocab_size=len(pre2ids.keys()),
                         suf_vocab_size=len(suf2ids.keys()),
                         alphabet_size=len(char2ids.keys()),
                         tagset_size=len(tags2ids.keys()),
                         token_padding_idx=token_pad,
                         pre_padding_idx=pre_pad,
                         suf_padding_idx=suf_pad,
                         char_padding_idx=char_pad,
                         token_level=token_level,
                         pre_suf_level=pre_suf_level,
                         char_level=char_level,
                         pretrained_vecs=pretrained_vecs)

    model.load_state_dict(state_dict=model_state_dict)
    model.to(device)

    preds = predict(model, test_loader, device, tags2ids)

    with open(f"test4.{task}", "w") as f:
        for sentence, tags in zip(test_dataset.sentences, preds):
            for word, tag in zip(sentence, tags):
                f.write(f"{word} {tag}" + "\n")
            f.write("\n")