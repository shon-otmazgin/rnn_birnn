from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTMTaggerA(nn.Module):
    def __init__(self, vocab_size, tagset_size, padding_idx):
        super(BiLSTMTaggerA, self).__init__()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.emb_size = 300
        self.s_dim = 768

        self.word_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)
        self.bilstm1 = nn.LSTM(bidirectional=True, input_size=self.emb_size,
                               hidden_size=self.s_dim, num_layers=1, batch_first=True)
        self.bilstm2 = nn.LSTM(bidirectional=True, input_size=2*self.s_dim,
                               hidden_size=self.s_dim, num_layers=1, batch_first=True)

        self.linear = nn.Linear(2*self.s_dim, tagset_size)

    def forward(self, input_ids, input_lens):
        embds = self.word_emb(input_ids)        # [batch, sequence_len, emb]

        x_packed = pack_padded_sequence(embds, input_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm1(x_packed)          # [batch, sequence_len, out_dim]
        lstm_out, _ = self.bilstm2(lstm_out)
        output_padded, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        output = self.linear(output_padded)
        return output


class BiLSTMTaggerB(nn.Module):
    def __init__(self, vocab_size, tagset_size, padding_idx):
        super(BiLSTMTaggerB, self).__init__()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.emb_size = 300
        self.s_dim = 768

        self.word_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)
        self.bilstm1 = nn.LSTM(bidirectional=True, input_size=self.emb_size,
                               hidden_size=self.s_dim, num_layers=1, batch_first=True)
        self.bilstm2 = nn.LSTM(bidirectional=True, input_size=2*self.s_dim,
                               hidden_size=self.s_dim, num_layers=1, batch_first=True)

        self.linear = nn.Linear(2*self.s_dim, tagset_size)

    def forward(self, input_ids, input_lens):
        embds = self.word_emb(input_ids)        # [batch, sequence_len, emb]

        x_packed = pack_padded_sequence(embds, input_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm1(x_packed)          # [batch, sequence_len, out_dim]
        lstm_out, _ = self.bilstm2(lstm_out)
        output_padded, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        output = self.linear(output_padded)
        return output