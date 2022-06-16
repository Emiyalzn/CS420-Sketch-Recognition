import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class BiGRU(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=256,
                 num_layers=2,
                 batch_first=True,
                 bidirect=True,
                 dropout=0.1,
                 requires_grad=True):
        super(BiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirect = bidirect

        self.rnn = nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           bidirectional=bidirect,
                           dropout=dropout)

        num_directs = 2 if bidirect else 1
        self.num_out_features = num_layers * num_directs * hidden_size

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points, lengths):
        batch_size = points.shape[0]
        point_dim = points.shape[2]

        if point_dim != self.input_size:
            points = points[:, :, :self.input_size]

        # use PackedSequence to speed up training
        points_packed = pack_padded_sequence(points, lengths, batch_first=self.batch_first, enforce_sorted=False)
        _, last_hidden = self.rnn(points_packed)

        last_hidden = last_hidden.transpose(0, 1).reshape(batch_size, -1)
        return last_hidden

class BiLSTM(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=256,
                 num_layers=2,
                 batch_first=True,
                 bidirect=True,
                 dropout=0.1,
                 requires_grad=True):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirect = bidirect

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           bidirectional=bidirect,
                           dropout=dropout)

        num_directs = 2 if bidirect else 1
        self.num_out_features = num_layers * num_directs * hidden_size

        # Initialization
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points, lengths):
        batch_size = points.shape[0]
        point_dim = points.shape[2]

        if point_dim != self.input_size:
            points = points[:, :, :self.input_size]

        # use PackedSequence to speed up training
        points_packed = pack_padded_sequence(points, lengths, batch_first=self.batch_first, enforce_sorted=False)
        _, (last_hidden, _) = self.rnn(points_packed)

        last_hidden = last_hidden.transpose(0, 1).reshape(batch_size, -1)
        return last_hidden

RNN_MODELS = {
    'lstm': BiLSTM,
    'gru': BiGRU,
}