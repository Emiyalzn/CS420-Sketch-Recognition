from .basemodel import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchRNN(BaseModel):
    def __init__(self,
                 rnn_fn,
                 num_categories,
                 input_size=5,
                 train_rnn=True,
                 device=None):
        super().__init__()

        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.rnn = rnn_fn(input_size=input_size)

        num_fc_in_features = self.rnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)

        nets.extend([self.rnn, self.fc])
        names.extend(['conv', 'fc'])    # 'conv' is not a bug. For compatibility, we continue to use it. 
                                        # If you want to use our pretrained model, please don't change this.
        train_flags.extend([train_rnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, points, lengths):
        # images does not used in this model
        rnnfeat = self.rnn(points, lengths)
        logits = self.fc(rnnfeat)

        return logits

    def embed(self, points, lengths):
        # images does not used in this model
        rnnfeat = self.rnn(points, lengths)

        return rnnfeat
