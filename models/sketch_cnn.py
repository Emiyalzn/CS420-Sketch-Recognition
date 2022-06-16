from .basemodel import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import os.path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchCNN(BaseModel):
    def __init__(self,
                 cnn_fn,
                 num_categories,
                 train_cnn=True,
                 device=None):
        super().__init__()

        self.eps = 1e-4
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.cnn = cnn_fn(pretrained=True, requires_grad=train_cnn, in_channels=1)

        num_fc_in_features = self.cnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)

        nets.extend([self.cnn, self.fc])
        names.extend(['conv', 'fc'])
        train_flags.extend([train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, images):
        cnnfeat = self.cnn(images)
        logits = self.fc(cnnfeat)

        return logits