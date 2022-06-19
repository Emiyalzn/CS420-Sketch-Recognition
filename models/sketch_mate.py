from .basemodel import BaseModel
import torch
import torch.nn as nn

class SketchMate(BaseModel):
    def __init__(self,
                 cnn_fn,
                 rnn_fn,
                 img_size,
                 num_categories,
                 rnn_input_size=3,
                 train_cnn=True,
                 device=None):
        super(SketchMate, self).__init__()

        self.img_size = img_size
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.rnn = rnn_fn(rnn_input_size)
        self.cnn = cnn_fn(pretrained=False, requires_grad=train_cnn, in_channels=1)

        num_fc_in_features = self.cnn.num_out_features + self.rnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)

        nets.extend([self.rnn, self.cnn, self.fc])
        names.extend(['rnn', 'cnn', 'fc'])
        train_flags.extend([True, train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, points_offset, lengths, imgs):
        cnnfeat = self.cnn(imgs)
        rnnfeat = self.rnn(points_offset, lengths)
        fused_feat = torch.cat([cnnfeat, rnnfeat], dim=1)
        logits = self.fc(fused_feat)

        return logits

    
    def embed(self, points_offset, lengths, imgs):
        cnnfeat = self.cnn(imgs)
        rnnfeat = self.rnn(points_offset, lengths)
        fused_feat = torch.cat([cnnfeat, rnnfeat], dim=1)
        # fused_feat = self.fc(fused_feat)

        return fused_feat