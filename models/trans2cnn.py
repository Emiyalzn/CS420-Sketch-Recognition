from .basemodel import BaseModel
from neuralline.rasterize import RasterIntensityFunc

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class Trans2CNN(BaseModel):
    def __init__(self,
                 cnn_fn,
                 img_size,
                 thickness,
                 num_categories,
                 intensity_channels=8,
                 train_cnn=True,
                 device=None):
        super().__init__()

        self.img_size = img_size
        self.thickness = thickness
        self.intensity_channels = intensity_channels
        self.eps = 1e-4
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.encoder = TransEncoder()
        self.cnn = cnn_fn(pretrained=False, requires_grad=train_cnn, in_channels=intensity_channels)

        num_fc_in_features = self.cnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)

        nets.extend([self.encoder, self.cnn, self.fc])
        names.extend(['transencoder', 'conv', 'fc'])
        train_flags.extend([True, train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, points):
        intensities = self.encoder()

        images = RasterIntensityFunc.apply(points, intensities, self.img_size, self.thickness, self.eps, self.device)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        cnnfeat = self.cnn(images)
        logits = self.fc(cnnfeat)

        return logits, intensities, images
