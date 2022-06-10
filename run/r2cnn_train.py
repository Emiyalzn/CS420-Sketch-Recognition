import os.path
import sys
import warnings
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from dataset.dataset import QuickDrawDataset, train_data_collate
from models.cnnmodels import CNN_MODELS, CNN_IMAGE_SIZES
from models.sketch_r2cnn import SketchR2CNN
from neuralline.rasterize import Raster

from .base_train import BaseTrain


class SketchR2CNNTrain(BaseTrain):
    def __init__(self):
        local_dir = os.path.join("../results", f'sketch-r2cnn-{datetime.now().strftime("%Y%m%d-%H%M")}')
        super(SketchR2CNNTrain).__init__(local_dir)

    def add_args(self, arg_parser):
        arg_parser.add_argument('--dropout', type=float, default=0.5)
        arg_parser.add_argument('--intensity_channels', type=int, default=1)
        # If `intensity_channels` in {1, 3} then can convert it to 3-channel
        # and the model can remain unchanged.
        # Otherwise the first conv layer should be reconstructed.
        arg_parser.add_argument('--model_fn', type=str, default='resnet50')
        return arg_parser

    def create_data_loaders(self, dataset_dict):
        data_loaders = {
            m: DataLoader(dataset_dict[m],
                          batch_size=self.config['batch_size'],
                          num_workers=3 if m == 'train' else 1,
                          shuffle=True if m == 'train' else False,
                          drop_last=True,
                          collate_fn=train_data_collate,
                          pin_memory=True) for m in self.modes
        }
        return data_loaders

    def create_model(self, num_categories):
        dropout = self.config['dropout']
        intensity_channels = self.config['intensity_channels']
        model_fn = self.config['model_fn']
        imgsize = CNN_IMAGE_SIZES[model_fn]
        thickness = self.config['thickness']

        return SketchR2CNN(CNN_MODELS[model_fn],
                           3,
                           dropout,
                           imgsize,
                           thickness,
                           num_categories,
                           intensity_channels=intensity_channels,
                           device=self.device)

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        is_train = mode == 'train'

        points = data_batch['points3'].to(self.device)
        points_offset = data_batch['points3_offset'].to(self.device)
        points_length = data_batch['points_length']
        category = data_batch['category'].to(self.device)

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, attention, images = model(points, points_offset, points_length)
            loss = criterion(logits, category)
            if is_train:
                loss.backward()
                optimizer.step()

        return logits, loss, category
    

if __name__ == '__main__':
    with SketchR2CNNTrain() as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()