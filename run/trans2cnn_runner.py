import os.path
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from dataset.dataset import QuickDrawDataset, r2cnn_collate
from models.cnnmodels import CNN_MODELS, CNN_IMAGE_SIZES
from models.trans2cnn import Trans2CNN
from .base_runner import BaseRunner

class Trans2CNNRunner(BaseRunner):
    def __init__(self, args=None, local_dir=None):
        if local_dir is None:
            local_dir = os.path.join("results", f'trans2cnn-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        super(Trans2CNNRunner, self).__init__(local_dir, args)

    def add_args(self, arg_parser):
        arg_parser.add_argument('--dropout', type=float, default=0.5)
        arg_parser.add_argument('--intensity_channels', type=int, default=8)
        arg_parser.add_argument('--thickness', type=float, default=1.0)
        arg_parser.add_argument('--model_fn', type=str, default='efficientnet_b0')

        # for augmentation
        arg_parser.add_argument('--stroke_removal_prob', type=float, default=0.0)
        arg_parser.add_argument('--do_augmentation', action='store_true')

        # for reconstruction
        arg_parser.add_argument('--do_reconstruction', action='store_true')

        return arg_parser

    def prepare_dataset(self):
        train_data = {
            m: QuickDrawDataset(
                mode=m,
                data_seq_dir=self.config['data_seq_dir'],
                stroke_removal_prob=self.config['stroke_removal_prob'],
                do_augmentation=self.config['do_augmentation'],
            ) for m in self.modes
        }
        return train_data

    def create_data_loaders(self, dataset_dict):
        data_loaders = {
            m: DataLoader(dataset_dict[m],
                          batch_size=self.config['batch_size'],
                          num_workers=3 if m == 'train' else 1,
                          shuffle=True if m == 'train' else False,
                          drop_last=True,
                          collate_fn=r2cnn_collate,
                          pin_memory=True) for m in self.modes
        }
        return data_loaders

    def create_model(self, num_categories):
        dropout = self.config['dropout']
        intensity_channels = self.config['intensity_channels']
        model_fn = self.config['model_fn']
        img_size = CNN_IMAGE_SIZES[model_fn]
        thickness = self.config['thickness']

        return Trans2CNN(model_fn,
                         img_size,
                         thickness,
                         num_categories,
                         intensity_channels,
                         device=self.device)

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        is_train = mode == 'train'

        points = data_batch['points3'].to(self.device).contiguous()
        category = data_batch['category'].to(self.device).contiguous()

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, attention, images = model(points)
            loss = criterion(logits, category)
            if is_train:
                loss.backward()
                optimizer.step()

        return logits, loss, category

