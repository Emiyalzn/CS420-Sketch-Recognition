import os.path
import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataset.dataset import QuickDrawDataset, r2cnn_collate
from models.cnnmodels import CNN_MODELS, CNN_IMAGE_SIZES
from models.sketch_r2cnn import SketchR2CNN
from neuralline.rasterize import Raster
import tqdm
from datetime import datetime
import numpy as np

from .base_runner import BaseRunner
from utils.utils import fix_seed


class SketchR2CNNRunner(BaseRunner):
    def __init__(self, args=None, local_dir=None):
        if local_dir is None:
            local_dir = os.path.join("results", f'sketch-r2cnn-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        super(SketchR2CNNRunner, self).__init__(local_dir, args)

    def add_args(self, arg_parser):
        arg_parser.add_argument('--dropout', type=float, default=0.5)
        arg_parser.add_argument('--intensity_channels', type=int, default=8)
        arg_parser.add_argument('--thickness', type=float, default=1.0)
        # If `intensity_channels` in {1, 3} then can convert it to 3-channel
        # and the model can remain unchanged.
        # Otherwise the first conv layer should be reconstructed.
        arg_parser.add_argument('--model_fn', type=str, default='efficientnet_b0')
        
        arg_parser.add_argument('--paddingLength', type=int, default=226)
        arg_parser.add_argument('--random_scale_factor', type=float, default=0.0)
        arg_parser.add_argument('--stroke_removal_prob', type=float, default=0.0)
        arg_parser.add_argument('--img_scale_ratio', type=float, default=1.0)
        arg_parser.add_argument('--img_rotate_angle', type=float, default=0.0)
        arg_parser.add_argument('--img_translate_dist', type=float, default=0.0)
        
        arg_parser.add_argument('--do_augmentation', action='store_true')
        
        return arg_parser

    def prepare_dataset(self):
        if ('robustness_experiment' in self.config.keys() and self.config['robustness_experiment']):
            train_data = {
                m : QuickDrawDataset(
                    mode=m,
                    data_seq_dir=self.config['data_seq_dir'],
                    stroke_removal_prob=self.config['stroke_removal_prob'],
                    do_augmentation=False,
                    robustness_experiment=True,
                    require_img=False,
                    scale_factor_rexp=self.config['scale_factor'],
                    rot_thresh_rexp=self.config['rot_thresh_rexp']
                ) for m in self.modes
            }
        else:
            train_data = {
                m: QuickDrawDataset(
                    mode=m,
                    data_seq_dir=self.config['data_seq_dir'],
                    do_augmentation=(self.config['do_augmentation'] if 'do_augmentation' in self.config.keys() else not self.config['disable_augmentation']),
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

        points = data_batch['points3'].to(self.device).contiguous()
        points_offset = data_batch['points3_offset'].to(self.device).contiguous()
        points_length = data_batch['points3_length'].contiguous()
        category = data_batch['category'].to(self.device).contiguous()

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, attention, images = model(points, points_offset, points_length)
            loss = criterion(logits, category)
            if is_train:
                loss.backward()
                optimizer.step()

        return logits, loss, category

    def embed_batch(self, model, data_batch):
        points = data_batch['points3'].to(self.device).contiguous()
        points_offset = data_batch['points3_offset'].to(self.device).contiguous()
        points_length = data_batch['points3_length'].contiguous()
        feats = model.embed(points, points_offset, points_length)
        category = data_batch['category'].to(self.device).contiguous()
        
        return feats, category