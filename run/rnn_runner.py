import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from datetime import datetime
import numpy as np
import ast

from dataset.dataset import SketchDataset, QuickDrawDataset, r2cnn_collate
from models.rnnmodels import RNN_MODELS
from models.sketch_rnn import SketchRNN
from utils.utils import fix_seed

from .base_runner import BaseRunner


class SketchRNNRunner(BaseRunner):
    def __init__(self, args=None, local_dir=None):
        if (local_dir is None):
            local_dir = os.path.join(f"results", f'rnn-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        super(SketchRNNRunner, self).__init__(local_dir, args)

    def add_args(self, arg_parser):
        arg_parser.add_argument('--paddingLength', type=int, default=226)
        arg_parser.add_argument('--model_fn', type=str, default='lstm')

        # augmentation
        arg_parser.add_argument('--random_scale_factor', type=float, default=0.0)
        arg_parser.add_argument('--stroke_removal_prob', type=float, default=0.0)
        arg_parser.add_argument('--img_scale_ratio', type=float, default=1.0)
        arg_parser.add_argument('--img_rotate_angle', type=float, default=0.0)
        arg_parser.add_argument('--img_translate_dist', type=float, default=0.0)

        arg_parser.add_argument('--disable_augmentation', action='store_false')
        return arg_parser

    def prepare_dataset(self):
        train_data = {
            m: QuickDrawDataset(
                m,
                self.config['data_seq_dir']
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
        model_fn = self.config['model_fn']

        return SketchRNN(RNN_MODELS[model_fn],
                         num_categories,
                         device=self.device)

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        is_train = mode == 'train'
        points_offset = data_batch['points5_offset'].to(self.device).contiguous()
        points_length = data_batch['points3_length'].contiguous()
        categories = data_batch['category'].to(self.device).contiguous()
        # points_offset = data_batch[1].to(self.device).contiguous()
        # points_length = data_batch[2].contiguous()
        # categories = data_batch[4].to(self.device).contiguous()

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(points_offset, points_length)
            loss = criterion(logits, categories)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_value_(model.params(True), 1.0) # gradient clipping
                optimizer.step()

        return logits, loss, categories

    def embed_batch(self, model, data_batch):
        points_offset = data_batch['points5_offset'].to(self.device).contiguous()
        points_length = data_batch['points3_length'].contiguous()
        categories = data_batch['category'].to(self.device).contiguous()
        feats = model.embed(points_offset, points_length)
        
        return feats, categories