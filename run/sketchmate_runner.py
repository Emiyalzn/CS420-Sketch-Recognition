import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataset.dataset import SketchDataset
from models.cnnmodels import CNN_MODELS, CNN_IMAGE_SIZES
from models.rnnmodels import RNN_MODELS
from models.sketch_mate import SketchMate
import tqdm
from datetime import datetime
from torchvision import transforms
import numpy as np

from .base_runner import BaseRunner
from utils.utils import fix_seed

class SketchMateRunner(BaseRunner):
    def __init__(self, args=None, local_dir=None):
        if local_dir is None:
            local_dir = os.path.join("results", f'sketchmate-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        super(SketchMateRunner, self).__init__(local_dir, args)

        self.transform = transforms.Resize(CNN_IMAGE_SIZES[self.config['cnn_fn']])
        if self.config['cnn_fn'] in ['densenet161', 'resnet50', 'efficientnet_b0']:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def add_args(self, arg_parser):
        arg_parser.add_argument('--cnn_fn', type=str, default='resnet50')
        arg_parser.add_argument('--rnn_fn', type=str, default='lstm')

        arg_parser.add_argument('--paddingLength', type=int, default=226)
        arg_parser.add_argument('--random_scale_factor', type=float, default=0.0)
        arg_parser.add_argument('--stroke_removal_prob', type=float, default=0.0)
        arg_parser.add_argument('--img_scale_ratio', type=float, default=1.0)
        arg_parser.add_argument('--img_rotate_angle', type=float, default=0.0)
        arg_parser.add_argument('--img_translate_dist', type=float, default=0.0)

        arg_parser.add_argument('--disable_augmentation', action='store_false')

        return arg_parser

    def prepare_dataset(self):
        train_data = {
            m: SketchDataset(
                m,
                self.config['data_seq_dir'],
                self.config['data_img_dir'],
                self.config['categories']
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
                          pin_memory=True) for m in self.modes
        }
        return data_loaders

    def create_model(self, num_categories):
        cnn_fn = CNN_MODELS[self.config['cnn_fn']]
        rnn_fn = RNN_MODELS[self.config['rnn_fn']]
        imgsize = CNN_IMAGE_SIZES[self.config['cnn_fn']]

        return SketchMate(cnn_fn,
                          rnn_fn,
                          imgsize,
                          num_categories,
                          device=self.device)

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        is_train = mode == 'train'

        points_offset = data_batch[0].to(self.device).contiguous()
        points_length = data_batch[2].contiguous()
        images = self.transform(data_batch[3].repeat([1,3,1,1]).contiguous()).to(self.device)
        categories = data_batch[4].to(self.device).contiguous()

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(points_offset, points_length, images)
            loss = criterion(logits, categories)
            if is_train:
                loss.backward()
                optimizer.step()

        return logits, loss, categories

    def embed_batch(self, model, data_batch):
        points_offset = data_batch[0].to(self.device).contiguous()
        points_length = data_batch[2].contiguous()
        images = self.transform(data_batch[3].repeat([1,3,1,1]).contiguous()).to(self.device)
        categories = data_batch[4].to(self.device).contiguous()
        feats = model.embed(points_offset, points_length, images)
        
        return feats, categories