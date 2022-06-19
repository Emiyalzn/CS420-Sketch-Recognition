import argparse
import json
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import tqdm
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from datetime import datetime
import ast

from dataset.dataset import QuickDrawDataset, SketchDataset
from models.cnnmodels import CNN_MODELS, CNN_IMAGE_SIZES
from models.sketch_cnn import SketchCNN
from utils.utils import fix_seed

from .base_runner import BaseRunner

class SketchCNNRunner(BaseRunner):
    def __init__(self, args=None, local_dir=None):
        if (local_dir is None):
            local_dir = os.path.join("results", f'cnn-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        super(SketchCNNRunner, self).__init__(local_dir, args)
        
        self.transform = transforms.Resize(CNN_IMAGE_SIZES[self.config['model_fn']])
        if self.config['model_fn'] in ['densenet161', 'resnet50', 'efficientnet_b0']:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(mean=[0.485, 0.456, 0.406],                         #* All torchvision pre-trained models expect input images normalized in the same way
                                     std=[0.229, 0.224, 0.225])
            ])
        # https://pytorch.org/vision/stable/models.html#:~:text=eval()%20for%20details.-,All%20pre%2Dtrained%20models%20expect%20input%20images%20normalized%20in%20the%20same%20way,-%2C%20i.e.%20mini

    def add_args(self, arg_parser):
        arg_parser.add_argument('--model_fn', type=str, default='resnet50')

        # augmentation
        arg_parser.add_argument('--paddingLength', type=int, default=226)
        arg_parser.add_argument('--random_scale_factor', type=float, default=0.0)
        arg_parser.add_argument('--stroke_removal_prob', type=float, default=0.0)
        arg_parser.add_argument('--img_scale_ratio', type=float, default=1.0)
        arg_parser.add_argument('--img_rotate_angle', type=float, default=0.0)
        arg_parser.add_argument('--img_translate_dist', type=float, default=0.0)
        
        arg_parser.add_argument('--disable_augmentation', action='store_false')
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
                    require_img=True,
                    scale_factor_rexp=self.config['scale_factor'],
                    rot_thresh_rexp=self.config['rot_thresh_rexp']
                ) for m in self.modes
            }
        else:
            train_data = {
                m: SketchDataset(
                    mode=m,
                    data_seq_dir=self.config['data_seq_dir'],
                    data_img_dir=self.config['data_img_dir'],
                    categories=self.config['categories'],
                    paddingLength=self.config['paddingLength'],
                    random_scale_factor=self.config['random_scale_factor'],
                    stroke_removal_prob=self.config['stroke_removal_prob'],
                    img_scale_ratio=self.config['img_scale_ratio'],
                    img_rotate_angle=self.config['img_rotate_angle'],
                    img_translate_dist=self.config['img_translate_dist'],
                    disable_augmentation=self.config['disable_augmentation']
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
        model_fn = self.config['model_fn']

        return SketchCNN(CNN_MODELS[model_fn],
                         num_categories,
                         device=self.device)

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        is_train = mode == 'train'

        images = self.transform(data_batch[3].repeat([1, 3, 1, 1]).contiguous()).to(self.device)
        categories = data_batch[4].to(self.device)

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(images=images)
            loss = criterion(logits, categories)
            if is_train:
                loss.backward()
                optimizer.step()

        return logits, loss, categories


    def embed_batch(self, model, data_batch):
        images = self.transform(data_batch[3].repeat([1, 3, 1, 1]).contiguous()).to(self.device)
        categories = data_batch[4].to(self.device)
        feats = model.embed(images=images)
        
        return feats, categories