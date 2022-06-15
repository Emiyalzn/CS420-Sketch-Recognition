import argparse
import json
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset.dataset import R2CNNDataset, r2cnn_collate
from utils.logger import Logger
from utils.utils import args_print, fix_seed


class BaseTrain(object):
    def __init__(self, local_dir, args=None):
        if (args is not None):
            self.config = self._parse_args(args)
        else:
            self.config = self._parse_args()
        self.modes = ['train', 'valid', 'test']
        self.step_counters = {m: 0 for m in self.modes}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # init folder and logger
        self.model_dir = os.path.join(local_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger = Logger.init_logger(filename=local_dir + '/_output_.log')
        args_print(self.config, self.logger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _parse_args(self, args=None):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--batch_size', type=int, default=48)
        arg_parser.add_argument('--lr_step', type=int, default=-1)
        arg_parser.add_argument('--lr', type=float, default=0.0001)
        arg_parser.add_argument('--weight_decay', type=float, default=-1)
        arg_parser.add_argument('--seed', type=int, default=42)
        arg_parser.add_argument('--num_epoch', type=int, default=20)
        arg_parser.add_argument('--valid_freq', type=int, default=1)
        
        # Added for compatibility
        arg_parser.add_argument('--ckpt_nets', nargs='*')
        arg_parser.add_argument('--ckpt_prefix', type=str, default=None)

        arg_parser = self.add_args(arg_parser)
        
        config = vars(arg_parser.parse_args(args))

        if config['seed'] is None:
            config['seed'] = random.randint(0, 2 ** 31 - 1)
        fix_seed(config['seed'])

        return config

    def add_args(self, arg_parser):
        return arg_parser

    def checkpoint_prefix(self):
        return self.config['ckpt_prefix']

    def weight_decay_excludes(self):
        return ['bias']

    def prepare_dataset(self, dataset_dict):
        pass

    def create_model(self, num_categories):
        raise NotImplementedError

    def create_data_loaders(self, dataset_dict):
        raise NotImplementedError

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
