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
from datetime import datetime

from dataset.dataset import QuickDrawDataset, train_data_collate
from models.cnnmodels import CNN_MODELS, CNN_IMAGE_SIZES
from models.sketch_r2cnn import SketchR2CNN
from neuralline.rasterize import Raster
from utils.logger import Logger
from utils.utils import args_print, fix_seed

class BaseTrain(object):
    def __init__(self, local_dir):
        self.config = self._parse_args()
        self.modes = ['train', 'valid']
        self.step_counters = {m: 0 for m in self.modes}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # init folder and logger
        self.model_dir = os.path.join(local_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger = Logger.init_logger(filename=local_dir + '/_output_.log')
        args_print(self.config, self.logger)

    def _parse_args(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--batch_size', type=int, default=48)
        arg_parser.add_argument('--lr_step', type=int, default=-1)
        arg_parser.add_argument('--lr', type=float, default=0.0001)
        arg_parser.add_argument('--weight_decay', type=float, default=-1)
        arg_parser.add_argument('--seed', type=int, default=42)
        arg_parser.add_argument('--num_epoch', type=int, default=1)
        arg_parser.add_argument('--valid_freq', type=int, default=1)

        arg_parser = self.add_args(arg_parser)
        config = vars(arg_parser.parse_args())

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
        weight_decay = self.config['weight_decay']
        lr = self.config['lr']
        lr_step = self.config['lr_step']
        num_epochs = self.config['num_epochs']
        valid_freq = self.config['valid_freq']

        train_data = {
            m: QuickDrawDataset(m) for m in self.modes
        }
        self.prepare_dataset(train_data)
        num_categories = train_data[self.modes[0]].num_categories()
        self.logger.info(f"Number of categories: {num_categories}")

        net = self.create_model(num_categories)
        data_loaders = self.create_data_loaders(train_data)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.params_to_optimize(weight_decay, self.weight_decay_excludes()), lr)
        if lr_step > 0:
            lr_exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5)
        else:
            lr_exp_scheduler = None

        best_acc = 0.0
        best_epoch = -1

        ckpt_prefix = self.checkpoint_prefix()
        ckpt_nets = self.config['ckpt_nets']
        if ckpt_prefix is not None:
            loaded_paths = net.load(ckpt_prefix, ckpt_nets)
            self.logger.info(f"load pretrained model from {loaded_paths}")

        for epoch in range(1, num_epochs+1):
            self.logger.info('-' * 20)
            self.logger.info(f"Epoch {epoch}/{num_epochs}")

            for mode in self.modes:
                is_train = mode == 'train'
                if not is_train and epoch % valid_freq != 0:
                    continue
                self.logger.info(f"Starting {mode} mode.")

                if is_train:
                    if lr_exp_scheduler is not None:
                        lr_exp_scheduler.step()
                    net.train_mode()
                else:
                    net.eval_mode()

                running_corrects = 0
                num_samples = 0
                pbar = tqdm.tqdm(total=len(data_loaders[mode]))
                for bid, data_batch in enumerate(data_loaders[mode]):
                    self.step_counters[mode] += 1

                    logits, loss, gt_category = self.forward_batch(net, data_batch, mode, optimizer, criterion)
                    _, predicts = torch.max(logits, 1)
                    predicts_accu = torch.sum(predicts == gt_category)
                    running_corrects += predicts_accu.item()

                    sampled_batch_size = gt_category.size(0)
                    num_samples += sampled_batch_size

                    pbar.update()
                pbar.close()
                epoch_acc = float(running_corrects) / float(num_samples)
                self.logger.info(f"{mode} acc: {epoch_acc:.4f}")

                if not is_train:
                    if epoch_acc > best_acc:
                        self.logger.info("New best valid acc, save model to disk.")
                        best_acc = epoch_acc
                        best_epoch = epoch
                        net.save(self.model_dir, 'best')
        self.logger.info(f"Best valid acc: {best_acc:.4f}, corresponding epoch: {best_epoch}.")

        for m in self.modes:
            train_data[m].dispose()

class SketchR2CNNTrain(BaseTrain):
    def __init__(self):
        local_dir = os.path.join("../local", f'sketch-r2cnn-{datetime.now().strftime("%Y%m%d-%H%M")}')
        super(SketchR2CNNTrain).__init__(local_dir)

    def add_args(self, arg_parser):
        arg_parser.add_argument('--dropout', type=float, default=0.5)
        arg_parser.add_argument('--intensity_channels', type=int, default=1)
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



