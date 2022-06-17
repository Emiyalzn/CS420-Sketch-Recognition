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
import ast

from utils.logger import Logger
from utils.utils import args_print, fix_seed

from utils.evaluation import compute_running_corrects


class BaseRunner(object):
    def __init__(self, local_dir, args=None):
        if (args is not None):
            self.config = self._parse_args(args)
        else:
            self.config = self._parse_args()
        # self.step_counters = {m: 0 for m in self.modes}
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
        arg_parser.add_argument('--seed', nargs='?', default='[42,43,44]', help='random seed')
        arg_parser.add_argument('--early_stopping', type=int, default=3)
        arg_parser.add_argument('--num_epoch', type=int, default=20)
        arg_parser.add_argument('--valid_freq', type=int, default=1)
        arg_parser.add_argument('--categories', type=ast.literal_eval, default="['bear', 'cat', 'crocodile', 'elephant', 'giraffe', 'horse', 'lion', 'owl', 'penguin', 'raccoon', 'sheep', 'tiger', 'zebra', 'camel', 'cow', 'dog', 'flamingo', 'hedgehog', 'kangaroo', 'monkey', 'panda', 'pig', 'rhinoceros', 'squirrel', 'whale']")

        # for dataset
        arg_parser.add_argument('--data_seq_dir', type=str, default=None)
        arg_parser.add_argument('--data_img_dir', type=str, default=None)

        # Added for compatibility
        arg_parser.add_argument('--ckpt_nets', nargs='*')
        arg_parser.add_argument('--ckpt_prefix', type=str, default=None)

        arg_parser = self.add_args(arg_parser)
        
        config = vars(arg_parser.parse_args(args))

        if config['seed'] is None:
            config['seed'] = [random.randint(0, 2 ** 31 - 1)]
        else:
            config['seed'] = eval(config['seed'])

        return config

    def add_args(self, arg_parser):
        return arg_parser

    def checkpoint_prefix(self):
        return self.config['ckpt_prefix']

    def weight_decay_excludes(self):
        return ['bias']

    def prepare_dataset(self):
        raise NotImplementedError

    def create_model(self, num_categories):
        raise NotImplementedError

    def create_data_loaders(self, dataset_dict):
        raise NotImplementedError

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        raise NotImplementedError

    def train(self):
        # During train(), 'train', 'valid' datasets are required
        self.modes = ['train', 'valid']
        
        weight_decay = self.config['weight_decay']
        lr = self.config['lr']
        lr_step = self.config['lr_step']
        num_epochs = self.config['num_epoch']
        valid_freq = self.config['valid_freq']
        early_stopping = self.config['early_stopping']

        train_data = self.prepare_dataset()
        num_categories = len(self.config['categories'])
        self.logger.info(f"Number of categories: {num_categories}")

        data_loaders = self.create_data_loaders(train_data)
        criterion = nn.CrossEntropyLoss()

        best_acc_record = {'valid': [], 'test': []}
        for seed in self.config['seed']:
            # initialize network
            net = self.create_model(num_categories)
            optimizer = optim.Adam(net.params_to_optimize(weight_decay, self.weight_decay_excludes()), lr)
            if lr_step > 0:
                lr_exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5)
            else:
                lr_exp_scheduler = None

            self.logger.info(f"Fix seed {seed}.")
            fix_seed(seed)
            best_val_acc = 0.0
            best_epoch = -1
            epoch_count = 0 # use for early stopping

            for epoch in range(1, num_epochs + 1):
                self.logger.info('-' * 20)
                self.logger.info(f"Epoch {epoch}/{num_epochs}")

                for mode in self.modes:
                    is_train = mode == 'train'
                    if not is_train and epoch % valid_freq != 0:
                        continue
                    self.logger.info(f"Start {mode} mode.")

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
                        # self.step_counters[mode] += 1

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

                    if mode == 'valid':
                        if epoch_acc > best_val_acc:
                            self.logger.info("New best valid acc, save model to disk.")
                            best_val_acc = epoch_acc
                            best_epoch = epoch
                            net.save(self.model_dir, f'best_{seed}')
                            epoch_count = 0
                        else:
                            epoch_count += 1

                    # if mode == 'test' and best_epoch == epoch:
                    #     best_test_acc = epoch_acc

                # early stopping
                if epoch_count >= early_stopping:
                    self.logger.info("Early stopping...")
                    break

            self.logger.info(f"Best valid acc: {best_val_acc:.4f}, "
                             f"corresponding epoch: {best_epoch}.")
            best_acc_record['valid'].append(best_val_acc)
            # best_acc_record['test'].append(best_test_acc)

        self.logger.info(
            f"Average valid acc: {np.mean(best_acc_record['valid']):.4f}±{np.std(best_acc_record['valid']):.4f}")
            # f"Average test acc: {np.mean(best_acc_record['test']):.4f}±{np.std(best_acc_record['test']):.4f}")

        for m in self.modes:
            train_data[m].dispose()

    class DummyOptimizer(object):
        def zero_grad(self):
            pass
        def step(self):
            pass

    def evaluate(self):
        # During evaluate(), 'test' datasets are required
        self.modes = ['test']

        test_data = self.prepare_dataset()
        num_categories = len(self.config['categories'])
        self.logger.info(f"Number of categories: {num_categories}")
        
        data_loaders = self.create_data_loaders(test_data)

        for seed in self.config['seed']:
            # initialize network
            net = self.create_model(num_categories)
            path_prefix = os.path.join(self.model_dir, f'_iter_best_{seed}')
            net.load(path_prefix)

            self.logger.info(f"Read model of seed {seed}.")
            fix_seed(seed)

            self.logger.info('-' * 20)

            # self.modes = ['test']
            for mode in self.modes:
                self.logger.info(f"Start {mode} mode.")
                net.eval_mode()

                confusion_matrix = torch.zeros((num_categories, num_categories), dtype=torch.int)
                running_corrects_top1 = 0
                running_corrects_top5 = 0
                num_samples = 0
                pbar = tqdm.tqdm(total=len(data_loaders[mode]))
                for bid, data_batch in enumerate(data_loaders[mode]):
                    # self.step_counters[mode] += 1

                    logits, _, gt_category = self.forward_batch(net, data_batch, mode, BaseRunner.DummyOptimizer(), lambda _1, _2: 0)
                    _, predicts = torch.max(logits, 1)
                    confusion_matrix[gt_category, predicts] += 1
                    # predicts_accu = torch.sum(predicts == gt_category)
                    running_corrects = compute_running_corrects(logits, gt_category, (1, 5))
                    running_corrects_top1 += running_corrects[1]
                    running_corrects_top5 += running_corrects[5]

                    sampled_batch_size = gt_category.size(0)
                    num_samples += sampled_batch_size

                    pbar.update()
                pbar.close()

                epoch_acc_1 = float(running_corrects_top1) / float(num_samples)
                epoch_acc_5 = float(running_corrects_top5) / float(num_samples)
                self.logger.info(f"{mode}:\tTop1-Accuracy: {epoch_acc_1:.4f} | Top5-Accuracy: {epoch_acc_5:.4f}")
                
                for i, k in enumerate(self.config['categories']):
                    TP = confusion_matrix[i, i]
                    TN = torch.sum(confusion_matrix[:i, :i]) + torch.sum(confusion_matrix[:i, i+1:]) \
                    + torch.sum(confusion_matrix[i+1:, :i]) + torch.sum(confusion_matrix[i+1:, i+1:])
            
                    T = torch.sum(confusion_matrix[i])    # = TP + FN
                    P = torch.sum(confusion_matrix[:, i]) # = TP + FP
                    
                    FN = T - TP
                    FP = P - TP
                    
                    accuracy = (TP + TN) / (TP + FN + FP + TN)
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    f1 = 2 * precision * recall / (precision + recall)
                    
                    self.logger.info(f'{mode}-{k}: Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}')
                np.save(os.path.join(self.local_dir, f"confusion_matrix_{seed}.npy"), confusion_matrix.cpu().numpy())

        for m in self.modes:
            test_data[m].dispose()
