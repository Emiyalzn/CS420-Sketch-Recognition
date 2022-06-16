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

from .base_train import BaseTrain


class SketchRNNTrain(BaseTrain):
    def __init__(self, args=None):
        local_dir = os.path.join(f"results", f'rnn-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        super(SketchRNNTrain, self).__init__(local_dir, args)

    def add_args(self, arg_parser):
        arg_parser.add_argument('--paddingLength', type=int, default=226)
        arg_parser.add_argument('--model_fn', type=str, default='lstm')

        # augmentation
        arg_parser.add_argument('--random_scale_factor', type=float, default=0.0)
        arg_parser.add_argument('--augment_stroke_prob', type=float, default=0.0)
        arg_parser.add_argument('--img_scale_ratio', type=float, default=1.0)
        arg_parser.add_argument('--img_rotate_angle', type=float, default=0.0)
        arg_parser.add_argument('--img_translate_dist', type=float, default=0.0)

        arg_parser.add_argument('--disable_augmentation', action='store_false')
        return arg_parser

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

    def run(self):
        weight_decay = self.config['weight_decay']
        lr = self.config['lr']
        lr_step = self.config['lr_step']
        num_epochs = self.config['num_epoch']
        valid_freq = self.config['valid_freq']
        self.config['data_img_dir'] = None # set none to avoid load figures

        train_data = {
            m: QuickDrawDataset(
                m,
                self.config['data_seq_dir']
            ) for m in self.modes
        }
        self.prepare_dataset(train_data)
        num_categories = len(self.config['categories'])
        self.logger.info(f"Number of categories: {num_categories}")

        net = self.create_model(num_categories)
        data_loaders = self.create_data_loaders(train_data)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.params_to_optimize(weight_decay, self.weight_decay_excludes()), lr)
        if lr_step > 0:
            lr_exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5)
        else:
            lr_exp_scheduler = None

        ckpt_prefix = self.checkpoint_prefix()
        ckpt_nets = self.config['ckpt_nets']
        if ckpt_prefix is not None:
            loaded_paths = net.load(ckpt_prefix, ckpt_nets)
            self.logger.info(f"load pretrained model from {loaded_paths}")

        best_acc_record = {'valid': [], 'test': []}
        for seed in self.config['seed']:
            self.logger.info(f"Fix seed {seed}.")
            fix_seed(seed)
            best_val_acc = 0.0
            best_test_acc = 0.0
            best_epoch = -1

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

                    if mode == 'valid':
                        if epoch_acc > best_val_acc:
                            self.logger.info("New best valid acc, save model to disk.")
                            best_val_acc = epoch_acc
                            best_epoch = epoch
                            net.save(self.model_dir, f'best_{seed}')
                    if mode == 'test' and best_epoch == epoch:
                        best_test_acc = epoch_acc

            self.logger.info(f"Best valid acc: {best_val_acc:.4f}, test acc: {best_test_acc:.4f}, "
                             f"corresponding epoch: {best_epoch}.")
            best_acc_record['valid'].append(best_val_acc)
            best_acc_record['test'].append(best_test_acc)

        self.logger.info(
            f"Average valid acc: {np.mean(best_acc_record['valid']):.4f}±{np.std(best_acc_record['valid']):.4f}\n"
            f"Average test acc: {np.mean(best_acc_record['test']):.4f}±{np.std(best_acc_record['test']):.4f}")

        for m in self.modes:
            train_data[m].dispose()
