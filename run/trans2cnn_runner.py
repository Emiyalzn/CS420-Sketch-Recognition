import os.path
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

from dataset.dataset import QuickDrawDataset, r2cnn_collate
from dataset.data_utils import seq_5d_to_3d
from models.cnnmodels import CNN_MODELS, CNN_IMAGE_SIZES
from models.trans2cnn import Trans2CNN
from models.trans_utils import compute_reconstruction_loss
from .base_runner import BaseRunner


class Trans2CNNRunner(BaseRunner):
    def __init__(self, args=None, local_dir=None):
        if local_dir is None:
            local_dir = os.path.join("results", f'trans2cnn-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        super(Trans2CNNRunner, self).__init__(local_dir, args)

    def add_args(self, arg_parser):
        arg_parser.add_argument('--dropout', type=float, default=0.1)
        arg_parser.add_argument('--intensity_channels', type=int, default=8)
        arg_parser.add_argument('--thickness', type=float, default=1.0)
        arg_parser.add_argument('--model_fn', type=str, default='efficientnet_b0')

        # for augmentation
        arg_parser.add_argument('--stroke_removal_prob', type=float, default=0.0)
        arg_parser.add_argument('--do_augmentation', action='store_true')

        # for reconstruction
        arg_parser.add_argument('--do_reconstruction', action='store_true')
        arg_parser.add_argument('--recon_weight', type=float, default=1.0)

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
        model_fn = CNN_MODELS[self.config['model_fn']]
        img_size = CNN_IMAGE_SIZES[self.config['model_fn']]
        thickness = self.config['thickness']
        do_reconstruction = self.config['do_reconstruction']

        return Trans2CNN(model_fn,
                         img_size,
                         thickness,
                         num_categories,
                         intensity_channels=intensity_channels,
                         do_reconstruction=do_reconstruction,
                         dropout=dropout,
                         device=self.device)

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        is_train = mode == 'train'

        points = data_batch['points3'].to(self.device).contiguous()
        points_offset = data_batch['points5_offset'].to(self.device).contiguous()
        category = data_batch['category'].to(self.device).contiguous()

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, recon_output = model(points_offset, points, is_train)
            loss = criterion(logits, category)
            if self.config['do_reconstruction']:
                tar_real = points_offset[:, 1:, ...]
                recon_loss = compute_reconstruction_loss(tar_real, recon_output)
                loss += recon_loss * self.config['recon_weight']
            if is_train:
                loss.backward()
                optimizer.step()

        return logits, loss, category

    def embed_batch(self, model, data_batch):
        points = data_batch['points3'].to(self.device).contiguous()
        points_offset = data_batch['points5_offset'].to(self.device).contiguous()
        categories = data_batch['category'].to(self.device).contiguous()
        feats = model.embed(points_offset, points, False)
        
        return feats, categories
    
    def decode_batch(self, model, cnnfeats):
        with torch.no_grad():
            decodings = model.decode(cnnfeats).cpu().numpy()
            # decodings = decodings[:, 1:, :] ?
            seqs_3d = []
            
            for i in range(decodings.shape[0]):
                seqs_3d.append(seq_5d_to_3d(decodings[i]))
            
        return decodings, np.array(seqs_3d)
    
    # def compute_mean_feat(self, modes=['test']):
    #     # During evaluate(), 'test' datasets are required
    #     self.modes = modes

    #     test_data = self.prepare_dataset()
    #     num_categories = len(self.config['categories'])
    #     self.logger.info(f"Number of categories: {num_categories}")
        
    #     data_loaders = self.create_data_loaders(test_data)

    #     for seed in self.config['seed']:
    #         # initialize network
    #         net = self.create_model(num_categories)
    #         path_prefix = os.path.join(self.model_dir, f'_iter_best_{seed}')
    #         net.load(path_prefix)

    #         self.logger.info(f"Read model of seed {seed}.")
    #         fix_seed(seed)

    #         self.logger.info('-' * 20)

    #         # self.modes = ['test']
    #         with torch.no_grad():
    #             for mode in self.modes:
    #                 self.logger.info(f"Start {mode} mode.")
    #                 net.eval_mode()
                    
    #                 confusion_matrix = torch.zeros((num_categories, num_categories), dtype=torch.int)
    #                 running_corrects_top1 = 0
    #                 running_corrects_top5 = 0
    #                 num_samples = 0
    #                 pbar = tqdm.tqdm(total=len(data_loaders[mode]))
    #                 for bid, data_batch in enumerate(data_loaders[mode]):
    #                     # self.step_counters[mode] += 1

    #                     logits, _, gt_category = self.forward_batch(net, data_batch, mode, BaseRunner.DummyOptimizer(), lambda _1, _2: 0)
    #                     _, predicts = torch.max(logits, 1)
    #                     confusion_matrix[gt_category, predicts] += 1
    #                     # predicts_accu = torch.sum(predicts == gt_category)
    #                     running_corrects = compute_running_corrects(logits, gt_category, (1, 5))
    #                     running_corrects_top1 += running_corrects[1]
    #                     running_corrects_top5 += running_corrects[5]

    #                     sampled_batch_size = gt_category.size(0)
    #                     num_samples += sampled_batch_size

    #                     pbar.update()
    #                 pbar.close()

    #                 epoch_acc_1 = float(running_corrects_top1) / float(num_samples)
    #                 epoch_acc_5 = float(running_corrects_top5) / float(num_samples)
    #                 self.logger.info(f"{mode}:\tTop1-Accuracy: {epoch_acc_1:.4f} | Top5-Accuracy: {epoch_acc_5:.4f}")
                    
    #                 for i, k in enumerate(self.config['categories']):
    #                     TP = confusion_matrix[i, i]
    #                     TN = torch.sum(confusion_matrix[:i, :i]) + torch.sum(confusion_matrix[:i, i+1:]) \
    #                     + torch.sum(confusion_matrix[i+1:, :i]) + torch.sum(confusion_matrix[i+1:, i+1:])
                
    #                     T = torch.sum(confusion_matrix[i])    # = TP + FN
    #                     P = torch.sum(confusion_matrix[:, i]) # = TP + FP
                        
    #                     FN = T - TP
    #                     FP = P - TP
                        
    #                     accuracy = (TP + TN) / (TP + FN + FP + TN)
    #                     precision = TP / (TP + FP)
    #                     recall = TP / (TP + FN)
    #                     f1 = 2 * precision * recall / (precision + recall)
                        
    #                     self.logger.info(f'{mode}-{k}: Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}')
    #                 np.save(os.path.join(self.local_dir, f"confusion_matrix_{mode}_{seed}.npy"), confusion_matrix.cpu().numpy())

    #     for m in self.modes:
    #         test_data[m].dispose()