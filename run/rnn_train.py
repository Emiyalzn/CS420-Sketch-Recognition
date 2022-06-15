import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from datetime import datetime
import ast

from dataset.dataset import SketchDataset
from models.rnnmodels import BiLSTM
from models.sketch_rnn import SketchRNN

from .base_train import BaseTrain

class SketchRNNTrain(BaseTrain):
    def __init__(self, args=None):
        local_dir = os.path.join(f"results", f'rnn-{datetime.now().strftime("%Y%m%d-%H%M")}')
        super(SketchRNNTrain, self).__init__(local_dir, args)

    def add_args(self, arg_parser):
        arg_parser.add_argument('--data_seq_dir', type=str, default=None)
        arg_parser.add_argument('--paddingLength', type=int, default=226)

        arg_parser.add_argument('--disable_augmentation', action='store_true')
        return arg_parser




