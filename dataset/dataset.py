import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os.path as osp
import pickle

class QuickDrawDataset(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test':2}

    def __init__(self, mode):
        self.root_dir = 'data/'
        self.mode = mode
        self.data = None

        with open(osp.join(self.root_dir, 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']
            self.indices = saved_pkl['indices'][self.mode_indices[mode]]

        print('[*] Created a new {} dataset: {}'.format(mode, self.root_dir))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(osp.join(self.root_dir, 'quickdraw_{}.hdf5'.format(self.mode)), 'r')

        index_tuple = self.indices[idx]
        cid = index_tuple[0]
        sid = index_tuple[1]
        sketch_path = '/sketch/{}/{}'.format(cid, sid)

        sid_points = np.array(self.data[sketch_path][()], dtype=np.float32)
        sample = {'points3': sid_points, 'category': cid}
        return sample

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.data is not None:
            self.data.close()

    def num_categories(self):
        return len(self.categories)

    def get_name_prefix(self):
        return 'QuickDraw-{}'.format(self.mode)

def train_data_collate(batch):
    length_list = [len(item['points3']) for item in batch]
    max_length = max(length_list)

    points3_padded_list = list()
    points3_offset_list = list()
    intensities_list = list()
    category_list = list()
    for item in batch:
        points3 = item['points3']
        points3_length = len(points3)
        points3_padded = np.zeros((max_length, 3), np.float32)
        points3_padded[:, 2] = np.ones((max_length,), np.float32)
        points3_padded[0:points3_length, :] = points3
        points3_padded_list.append(points3_padded)

        points3_offset = np.copy(points3_padded)
        points3_offset[1:points3_length, 0:2] = points3[1:, 0:2] - points3[:points3_length - 1, 0:2]
        points3_offset_list.append(points3_offset)

        intensities = np.zeros((max_length,), np.float32)
        intensities[:points3_length] = 1.0 - np.arange(points3_length, dtype=np.float32) / float(points3_length - 1)
        intensities_list.append(intensities)

        category_list.append(item['category'])

    batch_padded = {
        'points3': points3_padded_list,
        'points3_offset': points3_offset_list,
        'points3_length': length_list,
        'intensities': intensities_list,
        'category': category_list
    }

    sort_indices = np.argsort(-np.array(length_list))
    batch_collate = dict()
    for k, v in batch_padded.items():
        sorted_arr = np.array([v[idx] for idx in sort_indices])
        batch_collate[k] = torch.from_numpy(sorted_arr)
    return batch_collate
