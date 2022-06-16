import torch
from torch.utils.data import Dataset
import h5py
import math
import random
import numpy as np
import os
import os.path as osp
import pickle
import cv2
import six
from torchvision import transforms

import dataset.data_utils as utils


class QuickDrawDataset(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test':2}

    def __init__(self, mode, data_seq_dir):
        self.root_dir = data_seq_dir
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


class R2CNNDataset(Dataset):
    allowedModes = {'train', 'valid', 'test'}
    
    def __init__(self, 
                 mode: str,
                 data_seq_dir: str,
                 data_img_dir: str,                 # Dummy, unused
                 categories: list,
                 paddingLength: int=226,            # 226 in our dataset. For new dataset, this should be recomputed.
                 random_scale_factor: float=0.0,    # max randomly scale ratio (for sequences) (in absolute value)
                 augment_stroke_prob: float=0.0,    # data augmentation probability (for sequences)
                 img_scale_ratio: float=1.0,        # Dummy, unused
                 img_rotate_angle: float=0.0,       # Dummy, unused
                 img_translate_dist: float=0.0,     # Dummy, unused
                 disable_augmentation: bool=False   #! Whether to disable all augmentations
                 ) -> None:
        super(R2CNNDataset, self).__init__()
        
        assert (mode in SketchDataset.allowedModes), f"[x] mode '{mode}' is not supported."
        
        self.mode = mode
        self.categories = categories
        self._paddingLength = paddingLength
        self.random_scale_factor = random_scale_factor
        self.augment_stroke_prob = augment_stroke_prob
        self.disable_augmentation = disable_augmentation
        
        if (self.disable_augmentation):
            print(f"Data augmentation is disabled.")
        
        self.seqs = list()
        self.labels = list()

        for i, ctg in enumerate(categories):
            # load sequence data
            seq_path = os.path.join(data_seq_dir, 'r2cnn_' + ctg + '.npy')
            seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True).item()
                
            # import pdb; pdb.set_trace()
                
            print(f"[*] Loaded {len(seq_data[self.mode])} {self.mode} sequences from {ctg + '.npy'}")
            
            self.seqs.append(np.array(seq_data[self.mode], dtype=object))
            self.labels.append(i * np.ones([len(seq_data[self.mode])], dtype=int))
        
        # import pdb; pdb.set_trace()
        self.seqs = np.concatenate(self.seqs)
        self.labels = np.concatenate(self.labels)

    def __getitem__(self, index):
        # Sequence
        data = self.seqs[index].astype(dtype=np.double, order='A', copy=False)
        
        # Sequence Augmentation
        if (not self.disable_augmentation):
            data = self.random_scale_seq(self.seqs[index])
        if (self.augment_stroke_prob > 0 and not self.disable_augmentation):
            data = utils.augment_strokes(data, self.augment_stroke_prob)
        
        # Label Augmentation
        label = self.labels[index]
        
        sample = {
            'points3': data,
            'category': label
        }
        
        return sample
    
    def __len__(self):
        return len(self.labels)
    
    def random_scale_seq(self, data):
        """ Augment data by stretching x and y axis randomly [1-e, 1+e] """
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = data.astype(dtype=np.double, order='A', copy=False)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def num_categories(self):
        return len(self.categories)

    @property
    def maxSeqLen(self):
        maxLen = 0
        for seq in self.seqs:
            maxLen = max(maxLen, len(seq))
        return maxLen

    def dispose(self):
        pass


def r2cnn_collate(batch):
    length_list = [len(item['points3']) for item in batch]
    max_length = max(length_list)

    points3_padded_list = list()
    points3_offset_list = list()
    points5_offset_list = list()
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

        points5_padded = utils.seq_3d_to_5d(points3, max_length)
        points5_offset = np.copy(points5_padded)
        points5_offset[1:points3_length, 0:2] = points3[1:, 0:2] - points3[:points3_length - 1, 0:2]
        points5_offset_list.append(points5_offset)

        intensities = np.zeros((max_length,), np.float32)
        intensities[:points3_length] = 1.0 - np.arange(points3_length, dtype=np.float32) / float(points3_length - 1)
        intensities_list.append(intensities)

        category_list.append(item['category'])

    batch_padded = {
        'points3': points3_padded_list,
        'points3_offset': points3_offset_list,
        'points5_offset': points5_offset_list,
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

#! paddingLength is 226 in default. 
#! But it is recommended to manually set @property `paddingLength`
#! after obtaining maxSeqLen of train, valid and test sets.
class SketchDataset(Dataset):
    allowedModes = {'train', 'valid', 'test'}

    def __init__(self, 
                 mode: str,
                 data_seq_dir: str,
                 data_img_dir: str,
                 categories: list,
                 paddingLength: int=226,            # 226 in our dataset. For new dataset, this should be recomputed.
                 random_scale_factor: float=0.0,    # max randomly scale ratio (for sequences) (in absolute value)
                 augment_stroke_prob: float=0.0,    # data augmentation probability (for sequences)
                 img_scale_ratio: float=1.0,        # min randomly scaled ratio (for sequences) [0, 1]
                 img_rotate_angle: float=0.0,       # max randomly rotate angle (for images) (in absolute value, degree)
                 img_translate_dist: float=0.0,     # max randomly translate distance (for images) (in absolute value, pixel)
                 disable_augmentation: bool=False   #! Whether to disable all augmentations
                 ):
        super(SketchDataset, self).__init__()
        
        assert (mode in SketchDataset.allowedModes), f"[x] mode '{mode}' is not supported."

        self.mode = mode
        self.categories = categories
        self._paddingLength = paddingLength
        self.random_scale_factor = random_scale_factor
        self.augment_stroke_prob = augment_stroke_prob
        self.img_scale_ratio = img_scale_ratio
        self.img_rotate_angle = img_rotate_angle
        self.img_translate_dist = img_translate_dist
        self.disable_augmentation = disable_augmentation
        
        if (self.disable_augmentation):
            print(f"Data augmentation is disabled.")
        
        # self.seqs = None
        # self.imgs = None
        # self.labels = None
        
        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(img_scale_ratio, 1)),  #* including scaling and translation.
        #     transforms.RandomRotation(img_rotate_angle),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],    #* All pre-trained models expect input images normalized in the same way
        #                          std=[0.229, 0.224, 0.225])
        # ])

        self.seqs = list()
        self.imgs = list()
        self.labels = list()

        for i, ctg in enumerate(categories):
            # load sequence data
            curLen = None
            
            if (data_seq_dir is not None):
                seq_path = os.path.join(data_seq_dir, ctg + '.npz')
                if six.PY3:
                    seq_data = np.load(seq_path, encoding='latin1', allow_pickle=True)
                else:
                    seq_data = np.load(seq_path, allow_pickle=True)
                    
                print(f"[*] Loaded {len(seq_data[self.mode])} {self.mode} sequences from {ctg + '.npz'}")
                
                curLen = len(seq_data[self.mode])
                self.seqs.append(seq_data[self.mode])

            # load img data
            if (data_img_dir is not None):
                img_path = os.path.join(data_img_dir, ctg + '.npz')
                if six.PY3:
                    img_data = np.load(img_path, encoding='latin1', allow_pickle=True)
                else:
                    img_data = np.load(img_path, allow_pickle=True)
                
                if (curLen is not None):
                    assert (len(img_data[self.mode]) == curLen), f'[x] Category {ctg} has {len(img_data[self.mode])} images but {len(seq_data[self.mode])} sequences.'
                curLen = len(img_data[self.mode])
                print(f"[*] Loaded {len(img_data[self.mode])} {self.mode} images from {ctg + '.npz'}")
                
                self.imgs.append(img_data[self.mode])

            # create labels
            # if self.labels is None:
            #     self.labels = i * np.ones([len(seq_data[self.mode])], dtype=int)
            # else:
            #     self.labels = np.concatenate([self.labels, i * np.ones([len(seq_data[self.mode])], dtype=int)])
            self.labels.append(i * np.ones([curLen], dtype=int))
        
        self.seqs = np.concatenate(self.seqs) if (data_seq_dir is not None) else None
        self.imgs = np.concatenate(self.imgs) if (data_img_dir is not None) else None
        self.labels = np.concatenate(self.labels)

    def __getitem__(self, index):
        # Sequence
        if (self.seqs is not None):
            data = self.seqs[index].astype(dtype=np.double, order='A', copy=False)
            
            # Sequence Augmentation
            if (not self.disable_augmentation):
                data = self.random_scale_seq(self.seqs[index])
            if (self.augment_stroke_prob > 0 and not self.disable_augmentation):
                data = utils.augment_strokes(data, self.augment_stroke_prob)

            length = data.shape[0]
            strokes_3d = np.pad(data, ((0, self.paddingLength - data.shape[0]), (0, 0)), 'constant', constant_values=0)
            strokes_5d = utils.seq_3d_to_5d(data, self.paddingLength)
        else:
            length = 0
            strokes_3d = 0
            strokes_5d = 0

        # Image
        if (self.imgs is not None):
            data = np.copy(self.imgs[index])
            img = np.reshape(data, [1,data.shape[0],data.shape[1]])
            # Image Augmentation
            if (not self.disable_augmentation):
                img = self.random_scale_img(img)
                img = self.random_rotate_img(img)
                img = self.random_translate_img(img)
            img = img.astype(dtype=np.float32, order='A', copy=False) / 255.0
        else:
            img = 0
        
        # Label Augmentation
        label = self.labels[index]
        return strokes_3d, strokes_5d, length, img, label
        
        # data = np.copy(self.imgs[index])
        # img = np.reshape(data, [1,data.shape[0],data.shape[1]]).astype(dtype=np.float32) / 255.0
        # img = self.transform(img.repeat([1, 3, 1, 1]))
        # return strokes_5d, img, label
    
    def __len__(self):
        return len(self.labels)

    # TODO: Should data augmentation of image be coherent with that of sequence?
    def random_scale_seq(self, data):
        """ Augment data by stretching x and y axis randomly [1-e, 1+e] """
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = data.astype(dtype=np.double, order='A', copy=False)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def stroke_removal(self, data):
        pass

    def stroke_deform(self, data):
        pass

    def random_scale_img(self, data):
        """ Randomly scale image """
        out_imgs = np.copy(data)
        for i in range(data.shape[0]):
            in_img = data[i]
            ratio = random.uniform(self.img_scale_ratio,1)
            out_img = utils.rescale(in_img, ratio)
            out_imgs[i] = out_img
        return out_imgs

    def random_rotate_img(self, data):
        """ Randomly rotate image """
        out_imgs = np.copy(data)
        for i in range(data.shape[0]):
            in_img = data[i]
            angle = random.uniform(-self.img_rotate_angle,self.img_rotate_angle)
            out_img = utils.rotate(in_img, angle)
            out_imgs[i] = out_img
        return out_imgs
            
    def random_translate_img(self, data):
        """ Randomly translate image """
        out_imgs = np.copy(data)
        for i in range(data.shape[0]):
            in_img = data[i]
            dx = random.uniform(-self.img_translate_dist,self.img_translate_dist)
            dy = random.uniform(-self.img_translate_dist,self.img_translate_dist)
            out_img = utils.translate(in_img, dx, dy)
            out_imgs[i] = out_img
        return out_imgs
    
    def num_categories(self):
        return len(self.categories)

    @property
    def maxSeqLen(self):
        maxLen = 0
        for seq in self.seqs:
            maxLen = max(maxLen, len(seq))
        return maxLen
    
    @property
    def paddingLength(self):
        return self._paddingLength
    
    @paddingLength.setter
    def paddingLength(self, newPaddingLength):
        self._paddingLength = newPaddingLength

    def dispose(self):
        pass
