import logging
from glob import glob
from os import listdir
from pprint import pprint

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader

from .slicer import build_slices


def get_nii_files(directory, tag):
    path = directory + tag + '*'
    # mask_path = self.masks_dir + tag.replace(self.img_prefix, self.mask_prefix) + '*'

    file = glob(path)

    assert len(file) == 1, \
        f'Either no file or multiple files found for the ID {directory}: {file}'

    img = nib.load(file[0])

    return img.dataobj


def standardize(m, mean, std):
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    """
    return (m - mean) / np.clip(std, a_min=1e-6, a_max=None)
        
        
def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img for img in images]
    )
    return np.mean(flat), np.std(flat), np.count_nonzero(flat), flat.size

def expand_file_indices(files, num_slices):
    return [range(file_no * num_slices, (file_no + 1) * num_slices) for file_no in files]
    

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_postfix):
        self.tags = [file[:file.index(img_postfix)] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.img_files = [get_nii_files(imgs_dir, tag) for tag in self.tags]
        self.mask_files = [get_nii_files(masks_dir, tag) for tag in self.tags]
        
        self.mean, self.std, zeroes, size = calculate_stats(self.img_files)

        self.slices = build_slices(self.img_files[0].shape)
        
        num_img_files = len(self.img_files)
        num_mask_files = len(self.mask_files)
        assert num_img_files == num_mask_files, \
            "There must be equal number of Images and Masks " + str(num_img_files) + " vs " + str(num_mask_files)

        logging.info(f'Input shape: {self.img_files[0].shape}')
        logging.info(f'Creating dataset with {len(self.tags)} examples and {len(self.slices)} slices each')

    def split_to_loaders(self, test_ratio, validation_ratio, batch_size):
        train_files, test_files = train_test_split(range(len(self.img_files)), test_size=test_ratio)
        val_train_ratio = validation_ratio / (1 - test_ratio)
        train_files, validation_files = train_test_split(train_files, test_size=val_train_ratio)
        
        subsets = [expand_file_indices(train_files, len(self.slices)) 
                    for files in [train_files, validation_files, test_files]]
        
        return [DataLoader(Subset(self, subset), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
                for subset in subsets]

    def __len__(self):
        return len(self.img_files) * len(self.slices)

    def pre_process(self, img_nd, mask_nd):
        
        img_nd = standardize(img_nd, self.mean, self.std)

        if len(img_nd.shape) == 3:
            img_nd = np.expand_dims(img_nd, axis=0)
            # mask_nd = np.expand_dims(mask_nd, axis=0)

        return img_nd, mask_nd

    def __getitem__(self, idx):
        file_idx = idx // len(self.slices)
        slice_idx = idx % len(self.slices)
        
        _slice = self.slices[slice_idx]
        #  Only loads the sliced img to data
        img = self.img_files[file_idx][_slice]
        mask = self.mask_files[file_idx][_slice]

        img, mask = self.pre_process(img, mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
