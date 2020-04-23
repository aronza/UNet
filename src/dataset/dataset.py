import logging
from glob import glob
from os import listdir

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .slicer import Slicer


def get_nii_files(directory, tag):
    path = directory + tag + '*'
    # mask_path = self.masks_dir + tag.replace(self.img_prefix, self.mask_prefix) + '*'

    file = glob(path)

    assert len(file) == 1, \
        f'Either no file or multiple files found for the ID {directory}: {file}'

    img = nib.load(file[0])

    return img.dataobj


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_postfix):
        self.tags = [file[:file.index(img_postfix)] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.img_files = [get_nii_files(imgs_dir, tag) for tag in self.tags]
        self.mask_files = [get_nii_files(masks_dir, tag) for tag in self.tags]

        slicer = Slicer(self.img_files, self.mask_files)
        self.img_slices = slicer.img_slices
        self.mask_slices = slicer.mask_slices

        assert len(self.img_slices) == self.mask_slices, "Images and Masks must be same shape "
        logging.info(f'Creating dataset with {len(self.tags)} examples')

    def __len__(self):
        return len(self.img_files) * len(self.img_slices)

    @classmethod
    def pre_process(cls, img_nd):

        if len(img_nd.shape) == 3:
            img_nd = np.expand_dims(img_nd, axis=0)

        return img_nd

    def __getitem__(self, idx):
        file_idx = idx // len(self.img_slices)
        slice_idx = idx % len(self.img_slices)

        #  Only loads the sliced img to data
        img = self.img_files[file_idx][slice_idx]
        mask = self.mask_files[file_idx][slice_idx]

        img = self.pre_process(img)
        mask = self.pre_process(mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
