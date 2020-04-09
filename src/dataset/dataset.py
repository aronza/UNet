from os.path import splitext
from os import listdir
import nibabel as nib
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import numpy as np


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_prefix, mask_prefix, scale=1):
        self.imgs_dir = imgs_dir
        self.img_prefix = img_prefix
        self.masks_dir = masks_dir
        self.mask_prefix = mask_prefix
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [file for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        print(self.ids)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, nib_img):
        img_nd = nib_img.get_fdata()

        if len(img_nd.shape) == 3:
            img_nd = np.expand_dims(img_nd, axis=0)

        return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path = self.imgs_dir  + idx + '*'
        mask_path = self.masks_dir + idx.replace(self.img_prefix, self.mask_prefix) + '*'
        
        img_file = glob(img_path)
        mask_file = glob(mask_path)

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {img_path}: {img_file}'
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {mask_path}: {mask_file}'

        mask = nib.load(mask_file[0])
        img = nib.load(img_file[0])

        assert img.shape == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
