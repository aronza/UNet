import logging
from glob import glob
from os import listdir

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
    return np.mean(flat), np.std(flat)


def expand_file_indices(files, num_slices):
    return [idx for file_no in files for idx in range(file_no * num_slices, (file_no + 1) * num_slices)]


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_postfix, device):
        self.tags = [file[:file.index(img_postfix)] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.img_files = [get_nii_files(imgs_dir, tag) for tag in self.tags]
        self.mask_files = [get_nii_files(masks_dir, tag) for tag in self.tags]

        logging.info(f'Input shape: {self.img_files[0].shape}')

        # self.mean, self.std = calculate_stats(self.img_files)

        self.slices = build_slices(self.img_files[0].shape)

        num_img_files = len(self.img_files)
        num_mask_files = len(self.mask_files)
        assert num_img_files == num_mask_files, \
            "There must be equal number of Images and Masks " + str(num_img_files) + " vs " + str(num_mask_files)

        logging.info(f'Creating dataset with {len(self.tags)} examples and {len(self.slices)} slices each')
        self.length = num_img_files * len(self.slices)
        self.device = device

    def split_to_loaders(self, validation_ratio, test_ratio, batch_size):
        """
        Splits the data set into training, validation and testing based on the given ratios and returns
        data loaders pointing to these subsets.

        :param validation_ratio: Percentages of whole data set to be used for validation
        :param test_ratio: Percentages of whole data set to be used for testing
        :param batch_size: Batch size for the data loader
        :return: List of torch.utils.data.DataLoader in order of training, validation and testing.
        """
        train_files, test_files = train_test_split(range(len(self.img_files)), test_size=test_ratio)
        val_train_ratio = validation_ratio / (1 - test_ratio)
        train_files, validation_files = train_test_split(train_files, test_size=val_train_ratio)

        logging.info(f'Test images: {test_files}')
        logging.info(f'Validation images: {validation_files}')

        subsets = [expand_file_indices(files, len(self.slices))
                   for files in [train_files, validation_files, test_files]]

        return [DataLoader(Subset(self, subset), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
                for subset in subsets]

    def __len__(self):
        return self.length

    def pre_process(self, img_nd, mask_nd):
        # img_nd = standardize(img_nd, self.mean, self.std)

        img_nd = np.expand_dims(img_nd, axis=0)

        img_nd = torch.from_numpy(img_nd)
        mask_nd = torch.from_numpy(mask_nd)

        img_nd = img_nd.to(device=self.device, dtype=torch.float32)
        mask_nd = mask_nd.to(device=self.device, dtype=torch.int64)

        return img_nd, mask_nd

    def __getitem__(self, idx):
        file_idx = idx // len(self.slices)
        slice_idx = idx % len(self.slices)

        _slice = self.slices[slice_idx]

        #  Only loads the sliced img to data
        img = self.img_files[file_idx][_slice]
        mask = self.mask_files[file_idx][_slice]

        img, mask = self.pre_process(img, mask)

        return {'image': img, 'mask': mask}
