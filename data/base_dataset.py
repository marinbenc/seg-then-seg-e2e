import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import albumentations as A

import utils

class BaseDataset(Dataset):
  """
  An abstract base class for datasets.

  Attributes:
    directory: The directory to load the dataset from. One of 'train', 'test', 'valid' or 'all'.
    augment: Whether to augment the dataset.
  """

  dataset_folder = None
  width = 256
  height = 256

  in_channels = 1
  out_channels = 1

  def __init__(self, directory, subset, input_size, augment=True, subjects=None):
    self.mode = directory
    self.augment = augment
    self.subset = subset
    self.input_size = input_size

    if directory == 'all' or subjects is not None:
      directories = ['train', 'valid', 'test']
    else:
      directories = [directory]

    self.file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), self.dataset_folder, subset, directory)
      directory_files = utils.listdir(p.join(directory, f'label_{input_size}'))
      directory_files = [p.join(directory, f'label_{input_size}', f) for f in directory_files]
      directory_files.sort()
      self.file_names += directory_files
      self.file_names.sort()

    if subjects is not None and subjects != 'all':
      self.file_names = [f for f in self.file_names if p.basename(f) in subjects]

  def get_item_np(self, idx):
    """
    Gets the raw unprocessed item in as a numpy array.
    """
    raise NotImplementedError('get_item_np() not implemented')
    
  def __len__(self):
    length = len(self.file_names)
    return length

  def __getitem__(self, idx):
    """
    Gets the raw unprocessed item in as a numpy array.
    """
    raise NotImplementedError('get_item_np() not implemented')
