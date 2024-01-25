import os.path as p
import shutil

from seg.train_seg import train_seg
from fine_tune import fine_tune

default_params = {
  'epochs': 100,
  'dataset': 'lesion',
  'folds': 5,
  'lr': 0.0001,
}

input_size_to_batch_size = {
  64: 16,
  128: 16,
  256: 16,
  512: 4,
}

params_seg = [
  # {
  #   **default_params,
  #   'subset': 'ph2',
  # },
  # {
  #   **default_params,
  #   'subset': 'dermofit',
  # },
  # {
  #   **default_params,
  #   'subset': 'dermquest',
  # },
  # {
  #   **default_params,
  #   'subset': 'dermis',
  # },
  {
    **default_params,
    'subset': 'isic',
  },
]

params_fine = [
  # {
  #   **default_params,
  #   'subset': 'ph2',
  # },
  # {
  #   **default_params,
  #   'subset': 'dermofit',
  # },
  # {
  #   **default_params,
  #   'subset': 'dermquest',
  # },
  # {
  #   **default_params,
  #   'subset': 'dermis',
  # },
  {
    **default_params,
    'subset': 'isic',
  },
]

for (param_seg, param_fine) in zip(params_seg, params_fine):
  for input_size in [64, 256, 512]:
    batch_size = input_size_to_batch_size[input_size]
    param_seg['batch_size'] = batch_size
    param_fine['batch_size'] = batch_size
    param_seg['input_size'] = input_size
    param_fine['input_size'] = input_size
    log_name = f'{param_seg["subset"]}_final_{param_seg["input_size"]}'
    log_dir = p.join('runs', log_name)
    if p.exists(log_dir):
      shutil.rmtree(log_dir)
    param_seg['log_name'] = log_name
    param_fine['log_name'] = log_name
    train_seg(**param_seg)
    fine_tune(**param_fine)
