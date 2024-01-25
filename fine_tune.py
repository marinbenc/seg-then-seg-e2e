import utils

import random
import os
import shutil
import os.path as p
from pathlib import Path
import cv2 as cv

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import segmentation_models_pytorch as smp

from torch.utils.tensorboard import SummaryWriter

import argparse
import datetime
import numpy as np

import data.datasets as data
import seg.loss as loss
import seg.train_seg as seg
import stn.train_stn as stn
import model as m
import stn.stn_losses as losses
from stn.stn_model import STN

from torch.nn import BCELoss
import kornia as K

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

device = 'cuda'

def get_model(dataset, log_name, fold):
  stn_model = stn.get_model(dataset)
  
  # add one more channel for the STN output mask
  seg_model = seg.get_model(dataset, num_channels=dataset.in_channels)
  seg_checkpoint_f = p.join('runs', log_name, 'seg', f'fold_{fold}', 'seg_best.pth')
  if p.exists(seg_checkpoint_f):
    seg_checkpoint = torch.load(seg_checkpoint_f)
    seg_model.load_state_dict(seg_checkpoint['model'])
    stn_model.loc_net.load_state_dict(seg_checkpoint['model'])
    # freeze stn model loc_net
    # print('Freezing stn model loc_net')
    # for param in stn_model.loc_net.parameters():
    #   param.requires_grad = False
  else:
    print('No seg checkpoint found, using random weights')

  # Add a channel for the initial seg mask
  seg_model.encoder.set_in_channels(dataset.in_channels + 1)

  model = m.TransformedSegmentation(stn_model, seg_model)
  return model

def transform_input(loader_output):
    x, x_highres, target = loader_output
    input = {'x': x, 'x_highres': x_highres}
    return input, target
    
def fine_tune(batch_size, epochs, lr, folds, dataset, subset, input_size, log_name):
    def worker_init(worker_id):
        np.random.seed(2022 + worker_id)

    os.makedirs(f'runs/{log_name}/fine', exist_ok=True)

    datasets = data.get_kfolds_datasets(dataset, subset, folds, log_name, input_size)

    for fold, (train_dataset, val_dataset) in enumerate(datasets):
      print(f'Fold {fold}')
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init)
      val_loader = DataLoader(val_dataset, worker_init_fn=worker_init)

      model = get_model(train_dataset, log_name, fold)
      model.to(device)
      model.output_theta = True

      optimizer = optim.Adam(model.parameters(), lr=lr)
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

      loss_fn = loss.DiceLoss()

      def calculate_loss(output, target):
          output_loc_img, output_img, output_theta = output
          loc_img_loss = loss_fn(output_loc_img, target)

          pred = output_img[0].squeeze().detach().cpu().numpy()
          gt = target[0].squeeze().detach().cpu().numpy()

          # plt_img = np.zeros((pred.shape[0], pred.shape[1], 3))
          # plt_img[:, :, 0] = pred
          # plt_img[:, :, 1] = gt
          # plt_img *= 255
          # plt_img = plt_img.astype(np.uint8)
          # plt.imshow(plt_img)
          # plt.show()

          alpha = 0.5
          beta = 1 - alpha
          img_loss = loss_fn(output_img, target) * alpha + loc_img_loss * beta
          return img_loss

      os.makedirs(f'runs/{log_name}/fine/fold_{fold}', exist_ok=True)

      writer = SummaryWriter(log_dir=f'runs/{log_name}/fine/fold_{fold}')
      for epoch in range(epochs):
        early_stop = utils.train(
          model, calculate_loss, optimizer, epoch, train_loader, val_loader, 
          writer=writer, checkpoint_name='fine_best.pth', scheduler=scheduler,
          transform_input=transform_input)
        if early_stop:
          break
      writer.close()

#TODO: Save arguments json file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine tuning'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--dataset', type=str, choices=data.dataset_choices, default='lesion', help='which dataset to use'
    )
    parser.add_argument(
        '--subset', type=str, choices=data.lesion_subsets, default='isic', help='which dataset to use'
    )
    parser.add_argument(
        '--input-size', type=int, default=128, help='input size of images',
    )
    parser.add_argument(
        '--folds', type=int, default=5, help='number of folds to use for cross validation',
    )
    parser.add_argument(
        '--log-name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), help='name of folder where checkpoints are stored',
    )
    args = parser.parse_args()
    log_dir = Path(f'runs/{args.log_name}')
    if p.exists(log_dir/'fine'):
        shutil.rmtree(log_dir/'fine')

    utils.save_args(args, 'fine')
    fine_tune(**vars(args))