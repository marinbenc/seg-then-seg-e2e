import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import utils

class TransformedSegmentation(nn.Module):
  """
  Transformed Segmentation Network

  Attributes:
    stn: The spatial transformer network.
    seg: The segmentation network.
    output_theta: if `True`, the model output will be `(y, theta)`, otherwise it will be `y`
  """
  def __init__(self, stn, seg):
      super(TransformedSegmentation, self).__init__()

      stn.output_theta = True
      self.output_stn_mask = True

      # Spatial transformer localization-network
      self.stn = stn
      self.seg = seg
      self._iters = 0
      self.output_theta = False

  def forward(self, x, x_highres=None):
    # transform image
    y_inital, y_initial_t, x_t, bboxes = self.stn(x, x_highres)

    # segment transformed image

    # plt.imshow(y_loc_net[0, 0].detach().cpu().numpy())
    # plt.title('y_loc_net')
    # plt.show()
    # plt.imshow(y_loc_net_t[0, 0].detach().cpu().numpy())
    # plt.title('y_loc_net_t')
    # plt.show()

    skip_stn = False

    # two channels, y_loc_net_t is the mask and x_t is the transformed image
    if skip_stn:
      seg_x = x
    else:
      # First channel: x_t
      # Second channel: y_initial_t
      seg_x = torch.cat([x_t, y_initial_t], dim=1)
    y_cropped = self.seg(seg_x)

    # plt.imshow(y_t[0, 0].detach().cpu().numpy())
    # plt.title('y_t')
    # plt.show()

    y = torch.zeros_like(x)
    y = y[:, 0:1, :, :]
    
    # reverse crop
    if not skip_stn:
      for i in range(y_cropped.shape[0]):
        bboxes_i = bboxes[i].int()
        h = bboxes_i[3] - bboxes_i[1]
        w = bboxes_i[2] - bboxes_i[0]
        x1, y1, x2, y2 = bboxes_i
        y_scaled = F.interpolate(y_cropped[i].unsqueeze(0), size=(h.int(), w.int()), mode='nearest')
        y[i, :, y1:y2, x1:x2] = y_scaled
        # plt.imshow(y_cropped[i, 0].detach().cpu().numpy())
        # plt.title('y_cropped')
        # plt.show()
        # plt.imshow(y[i, 0].detach().cpu().numpy())
        # plt.title('y')
        # plt.show()
    else:
      y = y_cropped

    #utils.show_torch(imgs=[x[0] + 0.5, x_t[0] + 0.5, y_t[0], y[0]])

    # self._iters += 1

    # if self._iters % 100 == 0:
    #   utils.show_torch(imgs=[x[0] + 0.5, x_t[0] + 0.5, y_t[0], y[0]])

    outputs = [y]
    if self.output_stn_mask:
      outputs = [y_inital] + outputs
    if self.output_theta:
      outputs = outputs + [bboxes]

    if len(outputs) == 1:
      return outputs[0]
    else:
      return tuple(outputs)
