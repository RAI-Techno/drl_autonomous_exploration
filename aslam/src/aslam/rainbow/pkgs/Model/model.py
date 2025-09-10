#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
  def __init__(self, args, a_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.device = args.device
    self.action_space = a_space

    self.small_scope_occupancy_grid_model = nn.Sequential(nn.Conv2d(args.history_length, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm2d(16),   
                      nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
                      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                      nn.Conv2d(128, 32, kernel_size=1), nn.BatchNorm2d(32), nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.Flatten(), nn.Linear(800, 264),  nn.ReLU())

                  
    self.laser_scan_model   = nn.Sequential (nn.Conv1d(args.history_length, 8, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(8), nn.MaxPool1d(kernel_size=2, stride=2),
                         nn.Conv1d(8, 16, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(16), nn.MaxPool1d(kernel_size=2, stride=2),
                         nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(kernel_size=2, stride=2), 
                         nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(kernel_size=2, stride=2),
                         nn.Conv1d(64, 16, kernel_size=1), nn.BatchNorm1d(16),
                         nn.Flatten(), nn.Linear(576, 248),  nn.ReLU())
    
    self.fc_h_v   = NoisyLinear(512, 256, std_init = args.noisy_std)
    self.fc_h_a   = NoisyLinear(512, 256, std_init = args.noisy_std)
    self.fc_z_v   = NoisyLinear(256, self.atoms,        std_init = args.noisy_std)
    self.fc_z_a   = NoisyLinear(256, self.action_space* self.atoms, std_init = args.noisy_std)

  def forward(self, x, log=False):
    local_occupancy_grid = self.small_scope_occupancy_grid_model(x['local_occupancy_grid'])
    laser_scan = self.laser_scan_model(x['laser_scan'])
    x = torch.cat((local_occupancy_grid, laser_scan), dim = -1)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage v stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams for v
    
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with v_action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with v_action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
