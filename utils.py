import os

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn


def find_latest_ckpt(path):
  ps = os.listdir(path)
  ps = [os.path.basename(p) for p in ps if p.endswith('pth')]
  if len(ps) == 0:
    return None
  ps.sort(key=lambda a: int(os.path.basename(a).split('.')[-2]))
  return os.path.join(path, ps[-1])


def parse_config(config_file):
  if not os.path.isfile(config_file):
    raise FileNotFoundError("File %s don't exist" % config_file)
  with open(config_file, 'r') as f:
    content = f.read()
  __locals = {}
  exec("c = %s" % content, __locals)
  c = __locals['c']
  return c['config'], c['data_config']


def get_optimizer(config, parameters):
  config = config.copy()
  t = config.pop('type')
  return getattr(optim, t)(parameters, **config)


def get_scheduler(config, optimizer):
  config = config.copy()
  t = config.pop('type')
  return getattr(lr_scheduler, t)(optimizer, **config)


def get_criterion(config):
  config = config.copy()
  t = config.pop('type')
  return getattr(nn, t)(reduction='none', **config)

