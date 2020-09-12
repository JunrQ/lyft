import os
import importlib

import torch.optim as optim
from torch.optim import lr_scheduler


def find_latest_ckpt(path):
  ps = os.listdir(path)
  ps = [os.path.basename(p) for p in ps if p.endswith('pth')]
  if len(ps) == 0:
    return None
  ps.sort(key=lambda a: int(os.path.basename(a).split('.')[-2]))
  return os.path.join(path, ps[-1])


def parse_config(name):
  c = importlib.import_module('%s' % name)
  data_config = c.data_config.copy()
  config = c.config.copy()
  return config, data_config


def get_optimizer(config, parameters):
  config = config.copy()
  t = config.pop('type')
  return getattr(optim, t)(parameters, **config)


def get_scheduler(config, optimizer):
  config = config.copy()
  t = config.pop('type')
  return getattr(lr_scheduler, t)(optimizer, **config)
