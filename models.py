

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet


_MODELS = {}
def register_model(cls):
  n = cls.__name__
  if n in _MODELS:
    raise ValueError("Already exists")
  _MODELS[n] = _MODELS
  return cls


def get_model(name, *args, **kwargs):
  return _MODELS[name](*args, **kwargs)


@register_model
class ResNetModel(nn.Module):
  def __init__(self, depth, input_channels, num_outputs):
    super().__init__()
    depth_map = {
      18 : resnet.resnet18
      34 : resnet.resnet34,
      50 : resnet.resnet50,
      101 : resnet.resnet101
    }
    self.depth = depth
    self.backbone = depth_map[depth](pretrained=True, progress=True)

    self.backbone.conv1 = nn.Conv2d(
      input_channels,
      self.backbone.conv1.out_channels,
      kernel_size=self.backbone.conv1.kernel_size,
      stride=self.backbone.conv1.stride,
      padding=self.backbone.conv1.padding,
      bias=False)
    backbone_out_features = 512 if self.depth < 50 else 2048
    self.head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features=backbone_out_features, out_features=1024),
        nn.ReLU())
    self.logit = nn.Linear(1024, out_features=num_outputs)
        
  def forward(self, sample):
    x = sample["image"]

    x = self.backbone.conv1(x)
    x = self.backbone.bn1(x)
    x = self.backbone.relu(x)
    x = self.backbone.maxpool(x)

    x = self.backbone.layer1(x)
    x = self.backbone.layer2(x)
    x = self.backbone.layer3(x)
    x = self.backbone.layer4(x)

    x = self.backbone.avgpool(x)
    x = torch.flatten(x, 1)

    x = self.head(x)
    x = self.logit(x)
    return x



