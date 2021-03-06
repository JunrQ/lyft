

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet


_MODELS = {}
def register_model(cls):
  n = cls.__name__
  if n in _MODELS:
    raise ValueError("Already exists")
  _MODELS[n] = cls 
  return cls


def get_model(name, *args, **kwargs):
  return _MODELS[name](*args, **kwargs)


depth_map = {
  18 : resnet.resnet18,
  34 : resnet.resnet34,
  50 : resnet.resnet50,
  101 : resnet.resnet101
}


@register_model
class ResNetModel(nn.Module):
  def __init__(self, depth, input_channels, num_outputs,
               predict_classes=False, down_sample_times=4):
    super().__init__()
    self.depth = depth
    self.predict_classes = predict_classes
    self.down_sample_times = down_sample_times
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

    if self.down_sample_times > 4:
      extra_downsample_list = []
      last_channels = 64
      for _ in range(4, self.down_sample_times):
        extra_downsample_list += [
          nn.Conv2d(last_channels, last_channels, kernel_size=3, stride=2,
                    padding=1, bias=False),
          nn.BatchNorm2d(last_channels),
          nn.ReLU(),
        ]
      self.extra_downsample_list = nn.Sequential(*extra_downsample_list)

    if self.predict_classes:
      self.cls_head = nn.Sequential(
        nn.Linear(in_features=backbone_out_features, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=num_outputs // 2))

  def forward(self, sample):
    x = sample["image"]

    x = self.backbone.conv1(x)
    x = self.backbone.bn1(x)
    x = self.backbone.relu(x)
    x = self.backbone.maxpool(x)

    if self.down_sample_times > 4:
      x = self.extra_downsample_list(x)

    x = self.backbone.layer1(x)
    x = self.backbone.layer2(x)
    x = self.backbone.layer3(x)
    x = self.backbone.layer4(x)

    x = self.backbone.avgpool(x)
    x = torch.flatten(x, 1)

    r = self.head(x)
    r = self.logit(r)

    if self.predict_classes:
      c = F.sigmoid(self.cls_head(x))
      return r, c
    else:
      return r


class Encoder(nn.Module):
  def __init__(self, input_channels, 
               depth=50, encoder_feature_size=1024):
    super(Encoder, self).__init__()
    self.enc_image_size = encoded_image_size
    resnet = depth_map[depth](pretrained=True)
    resnet.conv1 = nn.Conv2d(
      input_channels,
      resnet.conv1.out_channels,
      kernel_size=resnet.conv1.kernel_size,
      stride=resnet.conv1.stride,
      padding=resnet.conv1.padding,
      bias=False)
    backbone_out_features = 512 if self.depth < 50 else 2048
    modules = list(resnet.children())[:-2]
    self.resnet = nn.Sequential(*modules)
    self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(backbone_out_features, encoder_feature_size, bias=False)
    # self.fine_tune()

  def forward(self, images):
    out = self.resnet(images)
    out = self.adaptive_pool(out)
    b, c, _, _ = out.shape
    out = out.view((b, c))
    out = self.fc(out)
    return out

  def fine_tune(self, fine_tune=True):
    for p in self.resnet.parameters():
      p.requires_grad = False
    # If fine-tuning, only fine-tune convolutional blocks 2 through 4
    for c in list(self.resnet.children())[5:]:
      for p in c.parameters():
        p.requires_grad = fine_tune


class Decoder(nn.Module):
  def __init__(self, decoder_dim=512,
               encoder_dim=512,
               prediction_length=50,
               input_dim=512,
               positions=10):
    super(Decoder, self).__init__()
    positions += 1 # plus 1
    self.encoder_dim = encoder_dim
    self.decoder_dim = decoder_dim
    self.prediction_length = prediction_length

    self.decode_step = nn.LSTMCell(input_size=input_dim,
                                   hidden_size=decoder_dim,
                                   bias=True)
    self.init_h = nn.Linear(positions * 2, decoder_dim)
    self.init_c = nn.Linear(encoder_dim, decoder_dim)
    self.input_encoder = nn.Linear(encoder_dim + decoder_dim, input_dim)
    self.predict_layer = nn.Linear(decoder_dim, 2)

  def init_hidden_state(self, positions, encoder_out):
    h = self.init_h(positions)
    c = self.init_c(encoder_out)
    return h, c

  def forward(self, encoder_out, history_positions):
    batch_size, encoder_dim = encoder_out.shape
    encoder_out = encoder_out.view(batch_size, encoder_dim)

    # TODO How to use encoder to init state
    h, c = self.init_hidden_state(history_positions, encoder_out)

    outputs = []
    for i in range(self.prediction_length):
      input = self.input_encoder(torch.cat([encoder_out, h]))
      h, c = self.decode_step(input, (h, c)) # TODO input
      preds = self.predict_layer(h)
      outputs.append(preds)

    output = torch.concat(outputs, dim=1)
    return output


@register_model
class EncoderDecoder(nn.Module):
  def __init__(self, decoder_params,
               encoder_params):
    super(EncoderDecoder, self).__init__(self)
    self.decoder = Decoder(**decoder_params)
    self.encoder = Encoder(**encoder_params)

  def forward(self, sample):
    image = sample['image']
    n, c, h, w = image.shape
    history_positions = sample['history_positions'].view((n, -1))

    encoder_output = self.encoder(image)
    output = self.decoder(history_positions, encoder_output)

    return output
