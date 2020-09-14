import torch
import logging
from tqdm import tqdm

from utils import find_latest_ckpt


class Trainer(object):
  def __init__(self, save_path,
               model,
               optimizer,
               criterion=None,
               max_epoch=100,
               max_steps=100000,
               logger=logging,
               scheduler=None,
               auto_resume=True,
               log_frequence=500,
               save_frequence=1000,
               eval_frequence=0,
               cls_loss_weight=0.1):
    self.init_state()
    self.save_path = save_path
    self.log_frequence = log_frequence
    self.save_frequence = save_frequence
    self.eval_frequence = eval_frequence
    self.auto_resume = auto_resume
    self.logger = logger
    self.max_epoch = max_epoch
    # If max_steps and max_epoch bost set,
    # Consider max_steps first
    self.max_steps = max_steps
    self.cls_loss_weight = cls_loss_weight

    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.criterion = criterion
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.logger.info("Using device %s" % self.device)
    self.model.to(self.device)

    if self.auto_resume:
      self.resume(self.save_path)

  def init_state(self):
    self.epoch = 0
    self.global_steps = 0
    self.finished = False

  def resume(self, path):
    latest_ckpt = find_latest_ckpt(path)
    if not latest_ckpt is None:
      checkpoint = torch.load(latest_ckpt)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.epoch = checkpoint['epoch']
      self.global_steps = checkpoint['steps']
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if self.scheduler is not None:
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      self.logger.info("Restore epoch %d global_steps %d from %s" % 
                       (self.epoch, self.global_steps, latest_ckpt))

  def set_train_dataset(self, ds):
    self.train_dataset = ds

  def set_eval_dataset(self, ds):
    self.eval_dataset = ds

  def eval(self):
    self.logger.info('Start evaluation epoch: %d global_steps: %d' % 
                     (self.epoch, self.global_steps))
    self.model.eval()

    steps = 0
    total_loss = 0.0
    for batch_idx, sample in tqdm(enumerate(self.eval_dataset), position=0, leave=True):
      for k in sample.keys():
        sample[k] = sample[k].to(self.device)

      if self.model.predict_classes:
        outputs, probs = self.model(sample)
      else:
        outputs = self.model(sample) # Accept sample as input
      target_availabilities = sample["target_availabilities"].unsqueeze(-1)
      targets = sample["target_positions"]
      outputs = outputs.reshape(targets.shape)

      loss = torch.square(targets - outputs)
      loss = loss * target_availabilities
      loss = loss.mean().detach_().item()
      total_loss += loss
      steps += 1
    self.logger.info("[Eval] Epoch %d Mean l2 loss: %.3f" % 
                     (self.epoch, 1.0 * total_loss / steps))

  def _train(self):
    self.logger.info('Start training epoch: %d' % self.epoch)
    self.model.train()

    for batch_idx, sample in enumerate(self.train_dataset):
      for k in sample.keys():
        sample[k] = sample[k].to(self.device)
      self.optimizer.zero_grad()

      target_availabilities = sample["target_availabilities"]
      targets = sample["target_positions"]
      if self.model.predict_classes:
        r, c = self.model(sample)
        r = r.reshape(targets.shape)
        loss = self.criterion(r, targets)
        loss = loss * target_availabilities.unsqueeze(-1)
        loss = loss.mean()
        cls_loss = nn.BCELoss(reduction='mean')(c, target_availabilities)
        loss = loss + self.cls_loss_weight * cls_loss
      else:
        outputs = self.model(sample) # Accept sample as input
        outputs = outputs.reshape(targets.shape)
        loss = self.criterion(outputs, targets)
        loss = loss * target_availabilities.unsqueeze(-1)
        loss = loss.mean()
      loss.backward()
      self.optimizer.step()

      if batch_idx > 0 and batch_idx % self.log_frequence == 0:
        self.logger.info("[Train] Epoch [%d] Batch [%d / %d] Loss: %.3f" % 
                         (self.epoch, batch_idx, len(self.train_dataset),
                         loss.detach_().item()))
      self.global_steps += 1
      if self.global_steps > 0 and self.global_steps % self.save_frequence == 0:
        self.save_model()

      if self.global_steps > 0 and self.eval_frequence > 0 \
          and self.global_steps % self.eval_frequence == 0:
        self.eval()

      if self.global_steps > self.max_steps:
        self.finish_train()
        return
      self.scheduler.step()
    self.logger.info("[Train] Epoch %d global_steps %d finished." % 
                     (self.epoch, self.global_steps))

  def finish_train(self):
    self.save_model()
    self.finished = True
    self.logger.info("[Train] Epoch %d global_steps %d finished." % 
                     (self.epoch, self.global_steps))

  def train(self):
    for _ in range(self.epoch, self.max_epoch):
      self._train()
      if self.finished:
        return
      self.epoch += 1

  def save_model(self):
    path = "%s/ckpt.%d.pth" % (self.save_path, self.global_steps)
    self.logger.info("Save model to %s" % path)
    torch.save({
      'model_state_dict' : self.model.state_dict(),
      'optimizer_state_dict' : self.optimizer.state_dict(),
      'scheduler_state_dict' : self.scheduler.state_dict(),
      'epoch' : self.epoch,
      'steps' : self.global_steps,
    }, path)

