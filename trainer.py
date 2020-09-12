import torch
import logging
import tqdm


from utils import find_latest_ckpt


class Trainer(object):
  def __init__(self, save_path,
               model,
               optimizer,
               max_epoch=100,
               logger=logging,
               scheduler=None,
               auto_resume=True,
               log_frequence=500,
               save_frequence=1000,
               eval_frequence=1000):
    self.start_epoch = 0
    self.log_frequence = log_frequence
    self.save_frequence = save_frequence
    self.auto_resume = auto_resume
    self.logger = logger
    self.max_epoch = max_epoch

    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.logger.info("Using device %s" % self.device)
    self.model.to(device)

    if self.auto_resume:
      self.resume(self.save_path)

  def resume(self, path):
    latest_ckpt = find_latest_ckpt(path)
    if not latest_ckpt is None:
      checkpoint = torch.load(latest_ckpt)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.start_epoch = checkpoint['epoch']
      self.global_steps = checkpoint['steps']
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if self.scheduler is not None:
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      self.logger.info("Restore epoch %d from %s" % (self.start_epoch, latest_ckpt))

  def set_train_dataset(self, ds):
    self.train_dataset = ds

  def set_eval_dataset(self, ds):
    self.eval_dataset = ds

  def eval(self):
    self.logger.info('Start evaluation epoch: %d' % self.epoch)
    self.model.eval()

    steps = 0
    total_loss = 0.0
    for batch_idx, sample in tqdm(enumerate(self.train_dataset)):
      for k in sample.keys():
        sample[k].to(self.device)

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
        sample[k].to(self.device)
      self.optimizer.zero_grad()

      outputs = self.model(sample) # Accept sample as input
      target_availabilities = sample["target_availabilities"].unsqueeze(-1)
      targets = sample["target_positions"]
      outputs = outputs.reshape(targets.shape)
      loss = criterion(outputs, targets)
      loss = loss * target_availabilities
      loss = loss.mean()
      loss.backward()
      self.optimizer.step()

      if batch_idx > 0 and batch_idx % self.log_frequence == 0:
        logger.info("[Train] Epoch [%d] Batch [%d / %d] Loss: %.3f % 
                    (self.epoch, batch_idx, len(trainloader), loss))
      self.global_steps += 1
      if self.global_steps > 0 and self.global_steps % self.save_frequence == 0:
        self.save_model()
      self.scheduler.step()
    logger.info("[Train] Epoch %d finished. Acc: %.3f" % (self.epoch, 1.0 * correct / total))

  def train(self):
    for _ in range(self.start_epoch, self.max_epoch):
      self._train()
      self.epoch += 1

  def save_model(self):
    torch.save({
      'model_state_dict' : self.model.state_dict(),
      'optimizer_state_dict' : self.optimizer.state_dict(),
      'scheduler_state_dict' : self.scheduler.state_dict(),
      'epoch' : self.epoch,
      'steps' : self.global_steps,
    }, "%s/ckpt.%d.pth" % (self.save_path, self.global_steps))
