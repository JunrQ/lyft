

import argparse


from dataset import get_dataset
from trainer import Trainer
from utils import parse_config
from models import *



parser = argparse.ArgumentParser(description='Lyft kaggle competition training code')
parser.add_argument('--config', type=str, help='Config file')
parser.add_argument('--dir', type=str, help='Save directory')
parser.add_argument('--num-workers', type=int, help='Number of workers to read dataset')
parser.add_argument('--log-frequence', type=int, help='Log frequence')
parser.add_argument('--save-frequence', type=int, help='Save frequence')
parser.add_argument('--eval-frequence', type=int, help='Eval frequence')
parser.add_argument('--resume', type=bool, help='Whether to resume')
args = parser.parse_args()


# Set logger
msg = []
logger = logging.getLogger('ImmuneNet')
logger.setLevel(logging.INFO)
if not os.path.isdir(args.dir):
  msg.append('%s not exist, make it' % args.dir)
  os.mkdir(args.dir)
log_file_path = os.path.join(args.dir, 'log.log')
if os.path.isfile(log_file_path):
  target_path = log_file_path + '.%s' % time.strftime("%Y%m%d%H%M%S")
  msg.append('Log file exists, backup to %s' % target_path)
  shutil.move(log_file_path, target_path)
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info(' '.join(sys.argv))
for m in msg:
  logger.warn(m)


config, data_config = parse_config(args.config)
train_ds = get_dataset(data_config, 'train')
valid_ds = get_dataset(data_config, 'valid')

optimizer = get
scheduler = 

model = get_model(config['model']['name'],
                  **model_config['model']['params'])
trainer = Trainer(save_path=args.dir,
                  model=model,
                  optimizer=optimizer,
                  max_epoch=config['max_epoch'],
                  logger=logger,
                  scheduler=scheduler,
                  auto_resume=args.resume,
                  log_frequence=args.log_frequence,
                  save_frequence=args.save_frequence,
                  eval_frequence=args.eval_frequence)

trainer.train()
