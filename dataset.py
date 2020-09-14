import os
import argparse
import numpy as np

from torch.utils.data import DataLoader

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

from utils import parse_config


def get_dataset(config, name='train', batch_size=None):
  dir_input = config['path']
  # set env variable for data
  os.environ["L5KIT_DATA_FOLDER"] = dir_input
  dm = LocalDataManager(None)
  rasterizer = build_rasterizer(config, dm)

  if name == 'train':
    # Train dataset/dataloader
    train_zarr = ChunkedDataset(dm.require(config['train_data_loader']["key"])).open()
    train_dataset = AgentDataset(config, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=config['train_data_loader']["shuffle"],
                                  batch_size=config['train_data_loader']["batch_size"] if batch_size is None else batch_size,
                                  num_workers=config['train_data_loader']["num_workers"])
    return train_dataloader
  elif name == 'valid':
    # Valid dataset/dataloader
    valid_zarr = ChunkedDataset(dm.require(config['validate_data_loader']["key"])).open()
    valid_dataset = AgentDataset(config, valid_zarr, rasterizer)
    valid_dataloader = DataLoader(valid_dataset,
                                  shuffle=config['validate_data_loader']["shuffle"],
                                  batch_size=config['validate_data_loader']["batch_size"] if batch_size is None else batch_size,
                                  num_workers=config['validate_data_loader']["num_workers"])
    return valid_dataloader
  elif name == 'sample':
    sample_zarr = ChunkedDataset(dm.require(config['sample_data_loader']["key"])).open()
    sample_dataset = AgentDataset(config, sample_zarr, rasterizer)
    sample_dataloader = DataLoader(sample_dataset,
                                  shuffle=config['sample_data_loader']["shuffle"],
                                  batch_size=config['sample_data_loader']["batch_size"] if batch_size is None else batch_size,
                                  num_workers=config['sample_data_loader']["num_workers"])
  elif name == 'test':
    # Test dataset/dataloader
    test_zarr = ChunkedDataset(dm.require(config['test_data_loader']["key"])).open()
    test_mask = np.load(f"{dir_input}/scenes/mask.npz")["arr_0"]
    test_dataset = AgentDataset(config, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,
                                 shuffle=config['test_data_loader']["shuffle"],
                                 batch_size=config['test_data_loader']["batch_size"] if batch_size is None else batch_size,
                                 num_workers=config['test_data_loader']["num_workers"])
    return test_dataloader
  else:
    raise ValueError("%s not recognized" % name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Lyft kaggle competition dataset generation code')
  parser.add_argument('--config', type=str, help='Config file')
  parser.add_argument('--dir', type=str, help='Save directory')
  parser.add_argument('--log-frequence', type=int, help='Log frequence')
  parser.add_argument('--name', type=str, help='Train, valid, test or sample')
  args = parser.parse_args()

  n = args.name
  _, data_config = parse_config(args.config)
  dataset = get_dataset(data_config, n, batch_size=1)
  length_per_dir = 204800

  for i in range(len(dataset)):
    save_index = i // length_per_dir
    output_path = os.path.join(args.dir, '%d' % save_index)
    if not os.path.isdir(output_path):
      os.mkdir(output_path)
    filename = "%d.npz" % (i % length_per_dir)
    output_filename = os.path.join(output_path, filename)
    if os.path.isfile(output_filename):
      continue

    if i > 0 and i % args.log_frequence == 0:
      print("Save to %s" % output_filename)

    sample = dataset[i]
    save_dict = {}
    for k, v in sample.items():
      v = v.unsqueeze(0).numpy()
      save_dict[k] = v
    np.savez(, **save_dict)
