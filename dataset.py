import os


from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer


def get_dataset(config, name='train'):
  dir_input = DIR_INPUT['path']
  # set env variable for data
  os.environ["L5KIT_DATA_FOLDER"] = dir_input
  dm = LocalDataManager(None)
  rasterizer = build_rasterizer(config, dm)

  if name == 'train':
    # Train dataset/dataloader
    train_zarr = ChunkedDataset(dm.require(config['train_data_loader']["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=config['train_data_loader']["shuffle"],
                                  batch_size=config['train_data_loader']["batch_size"],
                                  num_workers=config['train_data_loader']["num_workers"])
    return train_dataloader
  elif name == 'valid':
    # Valid dataset/dataloader
    valid_zarr = ChunkedDataset(dm.require(config['validate_data_loader']["key"])).open()
    valid_dataset = AgentDataset(cfg, valid_zarr, rasterizer)
    valid_dataloader = DataLoader(valid_dataset,
                                  shuffle=config['validate_data_loader']["shuffle"],
                                  batch_size=config['validate_data_loader']["batch_size"],
                                  num_workers=config['validate_data_loader']["num_workers"])
    return valid_dataloader
  else:
    # Test dataset/dataloader
    test_zarr = ChunkedDataset(dm.require(config['test_data_loader']["key"])).open()
    test_mask = np.load(f"{dir_input}/scenes/mask.npz")["arr_0"]
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,
                                 shuffle=config['test_data_loader']["shuffle"],
                                 batch_size=config['test_data_loader']["batch_size"],
                                 num_workers=config['test_data_loader']["num_workers"])
    return test_dataloader

