


config = {
    'model' : {
        'name' : 'ResNetModel',
        'params' : {
            'depth' : 34,
            'input_channels' : 3 + 2 * (config['model_params']['history_num_frames'] + 1),
            'num_outputs' : 2 * config["model_params"]["future_num_frames"]
        }
    },

    'max_epoch' : 100,
    'optimizer' : {
        'type' : 'SGD',
        'lr' : 0.01,
        'momentum' : 0.9,
        'weight_decay' : 1e-4,
    },

    'scheduler' : {
        'type' : 'MultiStepLR',
        'milestones' : [2000, 4000, 6000, 8000],
        'gamma' : 0.1
    }
}


data_config = {
    'format_version': 4,
    'model_params': {
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4
    },

    'valid_data_loader' : {
        'key': 'scenes/validate.zarr',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 4
    }

    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4
    }
    
    'train_params': {
        'max_num_steps': 10000,
        'checkpoint_every_n_steps': 3000,
    }
}
