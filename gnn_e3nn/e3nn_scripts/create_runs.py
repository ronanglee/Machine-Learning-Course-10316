import yaml
import os
import pandas as pd

from ase.atoms import Atoms
from ase.io import read
# from ase.db import connect

hyperparameters_1 = {
    'r_max': [3, 4, 5, 6],
    'batch_size': [10, 20, 30, 40, 50],
    'learning_rate': [0.02, 0.015, 0.01, 0.005, 0.003, 0.001],
}

num_features = [64, 128]
l_max = [2, 3, 4]

for v in hyperparameters_1['learning_rate']:
    os.makedirs(f'{k}/{k}_{i}', exist_ok=True)
    config['root'] = '.'
    config['wandb_project'] = f'{k}'
    config['run_name'] = f'{k}_{i}'
    config[k] = i
    with open(f'{k}/config_{k}_{i}.yaml', 'w') as file:
        yaml.dump(config, file)

