# -*- coding: utf-8 -*-
from pathlib import Path

import torch

config_path = Path('config.json')

scene_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
class_mapping = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
images_width = images_height = 150

data_root = Path('data')
out_root = Path('results')
out_root.mkdir(exist_ok=True, parents=True)
model_path = out_root / 'best_model.pt'
tensorboard_root = Path('tensorboard')
tensorboard_root.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.BCELoss()
