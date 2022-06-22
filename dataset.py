# -*- coding: utf-8 -*-
import csv
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as TF

import util
from constants import data_root, scene_classes, class_mapping, images_width, images_height


def create_dataset_file(data_folder: Path, dataset_file: Path):
    print(f'creating dataset file <{dataset_file}>')
    images = list(data_folder.rglob("*.jpg"))
    ids = [int(img.stem) for img in images]
    labels = [img.parent.stem for img in images]
    invalid_size_file = data_folder / 'invalid.csv'
    with open(dataset_file, 'w') as f_dataset, open(invalid_size_file, 'w') as f_invalid:
        dataset_writer = csv.writer(f_dataset)
        dataset_writer.writerow(['id', 'label', 'path'])
        invalid_writer = csv.writer(f_invalid)
        invalid_writer.writerow(['id', 'label', 'path', 'width', 'height'])
        for img_id, label, path in zip(ids, labels, images):
            width, height = util.get_image_shape(path)
            if (width, height) == (images_width, images_height):
                dataset_writer.writerow([img_id, label, path])
            else:
                invalid_writer.writerow([img_id, label, path, width, height])


def read_dataset_file(dataset_file: Path) -> Dict:
    data = {'ids': [], 'labels': [], 'paths': []}
    with open(dataset_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['ids'].append(int(row['id']))
            data['labels'].append(row['label'])
            data['paths'].append(row['path'])
    return data


class SceneDataset(Dataset):
    def __init__(self, split: str):
        data_folder = data_root / split
        dataset_file = data_folder / 'dataset.csv'
        if not dataset_file.exists():
            create_dataset_file(data_folder, dataset_file)
        data = read_dataset_file(dataset_file)
        print(f'loaded dataset from <{dataset_file}>')
        self.images = data['paths']
        self.labels = data['labels']
        self.ids = data['ids']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.ids[idx], idx

    def get_train_val_subsets(self, train_split_size: int = 0.8) -> Tuple[List, List]:
        train_indices, val_indices = [], []
        for cls in scene_classes:
            cls_indices = [i for i, lbl in enumerate(self.labels) if lbl == cls]
            split_point = int(len(cls_indices) * train_split_size)
            train_indices += cls_indices[:split_point]
            val_indices += cls_indices[split_point:]
        print(f'number of training images: {len(train_indices)}')
        print(f'number of validation images: {len(val_indices)}')
        return train_indices, val_indices


def _label_to_one_hot(label: str) -> Tensor:
    vec = torch.zeros(size=(len(scene_classes),))
    vec[class_mapping[label]] = 1
    return vec


class TrainingDataset(SceneDataset):
    random_pil_augmentations = TF.Compose([TF.RandomHorizontalFlip()])
    pil_to_tensor = TF.Compose([TF.ToTensor()])

    def __init__(self, split: str, augment=False):
        super(TrainingDataset, self).__init__(split=split)
        self.augment = augment

    def __getitem__(self, idx):
        img_path, label, img_id, _ = super(TrainingDataset, self).__getitem__(idx)
        pil_image = Image.open(img_path)
        if self.augment:
            pil_image = self.random_pil_augmentations(pil_image)
        image: Tensor = self.pil_to_tensor(pil_image)
        image[:] -= torch.mean(image)
        image[:] /= torch.std(image)
        target_vector = _label_to_one_hot(label)
        return image, target_vector, label, img_id, idx
