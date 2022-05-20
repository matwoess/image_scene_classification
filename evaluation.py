# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants import device, loss_fn, model_path, out_root
from dataset import TrainingDataset
from util import write_dict_to_file


def evaluate_model(hyper_params: dict, network_params: dict, writer: SummaryWriter):
    def evaluate(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, dict]:
        loss = torch.tensor(0., device=device)
        all_targets, all_predictions = [], []
        with torch.no_grad():
            for inputs, targets, labels, img_ids, idx in tqdm(dataloader, desc='evaluating', position=0):
                all_targets.append(targets.numpy())
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                predictions = net(inputs)
                # accumulate data
                loss += loss_fn(predictions, targets)
                detach_predictions = predictions.detach().cpu().numpy()
                all_predictions.append(detach_predictions)
            loss /= len(dataloader)
            # compute final metrics
            metrics = calculate_metrics(all_targets, all_predictions)
            return loss, metrics

    def log_params(metrics):
        # combine parameter dictionaries into a single one
        all_params = hyper_params.copy()
        all_params.update(network_params)
        # log hyper-parameters and metrics to tensorboard
        writer.add_hparams(all_params, metrics)

    net = torch.load(model_path, map_location='cpu').to(device)
    # create datasets and loaders
    train_dataset = TrainingDataset(split='seg_train')
    test_dataset = TrainingDataset(split='seg_test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # evaluate on all individual files for both train and test set
    train_loss, train_metrics = evaluate(train_loader)
    test_loss, test_metrics = evaluate(test_loader)
    # write results to files and log parameters
    save_losses(train_loss, test_loss)
    save_metrics(train_metrics, test_metrics)
    log_params(test_metrics)
    print(f'final test accuracy: {test_metrics["accuracy"]}')
    print(f'final test top 2 accuracy: {test_metrics["top_2_accuracy"]}')
    print(f'final test macro F-score: {test_metrics["macro_f_score"]}')


def save_losses(train_loss, test_loss):
    losses_file = Path('results') / 'final_losses.txt'
    with open(losses_file, 'w') as f:
        print(f'Losses:', file=f)
        print(f'test set loss: {test_loss}', file=f)
        print(f'train set loss: {train_loss}', file=f)


def calculate_metrics(targets_list: List[np.ndarray], predictions_list: List[np.ndarray]) -> dict:
    # prepare targets and predictions
    targets_array = np.concatenate(targets_list)
    targets = np.argmax(targets_array, axis=1)
    predictions_array = np.concatenate(predictions_list)
    predictions = np.argmax(predictions_array, axis=1)
    # calculate metrics
    accuracy = accuracy_score(targets, predictions)
    top_k_accuracy = top_k_accuracy_score(targets, predictions_array, k=2)
    f_score = f1_score(targets, predictions, average='macro')
    return {'accuracy': accuracy, 'top_2_accuracy': top_k_accuracy, 'macro_f_score': f_score}


def save_metrics(train_metrics, test_metrics):
    metrics_path = out_root
    write_dict_to_file(test_metrics, metrics_path / 'test_metrics.txt')
    write_dict_to_file(train_metrics, metrics_path / 'train_metrics.txt')
