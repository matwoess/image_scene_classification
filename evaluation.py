# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Tuple, List

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants import device, loss_fn, model_path, out_root, scene_classes, evaluate_on_training_set
from dataset import TrainingDataset
from util import write_dict_to_file


def evaluate_model(hyper_params: dict, network_params: dict, writer: SummaryWriter):
    @torch.no_grad()
    def evaluate(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, dict, np.ndarray]:
        loss = torch.tensor(0., device=device)
        all_targets, all_predictions = [], []
        for inputs, targets, labels, img_ids, idx in tqdm(dataloader, desc='evaluating', position=0):
            all_targets.append(targets.numpy())
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            predictions = net(inputs)
            loss += loss_fn(predictions, targets)
            detach_predictions = predictions.detach().cpu().numpy()
            all_predictions.append(detach_predictions)
        loss /= len(dataloader)
        # compute final metrics
        metrics = calculate_metrics(all_targets, all_predictions)
        cm = calculate_confusion_matrix(all_targets, all_predictions)
        return loss, metrics, cm

    def log_params(metrics):
        # combine parameter dictionaries into a single one
        all_params = hyper_params.copy()
        all_params.update(network_params)
        # log hyper-parameters and metrics to tensorboard
        writer.add_hparams(all_params, metrics)

    net = torch.load(model_path, map_location='cpu').to(device)

    if evaluate_on_training_set:
        train_dataset = TrainingDataset(split='seg_train')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
        train_loss, train_metrics, train_cm = evaluate(train_loader)
        write_dict_to_file(train_metrics, out_root / 'train_metrics.txt')
        plot_confusion_matrix(train_cm, train_metrics['accuracy'], img_path=out_root / 'train_cm.png')

    test_dataset = TrainingDataset(split='seg_test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loss, test_metrics, test_cm = evaluate(test_loader)
    write_dict_to_file(test_metrics, out_root / 'test_metrics.txt')
    plot_confusion_matrix(test_cm, test_metrics['accuracy'], img_path=out_root / 'test_cm.png')

    log_params(test_metrics)
    print(f'final test accuracy: {test_metrics["accuracy"]}')
    print(f'final test top 2 accuracy: {test_metrics["top_2_accuracy"]}')
    print(f'final test macro F-score: {test_metrics["macro_f_score"]}')


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


def calculate_confusion_matrix(targets_list: List[np.ndarray], predictions_list: List[np.ndarray]) -> np.ndarray:
    targets = np.argmax(np.concatenate(targets_list), axis=1)
    predictions = np.argmax(np.concatenate(predictions_list), axis=1)
    return confusion_matrix(targets, predictions)


def plot_confusion_matrix(cm: np.ndarray, accuracy: float, img_path: Path):
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = sns.color_palette("flare", as_cmap=True)
    sns.heatmap(cm, ax=ax, annot=True, fmt=".0f", cmap=cmap, xticklabels=scene_classes, yticklabels=scene_classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    fig.suptitle('Confusion Matrix (accuracy score: {:.3f})'.format(accuracy))
    fig.savefig(img_path)
