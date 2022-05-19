import shutil
from datetime import datetime
from typing import Iterator

import numpy as np
import torch
from pathlib import Path

import tqdm as tqdm
from torch import Tensor
from torch.nn import Parameter
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import evaluation
import util
from architecture import SimpleCNN
from constants import device, loss_fn, tensorboard_root, config_path, out_root, model_path
from dataset import TrainingDataset


def validate_model(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tensor:
    loss = torch.tensor(0., device=device)
    with torch.no_grad():
        for inputs, targets, labels, img_ids, idx in tqdm.tqdm(dataloader, desc='scoring', position=0):
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            predictions = net(inputs)
            loss += loss_fn(predictions, targets)
        loss /= len(dataloader)
    return loss


def main(hyper_params: dict, network_config: dict, eval_settings: dict, full_data_mode: bool = False):
    experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(tensorboard_root / experiment_id))
    shutil.copyfile(config_path, out_root / 'config.json')  # save current config file to results

    training_dataset = TrainingDataset(split="seg_train")
    if full_data_mode:
        train_subset = training_dataset
        val_loader = None
    else:
        train_indices, val_indices = training_dataset.get_train_val_subsets()
        train_subset = Subset(training_dataset, train_indices)
        val_subset = Subset(training_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=0)
    train_loader = DataLoader(train_subset, batch_size=hyper_params['batch_size'], shuffle=True, num_workers=0)

    net = SimpleCNN(**network_config)
    # Save initial model as "best" model (will be overwritten later)
    if not model_path.exists():
        torch.save(net, model_path)
    else:  # if there already exists a model, just load parameters
        print(f'reusing pre-trained model: "{model_path}"')
        net = torch.load(model_path, map_location=torch.device('cpu'))
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params['learning_rate'],
                                 weight_decay=hyper_params['weight_decay'])

    n_updates = hyper_params['n_updates']
    log_stats_at = eval_settings['log_stats_at']
    validate_at = eval_settings['validate_at']
    best_loss = np.inf  # best validation loss so far
    progress_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)
    update = 0  # current update counter

    while update <= n_updates:
        for inputs, targets, labels, img_ids, idx in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            if update % log_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

            if not full_data_mode and update % validate_at == 0 and update > 0:
                # evaluate model on validation set, log parameters and metrics
                val_loss = validate_model(net, val_loader)
                print(f'val_loss: {val_loss}')
                log_validation_params(writer, val_loss, net.parameters(), update)
                if val_loss < best_loss:
                    print(f'{val_loss} < {best_loss}... saving as new {model_path.parts[-1]}')
                    best_loss = val_loss
                    torch.save(net, model_path)
            elif full_data_mode:
                # in eval mode, just compare train_loss
                train_loss = loss.cpu()
                if train_loss < best_loss:
                    print(f'{train_loss} < {best_loss}... saving as new {model_path.parts[-1]}')
                    best_loss = train_loss
                    torch.save(net, model_path)

            # update progress and update-counter
            progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            progress_bar.update()
            update += 1
            if update >= n_updates:
                break

    progress_bar.close()
    print('finished training.')
    print('starting evaluation...')
    evaluation.evaluate_model(hyper_params, network_config, writer)
    print('zipping "results" folder...')
    util.zip_folder(out_root, 'results_' + experiment_id)


def log_validation_params(writer: SummaryWriter, val_loss: Tensor, params: Iterator[Parameter], update: int) -> None:
    writer.add_scalar(tag='validation/loss', scalar_value=val_loss, global_step=update)
    for i, param in enumerate(params):
        writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(), global_step=update)
    for i, param in enumerate(params):
        writer.add_histogram(tag=f'validation/gradients_{i}', values=param.grad.cpu(), global_step=update)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument('-full', help='train on whole training set for maximum test set score', required=False,
                        dest='full_data_mode', action='store_true', default=False)
    args = parser.parse_args()
    full_data_mode_arg = args.full_data_mode
    with open(config_path, 'r') as fh:
        config_args = json.load(fh)
    main(full_data_mode=full_data_mode_arg, **config_args)
