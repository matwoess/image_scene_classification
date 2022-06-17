# -*- coding: utf-8 -*-
import torch
from torch.nn import Conv2d, ReLU, Sequential, MaxPool2d, Linear, Flatten, Dropout

from constants import images_width, image_channels


class SimpleCNN(torch.nn.Module):
    def __init__(self, initial_kernels=16, kernel_size=3, n_hidden_ffn_neurons=2048, out_features=6):
        super(SimpleCNN, self).__init__()
        edge_dim = images_width  # == 'images_height'
        n_in_channels = image_channels
        n_kernels = initial_kernels
        padding = kernel_size // 2
        for i, pooling in enumerate((2, 3, 2, 2)):
            layer = Sequential(
                Conv2d(n_in_channels, n_kernels, kernel_size=kernel_size, padding=padding),
                ReLU(),
                Conv2d(n_kernels, n_kernels, kernel_size=kernel_size, padding=padding),
                ReLU(),
            )
            self.__setattr__(f'cnn_stage{i + 1}', layer)
            self.__setattr__(f'max_pool{i + 1}', MaxPool2d(pooling, pooling))
            edge_dim //= pooling
            n_in_channels = n_kernels
            n_kernels *= 2
        n_kernels //= 2  # last layer did not increase kernel size --- for computation of n_in_features
        self.flatten = Flatten(start_dim=1)
        n_in_features = edge_dim ** 2 * n_kernels
        self.fnn = Sequential(
            Linear(in_features=n_in_features, out_features=n_hidden_ffn_neurons),
            ReLU(),
            Dropout(p=0.5),
        )
        self.out_layer = Linear(in_features=n_hidden_ffn_neurons, out_features=out_features)

    def forward(self, x):
        x = self.cnn_stage1(x)  # (B, C, X, Y) -> (B, K, X, Y)
        x = self.max_pool1(x)  # (B, K, X, Y) -> (B, K, X/2, Y/2)
        x = self.cnn_stage2(x)  # (B, K, X/2, Y/2) -> (B, 2K, X/2, Y/2)
        x = self.max_pool2(x)  # (B, 2K, X/2, Y/2) -> (B, 2K, X/6, Y/6)
        x = self.cnn_stage3(x)  # (B, 2K, X/6, Y/6) -> (B, 4K, X/6, Y/6)
        x = self.max_pool3(x)  # (B, 4K, X/6, Y/6) -> (B, 4K, X//12, Y//12)
        x = self.cnn_stage4(x)  # (B, 4K, X/12, Y/12) -> (B, 8K, X/12, Y/12)
        x = self.max_pool4(x)  # (B, 8K, X/12, Y/12) -> (B, 8K, X/24, Y/24)
        x = self.flatten(x)  # (B, 8K, X/24, Y/24) -> (B, 8K*X/24*Y/24)
        x = self.fnn(x)  # (B, 8K*X/24*Y/24) -> (B, n_hidden_ffn_neurons)
        x = self.out_layer(x)  # (B, n_hidden_ffn_neurons) -> (B, out_features)
        return x
