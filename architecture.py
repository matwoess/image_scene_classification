import torch
from torch.nn import Conv2d, ReLU, Sequential, MaxPool2d, Linear, Softmax


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.resize(x.shape[0], x.shape[-2] * x.shape[-1])


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 3, n_hidden_layers: int = 2, n_kernels: int = 32, kernel_size: int = 3,
                 out_features: int = 6):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(SimpleCNN, self).__init__()
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(Conv2d(in_channels=n_in_channels, out_channels=n_kernels,
                              kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2)))
            cnn.append(ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = Sequential(*cnn)
        self.max_pool1 = MaxPool2d((3, 3))
        self.output_layer = Conv2d(in_channels=n_in_channels, out_channels=1,
                                   kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2))
        self.max_pool2 = MaxPool2d((2, 2))
        self.flatten = Flatten()
        self.fnn = Linear(in_features=625, out_features=out_features)
        self.softmax = Softmax(dim=0)

    def forward(self, x):
        x = self.hidden_layers(x)  # (B, C, X, Y) -> (B, K, X, Y)
        x = self.max_pool1(x)  # (B, 1, X, Y) -> (B, K, X/3, Y/3)
        x = self.output_layer(x)  # (B, K, X/3, Y/3) -> (B, 1, X/3, Y/3)
        x = self.max_pool2(x)  # (B, 1, X/3, Y/3) -> (B, K, X/6, Y/6)
        x = self.flatten(x)  # (B, 1, X/6, Y/6) -> (B, X/6*Y/6)
        x = self.fnn(x)  # (B, X*Y/36) -> (B, out_features)
        x = self.softmax(x)  # all predictions between 0 and 1, sum to 1
        return x
