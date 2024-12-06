from typing import List


class Params:
    def __init__(self, w=128, h=128, bt=2, n_epoch=2000, batch_size=24, n_layers=3, b=2, in_channels: List = None,
                 kernel_sizes: List = None, activation='relu'):
        self.w, self.h = w, h
        self.bt = bt
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.in_channels = in_channels.copy() if in_channels is not None else [128, 32, 16]
        self.n_layers = n_layers
        self.b = b
        self.activation = activation
        self.kernel_sizes = kernel_sizes.copy() if kernel_sizes is not None else [7, 5, 3]
