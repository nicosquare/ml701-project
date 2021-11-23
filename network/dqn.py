from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, Flatten, Linear
from torch.optim import Adam

from network.base_nn import NeuralNetwork


class DQN(nn.Module, NeuralNetwork):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        model = Sequential(OrderedDict([
            ('conv_1', Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), padding=(1, 1), stride=(4, 4))),
            ('pool_1', MaxPool2d(kernel_size=(2, 2))),
            ('activation_1', ReLU()),
            ('conv_2', Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2))),
            ('pool_2', MaxPool2d(kernel_size=(2, 2))),
            ('activation_2', ReLU()),
            ('conv_3', Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2))),
            ('pool_3', MaxPool2d(kernel_size=(2, 2))),
            ('activation_3', ReLU()),

        ]))

    def update_weights_biases(self, weights_biases: np.array) -> None:
        pass

    def get_weights_biases(self) -> np.array:
        pass
