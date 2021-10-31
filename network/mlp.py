from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from network.base_nn import NeuralNetwork


class MLPTorch(nn.Module, NeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, p=0.1):
        super(MLPTorch, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # self.dropout = nn.Dropout(p=p)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        output = torch.relu(self.linear1(x))
        # output = self.dropout(output)
        output = self.sigmoid(output)
        output = torch.relu(self.linear2(output))
        return output

    def get_weights_biases(self) -> np.array:
        parameters = self.state_dict().values()
        parameters = [p.flatten() for p in parameters]
        parameters = torch.cat(parameters, 0)
        return parameters.detach().numpy()

    def update_weights_biases(self, weights_biases: np.array) -> None:
        weights_biases = torch.from_numpy(weights_biases)
        shapes = [x.shape for x in self.state_dict().values()]
        shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

        partial_split = weights_biases.split(shapes_prod)
        model_weights_biases = []
        for i in range(len(shapes)):
            model_weights_biases.append(partial_split[i].view(shapes[i]))
        state_dict = OrderedDict(zip(self.state_dict().keys(), model_weights_biases))
        self.load_state_dict(state_dict)
