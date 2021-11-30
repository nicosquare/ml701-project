from builtins import object
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, Flatten, Linear, MSELoss, Softmax
from torch.optim import Adam
from torchinfo import summary


class PG(nn.Module):

    def __init__(self, model_path='models/rl/model.pt'):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.model_path = model_path

    def build_model(self, n_stacked_frames, n_actions, learning_rate):

        self.model = Sequential(OrderedDict([
            ('conv_1', Conv2d(in_channels=n_stacked_frames, out_channels=32, kernel_size=(8, 8), padding=(122, 122),
                              stride=(4, 4))),
            ('pool_1', MaxPool2d(kernel_size=(2, 2))),
            ('activation_1', ReLU()),
            ('conv_2', Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), padding=(21, 21), stride=(2, 2))),
            ('pool_2', MaxPool2d(kernel_size=(2, 2))),
            ('activation_2', ReLU()),
            ('conv_3', Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
            ('pool_3', MaxPool2d(kernel_size=(2, 2))),
            ('activation_3', ReLU()),
            ('flatten', Flatten()),
            ('linear_1', Linear(in_features=6400, out_features=512)),
            ('activation_4', ReLU()),
            ('linear_2', Linear(in_features=512,out_features=n_actions)),
            ('activation_5', Softmax())

        ]))

        self.optimizer = Adam(lr=learning_rate, params=self.model.parameters())
        self.criterion = MSELoss()
        self.model.to(self.device)

        self.save_model()

    def print_model(self):
        summary(model=self.model, input_size=(1, 4, 80, 80))

    def predict(self, state):

        state.to(self.device)

        return self.model(state.float())

    def train_on_batch(self, batch, gamma):

        train_loss = 0
        q_sa = None

        for s_t, a_t, r_t, s_t1, done in batch:

            target = self.predict(s_t)  # Predicted Q values
            q_s_t1 = self.predict(s_t1)  # Predicted Q values for the next state
            q_sa = q_s_t1

            if done:
                target[:, a_t] = r_t  # If terminated, only equals to reward
            else:
                target[:, a_t] = r_t + gamma * torch.max(q_s_t1)

            loss = self.criterion(target, q_s_t1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return (train_loss / len(batch)), torch.max(q_sa).item()  # Return normalized loss and max Q values

    def save_model(self):

        print('Saving model')
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.model_path)

    def load_model(self, training=False):

        state = torch.load(self.model_path)

        if training:
            print('Loading model to continue training')
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.model.train()
        else:
            print('Loading model for inference')
            self.model.eval()
