import os

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Sequential, ReLU, Linear, MSELoss
from torch.optim import Adam
from torchinfo import summary


class QNN(nn.Module):

    def __init__(self, model_path='models/rl/model.pt'):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.model_path = model_path
        self.input_size = None

    def build_model(self, input_size, hidden_size, n_actions, learning_rate):

        self.model = Sequential(OrderedDict([
            ('linear_1', Linear(in_features=input_size, out_features=hidden_size)),
            ('activation_1', ReLU()),
            ('linear_2', Linear(in_features=hidden_size, out_features=n_actions)),
            ('activation_2', ReLU()),
        ]))

        self.input_size = input_size
        self.optimizer = Adam(lr=learning_rate, params=self.model.parameters())
        self.criterion = MSELoss()
        self.model.to(self.device)

        # create model file if not present
        if not os.path.isfile(self.model_path):
            self.save_model()

    def print_model(self):
        summary(model=self.model, input_size=self.input_size)

    def predict(self, state):
        return self.model(state.float().to(self.device))

    def train_on_batch(self, batch, gamma):

        target_q_values = torch.tensor([])
        s_t1_q_values = torch.tensor([])

        for s_t, a_t, r_t, s_t1, done in batch:

            q_s_t = self.predict(s_t)  # Predicted Q values
            q_s_t1 = self.predict(s_t1)  # Predicted Q values for the next state
            target = q_s_t.clone()

            if done:
                target[a_t] = r_t  # If terminated, only equals to reward
            else:
                target[a_t] = r_t + gamma * torch.max(q_s_t1)

            target_q_values = torch.cat((target_q_values, target), dim=0)
            s_t1_q_values = torch.cat((s_t1_q_values, q_s_t1), dim=0)

        loss = self.criterion(target_q_values, s_t1_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
            self.model.load_state_dict(state['state_dict'])
            self.model.eval()
