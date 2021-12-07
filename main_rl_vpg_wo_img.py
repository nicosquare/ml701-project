import csv
import os
import pickle
import random
from pathlib import Path
from collections import deque

import gym
import gym_chrome_dino
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

from network.mlp_pg import MLPTorch
from utils.show_img import show_img


class GameSession:

    def __init__(
            self, session_env
    ):

        self.session_env = session_env

        # Display the processed image on screen using openCV, implemented using python coroutine
        self._display = show_img()
        # Initialize the display coroutine
        self._display.__next__()

    def train(self):

        """
        1. run the game
        2. get the parameters(probs, rewards, entropy)
        3. sum up total rewards
        4. calculate loss
        5. append rewards and entropy in to list
        6. backpropagation

        """
        # model = DQN()
        # model_optim = torch.optim.Adam(model.parameters(), lr=4e-3)
        reward_history = []
        entropy_history = []
        loss = []
        save_iteration = 1
        file_path = './models/rl_pl/reward_history.csv'
        self.session_env.reset()

        s_t, r_t, done, _ = self.session_env.step(0)

        model = MLPTorch(s_t.size, 10, 2)
        model_optim = optim.Adam(model.parameters(), lr=4e-3)

        no_iterations = 1000
        for i in range(no_iterations):
            print('iteration: ', i)
            log_probs, rewards, entropy = self.run_complete_game(model, s_t)
            # print(log_probs)
            # print(rewards)
            total_rewards = np.sum(rewards)
            model_loss = torch.sum(torch.stack(log_probs, 0), 0) * torch.tensor(total_rewards)
            model_loss = -torch.mean(model_loss)
            loss.append(float(model_loss))
            reward_history.append(total_rewards)
            entropy_history.append(entropy)
            model_optim.zero_grad()
            model_loss.backward()
            model_optim.step()

            print(reward_history)
            if no_iterations % save_iteration == 0:
                # open the file in the write mode

                mode = 'a' if os.path.exists(file_path) else 'w+'

                f = open(file_path, mode, newline='')

                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                # for re in rewardHistory:
                writer.writerow(reward_history)

                # close the file
                f.close()
                print('Done writing into file. Clear Reward History...')
                #reward_history = []

        # print(loss)
        # print(entropyHistory)
        plt.figure(0)
        plt.plot(entropy_history)
        plt.figure(1)
        plt.plot(loss)
        plt.figure(2)
        plt.plot(reward_history)
        # plt.plot(loss)
        print(np.max(rewardHistory))

        plt.show()


    def run_complete_game(self, model, initial_state):
        """
        1. initialize variable
        2. run the game:
            2.1 get state
            2.2 predict action
            2.3 take action
            2.4 input reward into list
            2.5 repeat until die
        """

        self.session_env.reset()
        # t = self.load_obj('time')
        # print('......', t)
        # t = self.load_obj('time')
        # print(t)
        done = False
        rewards = []
        log_probs = []
        entropies = []
        s_t = initial_state
        i = 0
        print('start game...', i)

        while not done:
            if i % 10 == 0:
                s_t = s_t.astype(np.float32)
                probs = model.forward(torch.from_numpy(s_t))
                m = torch.distributions.Categorical(probs)
                # print('Prob:', probs)
                entropy = m.entropy().detach().numpy()
                action = m.sample()
                log_prob = m.log_prob(action)
                # print('Action:', action):q
                x_t, r_t, done, _ = self.session_env.step(action)

                rewards.append(r_t)
                log_probs.append(log_prob)
                entropies.append(entropy)
            i += 1

        print('end game...', i)

        return log_probs, rewards, entropies


def create_required_folders():
    Path("models/rl_pl").mkdir(parents=True, exist_ok=True)


"""
    Main method definition
"""

# Read arguments from command line

if __name__ == '__main__':

    # Guarantee the creation of required folders
    create_required_folders()

    if __name__ == '__main__':

        # Guarantee the creation of required folders

        env = gym.make('ChromeDinoRLPo-v0')
        env.set_score_mode('normal')

        game_session = GameSession(
            session_env=env
        )

        try:
            game_session.train()
        except Exception as e:
            print('Closing environment due to exception')
            env.close()
            raise e
