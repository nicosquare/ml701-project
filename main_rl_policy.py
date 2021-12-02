import argparse
import os
import pickle
from pathlib import Path

import gym
import gym_chrome_dino
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from network.pg import PG
from utils.show_img import show_img


class GameSession:

    def __init__(
            self, session_env, initial_epsilon=0.1, final_epsilon=0.0001, observe=False,
            steps_to_observe=100, frames_to_action=1, frames_to_anneal=100000, replay_memory_size=50000,
            minibatch_size=16, n_actions=3, gamma=0.99, steps_to_save=1000,
            loss_path='./models/prl/loss.csv', scores_path='./models/prl/scores.csv',
            actions_path='./models/prl/actions.csv', q_values_path='./models/prl/q_values.csv',
    ):

        self.session_env = session_env
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.observe = observe
        self.steps_to_observe = steps_to_observe
        self.frames_to_action = frames_to_action
        self.frames_to_anneal = frames_to_anneal
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = minibatch_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.steps_to_save = steps_to_save
        self.loss_path = loss_path
        self.scores_path = scores_path
        self.actions_path = actions_path
        self.q_values_path = q_values_path

        # Display the processed image on screen using openCV, implemented using python coroutine
        self._display = show_img()
        # Initialize the display coroutine
        self._display.__next__()

        # Initialize the required files
        self.loss_df = pd \
            .read_csv(loss_path) if os.path.isfile(loss_path) else pd.DataFrame(columns=['loss'])
        self.scores_df = pd \
            .read_csv(scores_path) if os.path.isfile(scores_path) else pd.DataFrame(columns=['scores'])
        self.actions_df = pd \
            .read_csv(actions_path) if os.path.isfile(actions_path) else pd.DataFrame(columns=['actions'])
        self.q_values_df = pd \
            .read_csv(q_values_path) if os.path.isfile(q_values_path) else pd.DataFrame(columns=['q_values'])

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

        self.session_env.reset()

        # Build first 4 frames with action Do Nothing
        x_t, r_t, done, _ = self.session_env.step(0)
        x_t = torch.from_numpy(x_t)
        s_t = torch.stack((x_t, x_t, x_t, x_t))
        s_t = torch.reshape(s_t, (1, s_t.shape[0], s_t.shape[1], s_t.shape[2]))

        initial_state = s_t
        model = PG()
        model.build_model(n_stacked_frames=s_t.shape[1], n_actions=self.n_actions, learning_rate=1e-3)
        model_optim = torch.optim.Adam(model.parameters(), lr=4e-3)
        # no_iteration = 10
        # for i in range(no_iteration):
        #     self.run_complete_game()
        no_iterations = 100
        for i in range(no_iterations):
            print('iteration: ', i)
            log_probs, rewards, entropy = self.run_complete_game(model, initial_state)
            # print((log_probs))
            # print((rewards))
            # print((entropy))
            # print('-------------')
            # print(torch.sum(torch.stack(log_probs, 0), 0))

            total_rewards = np.sum(rewards)
            # print(torch.tensor(total_rewards))
            # print(-torch.mean(torch.sum(torch.stack(log_probs, 0), 0) * torch.tensor(total_rewards)))
            # model_loss = -torch.mean(torch.sum(torch.stack(log_probs, 0), 0) * torch.Tensor(total_rewards))
            model_loss = torch.sum(torch.stack(log_probs, 0), 0) * torch.tensor(total_rewards)
            model_loss = -torch.mean(model_loss)
            print(model_loss)
            loss.append(float(model_loss))
            reward_history.append(total_rewards)
            entropy_history.append(entropy)
            model_optim.zero_grad()
            model_loss.backward()
            model_optim.step()

        plt.plot(reward_history)
        # plt.plot(loss)
        print(np.max(reward_history))

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
        print('start game...')
        t = 0
        while not done:
            if t > 0:
                probs = model.predict(s_t)
                # print('probs', probs)
                m = torch.distributions.Categorical(probs)
                print('Prob:', probs)
                entropy = m.entropy().detach().numpy()
                action = m.sample()
                log_prob = m.log_prob(action)
                print('Action:', action)

                x_t, r_t, done, _ = self.session_env.step(action)

                rewards.append(r_t)
                log_probs.append(log_prob)
                entropies.append(entropy)
                t = 0
            print(t)
            t += 1

        print('end game...')

        return log_probs, rewards, entropies

    @staticmethod
    def save_obj(obj, name):
        with open('models/rl_policy/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        try:
            with open('models/rl_policy/' + name + '.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            raise e

    @staticmethod
    def cache_file_exists(name):
        return os.path.exists('models/rl_policy/' + name + '.pkl')

    @staticmethod
    def create_cache_file(name):
        print('Creating {}.pkl file'.format(name))
        open('models/rl_policy/' + name + '.pkl', 'w+').close()


def create_required_folders():
    Path("models/rl_policy").mkdir(parents=True, exist_ok=True)


"""
    Main method definition
"""
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--InitialEpsilon", help="Initial epsilon")
parser.add_argument("-f", "--FinalEpsilon", help="Final epsilon")
parser.add_argument("-s", "--StepsToSave", help="Steps to save")
parser.add_argument("-o", "--Observe", help="If used, no training is done, just playing", action='store_true')
parser.add_argument("-n", "--NoBrowser", help="Run without UI", action='store_true')

# Read arguments from command line
args = parser.parse_args()

if __name__ == '__main__':

    # Guarantee the creation of required folders
    create_required_folders()

    if not args.NoBrowser:
        env = gym.make('ChromeDino-v0')
    else:
        env = gym.make('ChromeDinoNoBrowser-v0')

    INITIAL_EPSILON = float(args.InitialEpsilon) if args.InitialEpsilon else 0.1
    FINAL_EPSILON = float(args.FinalEpsilon) if args.FinalEpsilon else 0.0001
    STEPS_TO_SAVE = float(args.StepsToSave) if args.StepsToSave else 1000
    OBSERVE = args.Observe

    game_session = GameSession(
        session_env=env, initial_epsilon=INITIAL_EPSILON, final_epsilon=FINAL_EPSILON,
        observe=OBSERVE, steps_to_save=STEPS_TO_SAVE,
    )

    try:
        game_session.train()
    except Exception as e:
        print('Closing environment due to exception')
        env.close()
        raise e
