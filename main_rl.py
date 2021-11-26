import os
import pickle
import random
from pathlib import Path
from collections import deque

import gym
import gym_chrome_dino
import argparse
import torch
import pandas as pd
from IPython.display import clear_output
from time import time

from utils.show_img import show_img
from network.dqn import DQN


class GameSession:

    def __init__(
            self, session_env, initial_epsilon=0.1, final_epsilon=0.0001, observe=False,
            steps_to_observe=100, frames_to_action=1, frames_to_anneal=100000, replay_memory_size=50000,
            minibatch_size=16, n_actions=3, gamma=0.99, steps_to_save=1000,
            loss_path='./models/rl/loss.csv', scores_path='./models/rl/scores.csv',
            actions_path='./models/rl/actions.csv', q_values_path='./models/rl/q_values.csv',
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

        # Initialize Cache

        self.initialize_cache()

    def run_complete_game(self):

        model = DQN()
        last_time = time()
        replay_memory = self.load_obj('replay_memory')
        self.session_env.reset()

        # Build first 4 frames with action Do Nothing

        x_t, r_t, done, _ = self.session_env.step(0)
        x_t = torch.from_numpy(x_t)
        s_t = torch.stack((x_t, x_t, x_t, x_t))
        s_t = torch.reshape(s_t, (1, s_t.shape[0], s_t.shape[1], s_t.shape[2]))

        model.build_model(n_stacked_frames=s_t.shape[1], n_actions=self.n_actions, learning_rate=1e-3)

        # Save initial state for resetting the terminal state
        initial_state = s_t

        if self.observe:
            self.steps_to_observe = float('inf')
            epsilon = self.final_epsilon
            model.load_model()
            print('Model loaded successfully for playing')
        else:
            epsilon = self.load_obj('epsilon')
            model.load_model(training=True)

        # Initialize time

        t = self.load_obj('time')

        while True:

            loss = 0
            max_q = 0
            a_t = 0
            r_t = 0

            # Choose an action with Epsilon-Greedy policy

            if t % self.frames_to_action == 0:

                if random.random() <= epsilon:
                    print('|======= RANDOM ACTION =======|')
                    a_t = random.randrange(self.n_actions)
                else:
                    q = model.predict(s_t)
                    a_t = torch.argmax(q)

            # Reduce the epsilon (exploration rate) gradually

            if epsilon > self.final_epsilon and t > self.steps_to_observe:
                epsilon -= (self.initial_epsilon - self.final_epsilon) / self.frames_to_anneal

            # Execute action

            x_t, r_t, done, _ = self.session_env.step(a_t)
            self._display.send(x_t)  # Display the observed image
            print('fps: {0}'.format(1 / (time() - last_time)))  # helpful for measuring frame rate
            x_t = torch.from_numpy(x_t)
            x_t = torch.reshape(x_t, (1, 1, x_t.shape[0], x_t.shape[1]))

            # Append the new image to input stack and remove the first one
            s_t1 = torch.cat((x_t, s_t[:, :3, :, :]), dim=1)

            # Store the transition in Replay Memory
            replay_memory.append((s_t, a_t, r_t, s_t1, done))

            if len(replay_memory) > self.replay_memory_size:
                replay_memory.popleft()

            if t > self.steps_to_observe:
                # Sample a minibatch to train on

                minibatch = random.sample(replay_memory, self.minibatch_size)

                # Experience Replay

                minibatch_loss, minibatch_max_q = model.train_on_batch(batch=minibatch, gamma=self.gamma)
                loss += minibatch_loss
                max_q = minibatch_max_q

                # Update the model files

                self.loss_df.loc[len(self.loss_df)] = loss
                self.q_values_df.loc[len(self.q_values_df)] = minibatch_max_q

            # Reset game to initial frame if we reached a terminal state

            s_t = initial_state if done else s_t1
            t += 1

            # Save progress every defined iterations

            if t % self.steps_to_save == 0:
                # Pause game while saving

                self.session_env.pause()
                model.save_model()
                self.save_obj(replay_memory, 'replay_memory')
                self.save_obj(t, 'time')
                self.save_obj(epsilon, 'epsilon')
                self.loss_df.to_csv(path_or_buf=self.loss_path, index=False)
                self.scores_df.to_csv(path_or_buf=self.scores_path, index=False)
                self.actions_df.to_csv(path_or_buf=self.actions_path, index=False)
                self.q_values_df.to_csv(path_or_buf=self.q_values_path, index=False)
                clear_output()
                self.session_env.resume()

            # Print progress

            state = ""
            if t <= self.steps_to_observe:
                state = "observe"
            elif self.steps_to_observe < t <= self.steps_to_observe + self.frames_to_anneal:
                state = "explore"
            else:
                state = "train"

            print('Step: {}, State: {}, epsilon: {}, action: {}, reward: {}, max Q: {}, loss: {}'.format(
                t, state, epsilon, a_t, r_t, max_q, loss
            ))

    def initialize_cache(self):

        if not self.cache_file_exists('epsilon'):
            self.create_cache_file('epsilon')
            self.save_obj(self.initial_epsilon, "epsilon")

        if not self.cache_file_exists('time'):
            self.create_cache_file('time')
            self.save_obj(0, "time")

        if not self.cache_file_exists('replay_memory'):
            self.create_cache_file('replay_memory')
            self.save_obj(deque(), "replay_memory")

    @staticmethod
    def save_obj(obj, name):
        with open('models/rl/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        try:
            with open('models/rl/' + name + '.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            raise e

    @staticmethod
    def cache_file_exists(name):
        return os.path.exists('models/rl/' + name + '.pkl')

    @staticmethod
    def create_cache_file(name):
        print('Creating {}.pkl file'.format(name))
        open('models/rl/' + name + '.pkl', 'w+').close()


def create_required_folders():
    Path("models/rl").mkdir(parents=True, exist_ok=True)


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
        game_session.run_complete_game()
    except Exception as e:
        print('Closing environment due to exception')
        print(e)
        env.close()
