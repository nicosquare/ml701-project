import os
import pickle
import random
import sys
from pathlib import Path
from collections import deque

import gym
import gym_chrome_dino  # This should be kept as it is registering the custom environments
import argparse
import torch
import pandas as pd
from IPython.display import clear_output
from time import time

from utils.show_img import show_img
from network.dqn import DQN as MyDQN

import wandb

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback


class GameSession:

    def __init__(
            self, session_env, initial_epsilon=0.1, final_epsilon=0.0001, observe=False, model_id=1,
            steps_to_observe=100, frames_to_action=1, frames_to_anneal=1000000, replay_memory_size=5000,
            minibatch_size=16, n_actions=3, gamma=0.99, steps_to_save=1000, learning_rate=1e-3,
            loss_path='./models/rl/dqn/loss.csv', scores_path='./models/rl/dqn/scores.csv',
            actions_path='./models/rl/dqn/actions.csv',
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
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.loss_path = './models/rl/dqn/loss_epsilon_i_{}_epsilon_f_{}_batch_{}.csv' \
            .format(initial_epsilon, final_epsilon, minibatch_size)
        self.scores_path = './models/rl/dqn/scores_epsilon_i_{}_epsilon_f_{}_batch_{}.csv' \
            .format(initial_epsilon, final_epsilon, minibatch_size)
        self.actions_path = './models/rl/dqn/actions_epsilon_i_{}_epsilon_f_{}_batch_{}.csv' \
            .format(initial_epsilon, final_epsilon, minibatch_size)
        self.model_path = './models/rl/dqn/model_epsilon_i_{}_epsilon_f_{}_batch_{}.pt' \
            .format(initial_epsilon, final_epsilon, minibatch_size)

        wandb.config = {
            "initial_epsilon": initial_epsilon,
            "final_epsilon": final_epsilon,
            "observe": observe,
            "steps_to_observe": steps_to_observe,
            "frames_to_action": frames_to_action,
            "frames_to_anneal": frames_to_anneal,
            "replay_memory_size": replay_memory_size,
            "minibatch_size": minibatch_size,
            "n_actions": n_actions,
            "gamma": gamma,
            "steps_to_save": steps_to_save
        }

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

        # Initialize Cache

        self.initialize_cache()

    def run_complete_game(self):

        model = MyDQN(model_path=self.model_path)
        last_time = time()
        replay_memory = self.load_obj('replay_memory')
        self.session_env.reset()

        # Hooks into the model to collect gradients and topology
        wandb.watch(model)

        # Build first 4 frames with action Do Nothing

        x_t, r_t, done, _ = self.session_env.step(0)
        x_t = torch.from_numpy(x_t)
        s_t = torch.stack((x_t, x_t, x_t, x_t))
        s_t = torch.reshape(s_t, (1, s_t.shape[0], s_t.shape[1], s_t.shape[2]))

        model.build_model(
            n_stacked_frames=s_t.shape[1], n_actions=self.n_actions, learning_rate=self.learning_rate,
            model_id=self.model_id
        )

        # Save initial state for resetting the terminal state
        initial_state = s_t

        if self.observe:
            self.steps_to_observe = float('inf')
            epsilon = self.final_epsilon
            model.load_model()
        else:
            epsilon = self.load_obj('epsilon')
            model.load_model(training=True)

        # Initialize time

        t = self.load_obj('time')

        while True:

            loss = 0
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

            x_t, r_t, done, info = self.session_env.step(a_t)
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

                minibatch_loss = model.train_on_batch(batch=minibatch, gamma=self.gamma)
                loss += minibatch_loss

                # Update the model files

                self.loss_df.loc[len(self.loss_df)] = loss
                self.actions_df.loc[len(self.actions_df)] = float(a_t)

            # Handle the terminal state

            if done:
                s_t = initial_state
                self.scores_df.loc[len(self.scores_df)] = info["score"]
                wandb.log({"scores": info["score"]})
            else:
                s_t = s_t1

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

            wandb.log({"loss": loss})
            wandb.log({"epsilon": epsilon})
            wandb.log({"action": a_t})

            print('Step: {}, State: {}, epsilon: {}, action: {}, reward: {}, loss: {}'.format(
                t, state, epsilon, a_t, r_t, loss
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

    def save_obj(self, obj, name):

        file_name = 'models/rl/dqn/' + name + '_i_{}_epsilon_f_{}_batch_{}.pkl' \
            .format(self.initial_epsilon, self.final_epsilon, self.minibatch_size)

        with open(file_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):

        file_name = 'models/rl/dqn/' + name + '_i_{}_epsilon_f_{}_batch_{}.pkl' \
            .format(self.initial_epsilon, self.final_epsilon, self.minibatch_size)

        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as exc:
            raise exc

    def cache_file_exists(self, name):

        file_name = 'models/rl/dqn/' + name + '_i_{}_epsilon_f_{}_batch_{}.pkl' \
            .format(self.initial_epsilon, self.final_epsilon, self.minibatch_size)

        return os.path.exists(file_name)

    def create_cache_file(self, name):

        file_name = 'models/rl/dqn/' + name + '_i_{}_epsilon_f_{}_batch_{}.pkl' \
            .format(self.initial_epsilon, self.final_epsilon, self.minibatch_size)

        print('Creating {} file'.format(file_name))
        open(file_name, 'w+').close()


def create_required_folders():
    Path("models/rl/dqn").mkdir(parents=True, exist_ok=True)


"""
    Main method definition
"""
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--InitialEpsilon", help="Initial epsilon")
parser.add_argument("-f", "--FinalEpsilon", help="Final epsilon")
parser.add_argument("-s", "--StepsToSave", help="Steps to save")
parser.add_argument("-m", "--MiniBatch", help="Mini batch size")
parser.add_argument("-r", "--Reward", help="Game time reward")
parser.add_argument("-p", "--Penalty", help="Game over penalty")
parser.add_argument("-l", "--LearningRate", help="Learning rate of the NN")
parser.add_argument("-mid", "--ModelId", help="Id of model to use in the CNN")
parser.add_argument("-o", "--Observe", help="If used, no training is done, just playing", action='store_true')
parser.add_argument("-n", "--NoBrowser", help="Run without UI", action='store_true')
parser.add_argument("-sb", "--StableBaselines", help="Run Stable Baselines DQN", action='store_true')

# Read arguments from command line
args = parser.parse_args()

if __name__ == '__main__':

    wandb.init(project="dqn-images", entity="madog")

    # Guarantee the creation of required folders
    create_required_folders()

    INITIAL_EPSILON = float(args.InitialEpsilon) if args.InitialEpsilon else 0.1
    FINAL_EPSILON = float(args.FinalEpsilon) if args.FinalEpsilon else 0.0001
    STEPS_TO_SAVE = int(args.StepsToSave) if args.StepsToSave else 1000
    MINIBATCH_SIZE = int(args.MiniBatch) if args.MiniBatch else 16
    REWARD = float(args.Reward) if args.Reward else 0.1
    PENALTY = float(args.Penalty) if args.Penalty else -1.0
    LEARNING_RATE = float(args.LearningRate) if args.LearningRate else 1e-4
    MODEL_ID = int(args.ModelId) if args.ModelId else 1
    OBSERVE = args.Observe
    SB = args.StableBaselines

    if not SB:

        if not args.NoBrowser:
            env = gym.make('ChromeDino-v0')
        else:
            env = gym.make('ChromeDinoNoBrowser-v0')

        game_session = GameSession(
            n_actions=2, frames_to_anneal=1000000, replay_memory_size=50000, learning_rate=LEARNING_RATE,
            session_env=env, initial_epsilon=INITIAL_EPSILON, final_epsilon=FINAL_EPSILON,
            observe=OBSERVE, steps_to_save=STEPS_TO_SAVE, minibatch_size=MINIBATCH_SIZE,
            model_id=MODEL_ID
        )

        env.set_acceleration(False)

        try:
            game_session.run_complete_game()
        except Exception as e:
            print('Closing environment due to exception')
            env.close()
            raise e

    else:

        if not OBSERVE:

            def make_env():

                if not args.NoBrowser:
                    return Monitor(gym.make('ChromeDinoNotNorm-v0'))
                else:
                    return Monitor(gym.make('ChromeDinoNotNormNoBrowser-v0'))


            run = wandb.init(
                project='dqn-images-sb',
                entity="madog",
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )

            model = DQN(
                policy="CnnPolicy",
                env=make_env(),
                verbose=1,
                learning_rate=LEARNING_RATE,
                learning_starts=100,
                batch_size=MINIBATCH_SIZE,
                device='cpu'
            )

            model.learn(
                total_timesteps=1000000,
                n_eval_episodes=10,
                log_interval=4,
                callback=WandbCallback(
                    model_save_freq=100,
                    verbose=1,
                    gradient_save_freq=10,
                    model_save_path=f"models/rl/dqn_wo_img/images/{run.id}"
                )
            )

            model.save(f"models/rl/dqn_sb/images/dqn_dino_{run.id}")

            run.finish()

        else:

            # Check if we are running python 3.8+
            # we need to patch saved model under python 3.6/3.7 to load them
            newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

            custom_objects = {}
            if newer_python_version:
                custom_objects = {
                    "learning_rate": 0.0,
                    "lr_schedule": lambda _: 0.0,
                    "clip_range": lambda _: 0.0,
                }

            model = DQN.load(path="./best_models/dino_dqn_img", custom_objects=custom_objects)

            env = DummyVecEnv([lambda: gym.make('ChromeDinoRLPo-v0')])
            env.training = False
            obs = env.reset()

            while True:

                try:

                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    if done:
                        obs = env.reset()

                except Exception as e:
                    print('Closing environment due to exception')
                    env.close()
                    raise e
