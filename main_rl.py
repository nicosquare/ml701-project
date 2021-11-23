import gym
import gym_chrome_dino
import argparse
import numpy as np

from utils.show_img import show_img


class GameSession:

    def __init__(
            self, session_env, n_episodes=100
    ):

        self.session_env = session_env
        self.n_episodes = n_episodes
        # display the processed image on screen using openCV, implemented using python coroutine
        self._display = show_img()
        # initialize the display coroutine
        self._display.__next__()

    def run_episode(self):

        obs = self.session_env.reset()
        done = False
        reward = 0

        while not done:

            obs, reward, done, _ = env.step(np.random.randint(0, 3))
            self._display.send(obs)

            print(reward)

            if done:
                break

        env.close()

    def run_complete_game(self):

        for episode in range(self.n_episodes):
            self.run_episode()


"""
    Main method definition
"""
parser = argparse.ArgumentParser()

# Read arguments from command line
args = parser.parse_args()

if __name__ == '__main__':

    env = gym.make('ChromeDinoNoBrowser-v0')

    done = False

    while not done:

        obs, reward, done, info = env.step(np.random.randint(0, 2))
        print(obs)

    env.close()
