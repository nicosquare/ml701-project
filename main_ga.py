import traceback
from pathlib import Path

import gym
import copy
import torch
import gym_chrome_dino
from typing import Tuple
import numpy as np
import pandas as pd
import argparse

from ga.individual import roulette_wheel_selection, crossover, mutation, Individual
from ga.population import Population
from network.base_nn import NeuralNetwork
from network.mlp import MLPTorch


# author: Nicolas
class MLPIndividual(Individual):

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return MLPTorch(input_size, hidden_size, output_size)

    def run_single(self, ind_env, n_episodes=100, render=False) -> Tuple[float, np.array]:
        obs = ind_env.reset()
        done = False
        fitness = 0
        while not done:

            if render:
                ind_env.render()
            obs = torch.from_numpy(obs).float()

            action = self.nn.forward(obs)
            obs, reward, done, _ = ind_env.step(torch.argmax(action))

            fitness = reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()


def generation(gen_env, old_population, new_population, p_mutation, p_crossover, p_inversion=None):
    for i in range(0, len(old_population) - 1, 2):  # Should we keep this?
        # Selection
        parent1 = roulette_wheel_selection(old_population)
        parent2 = roulette_wheel_selection(old_population)

        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 p_crossover)

        # Mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        print("Start: Calculating children fitness")

        child1.calculate_fitness(gen_env)
        child2.calculate_fitness(gen_env)

        print("End: Calculating children fitness")

        # If children fitness is greater thant parents update population
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2


def create_required_folders():
    Path("models/ga").mkdir(parents=True, exist_ok=True)


"""
    Main method definition
"""
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--Population", help="key in number of population.")
parser.add_argument("-g", "--Generation", help="key in max number of generation")
parser.add_argument("-m", "--Mutation", help="key in mutation rate")
parser.add_argument("-c", "--Crossover", help="key in crossover rate")
parser.add_argument("-obs", "--Obstacle", help="number of obstacles to include")
parser.add_argument("-o", "--Observe", help="If used, no training is done, just playing", action='store_true')
parser.add_argument("-n", "--NoBrowser", help="run without UI", action='store_true')

# Read arguments from command line
args = parser.parse_args()

if __name__ == '__main__':

    # Guarantee the creation of required folders
    create_required_folders()

    if args.Obstacle is None or int(args.Obstacle) == 1:
        if not args.NoBrowser:
            env = gym.make('ChromeDinoGAOneObstacle-v0')
        else:
            env = gym.make('ChromeDinoGAOneObstacleNoBrowser-v0')
    elif int(args.Obstacle) == 2:
        if not args.NoBrowser:
            env = gym.make('ChromeDinoGATwoObstacle-v0')
        else:
            env = gym.make('ChromeDinoGATwoObstacleNoBrowser-v0')
    else:
        raise Exception('Just 1 or 2 obstacles are supported')

    env.set_score_mode('normal')

    POPULATION_SIZE = int(args.Population) if args.Population else 30  # This value should be pair
    MAX_GENERATION = int(args.Generation) if args.Generation else 10
    MUTATION_RATE = float(args.Mutation) if args.Mutation else 0.1
    CROSSOVER_RATE = float(args.Crossover) if args.Crossover else 0.8
    OBSERVE = args.Observe

    if not OBSERVE:

        p = Population(lambda: MLPIndividual(len(env.reset()), 10, 3), POPULATION_SIZE, MAX_GENERATION,
                       MUTATION_RATE, CROSSOVER_RATE, None)

        try:
            p.run(env, generation, verbose=True, output_folder='./models/ga/dino', log=True)
        except Exception as e:
            print('Closing environment due to exception')
            env.close()
            raise e

    else:

        score_df = pd.DataFrame(columns=['score'])

        weight_biases = np.load('best_models/dino_ga.npy')
        model = MLPTorch(len(env.reset()), 10, 3)
        model.update_weights_biases(weights_biases=weight_biases)
        model.eval()

        env.training = False
        obs = env.reset()

        for i in range(100):

            done = False

            print(f'Rollout {i}')

            while done is not True:

                try:

                    obs = torch.from_numpy(obs).float()
                    action = model.forward(obs)
                    obs, reward, done, _ = env.step(torch.argmax(action))

                    if done:
                        obs = env.reset()

                except Exception as e:
                    print('Closing environment due to exception')
                    env.close()
                    raise e

            score_df.loc[len(score_df)] = reward

        score_df.to_csv(path_or_buf='models/eval_ga.csv', index=False)
        env.close()
