import gym
import copy
import torch
import gym_chrome_dino
from typing import Tuple
import numpy as np

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
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()


def generation(gen_env, old_population, new_population, p_mutation, p_crossover, p_inversion=None):
    for i in range(0, len(old_population) - 1, 2):
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


"""
    Main method definition
"""

if __name__ == '__main__':
    env = gym.make('ChromeDinoGA-v0')

    POPULATION_SIZE = 100  # This value should be pair
    MAX_GENERATION = 20
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8

    p = Population(MLPIndividual(7, 10, 3), POPULATION_SIZE, MAX_GENERATION, MUTATION_RATE, CROSSOVER_RATE, None)
    p.run(env, generation, verbose=True, output_folder='./models/ga_dino', log=True)

    env.close()
