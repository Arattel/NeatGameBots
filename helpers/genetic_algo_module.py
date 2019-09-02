import torch
from helpers.feature_extraction import feature_vector
import cv2 as cv
import numpy as np
import neat
import os

import gym
import gym_snake
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import torch
import random
from helpers.feature_extraction import feature_vector, find_apple_coordinates_pixel_array, distances_to_walls
from helpers.genetic_algorithms import GameNet, create_population, calculate_fitness, selection_boltzmann

# Construct Environment
env = gym.make('snake-v0')
env.grid_size = 15, 15
observation = env.reset()  # Constructs an instance of the game

# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]

observation = env.reset()
to_pixels = lambda x: cv.resize(x, env.grid_size)


def eval_genomes(individuals, config):
    global env
    nets = []
    ge = []
    for genome_id, genome in individuals:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
    to_pixels = lambda x: cv.resize(x, env.grid_size)

    for x, net in enumerate(nets):
        observation = env.reset()  # Constructs an instance of the game
        # Controller
        game_controller = env.controller
        # Grid
        grid_object = game_controller.grid
        grid_pixels = grid_object.grid
        # Snake(s)
        snakes_array = game_controller.snakes
        snake_object1 = snakes_array[0]
        fitness = 0
        features = feature_vector(snake_object1, grid_object.grid, env.grid_size, grid_object.FOOD_COLOR)
        output = nets[x].activate(features)
        action = np.argmax(output).item()
        print(type(action))
        observation, reward, done, info = env.step(action)
        fitness += reward
        while not done:
            env.render()
            features = feature_vector(snake_object1, observation, env.grid_size, grid_object.FOOD_COLOR)
            output = nets[x].activate(features)
            action = np.argmax(output).item()
            observation, reward, done, info = env.step(action)
            fitness += reward
        print(fitness)
        ge[x].fitness = fitness


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-lDefaultGenomeevel object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '../config-feedforward.txt')
    run(config_path)
