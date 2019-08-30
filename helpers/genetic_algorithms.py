import torch
import random
from helpers.feature_extraction import feature_vector
import cv2 as cv
import numpy as np
from copy import deepcopy
from more_itertools import sort_together


class GameNet(torch.nn.Module):
    def __init__(self):
        super(GameNet, self).__init__()
        self.fc1 = torch.nn.Linear(11, 14)
        self.fc2 = torch.nn.Linear(14, 4)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x.detach()

    def mutate(self, p=.15):
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            for index, weight in enumerate(state_dict[key]):
                if random.uniform(0, 1) <= p:
                    delta = random.uniform(-.5, .5)
                    state_dict[key][index] += delta
        return self.load_state_dict(state_dict)

    def crossover(self, other, p=.3):
        state_dict0 = deepcopy(self.state_dict())
        state_dict1 = deepcopy(other.state_dict())
        keys = list(state_dict1.keys())
        for key in keys:
            if random.uniform(0, 1) <= p:
                x = np.random.randint(1, len(state_dict0[key]))
                state_dict0[key][:x], state_dict1[key][:x] = state_dict1[key][:x].clone(), state_dict0[key][:x].clone()

        child_net0 = GameNet()
        child_net1 = GameNet()
        child_net0.load_state_dict(state_dict0)
        child_net1.load_state_dict(state_dict1)
        return child_net0.cuda(), child_net1.cuda()


def create_population(individuals=1000):
    generation = []
    for i in range(individuals):
        generation.append(GameNet().cuda())
    return generation


def calculate_fitness(individual, env=None):
    observation = env.reset()  # Constructs an instance of the game
    # Controller
    game_controller = env.controller
    # Grid
    grid_object = game_controller.grid
    grid_pixels = grid_object.grid
    # Snake(s)
    snakes_array = game_controller.snakes
    snake_object1 = snakes_array[0]
    to_pixels = lambda x: cv.resize(x, env.grid_size)

    env.render()
    fitness = 0
    features = feature_vector(snake_object1, grid_object.grid, env.grid_size, grid_object.FOOD_COLOR)
    features = torch.cuda.FloatTensor(features)
    action = np.argmax(individual.forward(features).cpu()).item()

    observation, reward, done, info = env.step(action)
    fitness += reward
    while not done:
        env.render()
        features = torch.cuda.FloatTensor(
            feature_vector(snake_object1, observation, env.grid_size, grid_object.FOOD_COLOR))
        action = np.argmax(individual.forward(features).cpu()).item()
        observation, reward, done, info = env.step(action)
        fitness += reward
    return fitness + 1


def create_mating_pool(fitness, population, to_choose):
    cumulative_fitness = sum(fitness)
    fitness = list(map(lambda x: x / cumulative_fitness, fitness))
    fitness, population = sort_together([fitness, population], reverse=True)
    mating_pool = []
    indexes = []
    for i in range(to_choose):
        p = np.random.uniform(0, 1)
        for j in range(len(fitness)):
            if sum(fitness[:j + 1]) >= p:
                if j not in indexes:
                    mating_pool.append(population[j])
                    indexes.append(j)
        while len(mating_pool) <= i:
            index_to_append = random.randint(0, len(fitness) - 1)
            if index_to_append not in indexes:
                indexes.append(index_to_append)
                mating_pool.append(population[index_to_append])

    return mating_pool
