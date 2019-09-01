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


def selection_best_percentage(fitness, population, to_choose):
    fitness, population = sort_together([fitness, population], reverse=True)
    mating_pool = population[:to_choose]
    return mating_pool


def selection_roulette_(items, fitness, n):
    total = float(sum(fitness))
    i = 0
    w, v = fitness[0], items[0]
    while n:
        x = total * (1 - np.random.random() ** (1.0 / n))
        total -= x
        while x > w:
            x -= w
            i += 1
            w, v = fitness[i], items[i]
        w -= x
        yield v
        n -= 1


def selection_roulette(items, fitness, n):
    return list(selection_roulette_(items, fitness, n))


def selection_tournament_(items, fitness, n, tsize=5):
    zipped = zip(items, fitness)
    for i in range(n):
        candidates = random.sample(items, tsize)
        candidate = max(candidates, key=lambda x: x[1])
        yield candidate[0]


def selection_tournament(items, fitness, n, tsize=5):
    return list(selection_tournament_(items, fitness, n, tsize=tsize))


def selection_rank(items, fitness, n):
    items, fitness = sort_together([fitness, items], reverse=True)
    rank = lambda i, l: l - i
    l = len(fitness)
    probabilities = [rank(i, l) / (l * (l - 1)) for i in range(l)]
    return np.random.choice(np.array(items), n, replace=False, p=probabilities)


def selection_boltzmann(items, fitness, n_to_select, current_generation, max_generations, alpha=.01, basic_temp=21):
    f_max = max(fitness)
    current_generation += 1
    k = 1 + 100 * current_generation / max_generations
    temperature = basic_temp * (1 - alpha) ** k
    probabilities = [np.exp(-(f_max - fitness[i]) / temperature) for i in range(len(fitness))]
    return np.random.choice(np.array(items), n_to_select, replace=False, p=probabilities)
