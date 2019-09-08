import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from more_itertools import sort_together

from helpers.feature_extraction import feature_vector
from helpers.GameNet import GameNet


def create_population(individuals=1000):
    generation = []
    for _ in range(individuals):
        generation.append(GameNet().cuda())
    return generation


def calculate_fitness(individual, env=None, save=False, display=False):
    observation = env.reset()  # Constructs an instance of the game
    game_controller = env.controller  # Controller
    grid_object = game_controller.grid  # Grid
    snakes_array = game_controller.snakes  # All snakes in the game
    snake_object = snakes_array[0]  
    
    saves = []
    fitness, steps, eaten_apples = 0, 0, 0
    
    features = feature_vector(snake_object, grid_object.grid, env.grid_size, grid_object.FOOD_COLOR)
    features = torch.cuda.FloatTensor(features)
    action = np.argmax(individual.forward(features).cpu()).item()
    observation, reward, done, info = env.step(action)
    
    steps += 1
    if reward == 1:
        eaten_apples += 1
        fitness += 4
    elif not reward:
        fitness -= 0.25
    else:
        fitness -= 100

    if display:
        # plt.imshow(observation)    
        env.render()
    if save:
        saves.append(observation)

    while not done:
        features = feature_vector(snake_object, grid_object.grid, env.grid_size, grid_object.FOOD_COLOR)
        features = torch.cuda.FloatTensor(features)
        action = np.argmax(individual.forward(features).cpu()).item()
        
        observation, reward, done, info = env.step(action)
        steps += 1

        if reward == 1:
            eaten_apples += 1
            fitness += 4 * eaten_apples ** .5
        elif not reward:
            fitness -= 0.25
        else:
            fitness -= 10 if steps > 15 else 100

        if display:
            plt.pause(.5)
            #  plt.imshow(observation)
            env.render()
        if save:
            saves.append(observation)
    
    if save:
        return fitness, saves
    return fitness


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


def selection_tournament(items, fitness, n, tsize=5):
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
