import numpy as np
import cv2 as cv
from math import atan2, pi


def find_apple_coordinates_pixel_array(pixel_arr, food_color):
    indices = np.where(np.all(pixel_arr == food_color, axis=-1))
    indexes = list(zip(indices[1], indices[0]))
    return np.array(indexes[0])


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm:
         return vector / norm
    return vector


def get_angle(snake, food):
    a = snake.head
    b = food
    a = normalize_vector(a)
    b = normalize_vector(b)
    return np.array([atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / pi])


def distances_to_walls(snake_obj, grid_size):
    up = snake_obj.head[1] - 1
    left = snake_obj.head[0] - 1
    down = grid_size[1] - up
    right = grid_size[0] - left
    return np.array([up, right, down, left])


def feature_vector(snake_object, pixelarr, grid_size, food_color):
    to_pixels = lambda x: cv.resize(x, grid_size)
    direction_vector = np.zeros(4)
    direction_vector[snake_object.direction] = 1
    pixel_grid = to_pixels(pixelarr)
    apple_coord = find_apple_coordinates_pixel_array(pixel_grid, food_color)
    distances = distances_to_walls(snake_object, grid_size)
    angle = get_angle(snake_object, apple_coord)
#     apple_distance = np.array([((snake_object.head[0] - apple_coord[0]) ** 2 + (snake_object.head[1] - apple_coord[1]) ** 2)])
    return np.concatenate((direction_vector, apple_coord, distances, angle))
