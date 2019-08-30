import unittest
import numpy as np
from helpers.feature_extraction import find_apple_coordinates_pixel_array
import matplotlib.pyplot as plt


class TestAppleCoord(unittest.TestCase):
    def test_one_apple_symmetric(self):
        test_arr = np.zeros((15, 15, 3))
        food_col = np.array([0, 0, 255])
        test_arr[1, 1] = food_col
        np.testing.assert_array_equal(find_apple_coordinates_pixel_array(test_arr, food_col), np.array([1, 1]))

    def test_one_apple_asymmetric(self):
        test_arr = np.zeros((15, 15, 3))
        food_col = np.array([0, 0, 255])
        test_arr[1, 5] = food_col
        plt.imshow(test_arr)
        np.testing.assert_array_equal(find_apple_coordinates_pixel_array(test_arr, food_col), np.array([1, 5]))


if __name__ == '__main__':
    test_arr = np.zeros((15, 15, 3))
    print(test_arr)
