import numpy as np
import tensorflow as tf
import tf_keras


class Entropy:
    def __init__(self):
        self.entropies = []

    def calculate_entropy(self, weight,index):

        if np.shape(weight)[0] != 0:
            p_i = np.abs(weight) / np.sum(np.abs(weight))
            entropy = -np.sum(p_i * np.log(p_i + 1e-10))
            self.entropies.append((index, entropy))
        else:
            self.entropies.append(0)

    def prune_weights(self, weights):

        # If the filter is not null, calculate the entropy
        if np.shape(weights)[0] != 0:
            original_size = np.shape(weights)[0]
            flatten_weights = weights.flatten()

        weight_index = 0
        for weight in flatten_weights:
            self.calculate_entropy(weight,weight_index)

    def get_probability_distribution(self, weights, index):


        distribution_values = np.random.normal(self.__means[index], self.__stds[index], np.size(weights))
        _, bins = np.histogram(distribution_values, bins=int(np.size(weights) - 1), density=True)

        probability_densities = 1 / (self.__stds[index] * np.sqrt(2 * np.pi)) * np.exp(
            - (bins - self.__means[index]) ** 2 / (2 * self.__stds[index] ** 2))
        return probability_densities