import numpy as np
import tf_keras
from tf_keras import layers,initializers
from tf_keras.models import Sequential
from tf2kerassurgeon import Surgeon

class EntropyPruning:
    def __init__(self, model, threshold, stability_margin=0.07, pruning_frequency=1, seed=42):
        self.model = model
        self.threshold = threshold
        self.stability_margin = stability_margin
        self.pruning_frequency = pruning_frequency
        self.seed = seed
        self.previous_loss = None  # Pour suivre la stabilité

    def calculate_entropy(self, layer):
        weights, _ = layer.get_weights()
        num_filters = weights.shape[-1]

        if np.shape(weights)[0] != 0:
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                raise ValueError("Les poids contiennent des valeurs NaN ou Inf.")

            total_weight_sum = np.sum(np.abs(weights))
            if total_weight_sum == 0:
                entropies = np.zeros(num_filters)
            else:
                p_i = np.abs(weights) / total_weight_sum
                entropies = -np.sum(p_i * np.log(p_i + 1e-10), axis=(0, 1, 2))
        else:
            entropies = np.zeros(num_filters)
        return entropies

    def model_is_stable(self, current_loss):
        if self.previous_loss is None:
            self.previous_loss = current_loss
            return True
        delta_loss = abs(current_loss - self.previous_loss)
        self.previous_loss = current_loss
        return delta_loss < self.stability_margin

    def prune_layer(self, layer, prev_output_depth):
        entropies = self.calculate_entropy(layer)
        pruned_indices = [i for i, h in enumerate(entropies) if h < self.threshold]
        if not pruned_indices:
            return layer, prev_output_depth

        weights, biases = layer.get_weights()
        flatten_weights = weights.flatten()
        flatten_biases = biases.flatten()

        for i in pruned_indices:
            start_index = i * flatten_weights.shape[0]
            end_index = start_index + flatten_weights.shape[0]
            flatten_weights[start_index:end_index] = 0
            flatten_biases[i] = 0

        new_weights = flatten_weights.reshape(weights.shape)
        new_biases = flatten_biases.reshape(biases.shape)


        layer.set_weights([new_weights, new_biases])

        return layer, len(pruned_indices)



    def prune_model(self, current_loss):
        if not self.model_is_stable(current_loss):
            print("Model is unstable; skipping pruning.")
            print(self.model_is_stable(current_loss))
            return self.model

        new_layers = []
        prev_output_depth = self.model.input_shape[-1]
        spatial_height, spatial_width = self.model.input_shape[1:3]

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, layers.Conv2D):
                new_layer, prev_output_depth = self.prune_layer(layer, prev_output_depth)
                if new_layer is not None:
                    new_layers.append(new_layer)
            elif isinstance(layer, layers.MaxPooling2D):
                pool_size = layer.pool_size
                strides = layer.strides if layer.strides else pool_size
                spatial_height = (spatial_height - pool_size[0]) // strides[0] + 1
                spatial_width = (spatial_width - pool_size[1]) // strides[1] + 1
                new_layers.append(layer)
            else:
                new_layers.append(layer)

        new_model = Sequential(new_layers)
        new_model.build((None, spatial_height, spatial_width, prev_output_depth))
        return new_model
