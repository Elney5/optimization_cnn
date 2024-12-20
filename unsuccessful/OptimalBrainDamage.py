import numpy as np
import tensorflow as tf
from tf_keras import losses

from SurgeonPackage.Surgeon import Surgeon

class OptimalBrainDamage:

    def __init__(self, model, sparsity):

        self.model = model
        self.sparsity = sparsity
        self.surgeon = Surgeon(self.model)

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
    #                               tf.TensorSpec(shape=(None,), dtype=tf.int32)])
    # def compute_hessian(self, targets, inputs):
    #     with tf.GradientTape(persistent=True) as tape:
    #         tape.watch(self.model.trainable_variables)
    #         predictions = self.model(inputs)
    #         loss = losses.SparseCategoricalCrossentropy()(targets, predictions)
    #
    #     jacobian = tape.jacobian(loss, self.model.trainable_variables)
    #
    #
    #     hessian = []
    #     for jacob in jacobian:
    #         with tf.GradientTape() as tape2:
    #             tape2.watch(self.model.trainable_variables)
    #             second_derivative = tape2.gradient(jacob, self.model.trainable_variables)
    #             hessian.append(second_derivative)
    #     print("Hessian",hessian)
    #     return hessian

    @tf.function
    def get_hessian(self,inputs,targets):
        with tf.GradientTape() as t2:
            t2.watch(inputs)
            with tf.GradientTape() as t1:
                t1.watch(inputs)
                predictions = self.model(inputs)
                loss = losses.SparseCategoricalCrossentropy()(targets, predictions)
            g = t1.gradient(loss, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return t2.jacobian(g, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)



    def prune_weights(self, hessian):
        """
        Applique la méthode de l'Optimal Brain Damage pour supprimer les poids.

        Args:
            hessian (list): La matrice Hessienne des poids.

        Returns:
            model: Le modèle avec les poids supprimés.
        """
        # Calculer le nombre de poids à supprimer
        num_to_prune = int(len(self.model.trainable_variables) * self.sparsity)

        # Pour chaque poids, calculer la contribution de la Hessienne et sélectionner ceux à supprimer
        for idx, grad2 in enumerate(hessian):
            grad2_values = np.array([np.sum(np.abs(g.numpy())) for g in grad2])  # Calculer l'importance
            sorted_indices = np.argsort(grad2_values)[:num_to_prune]

            # Supprimer les poids en utilisant Surgeon
            for i in sorted_indices:
                self.surgeon.delete_layer(self.model.layers[idx], i)  # Utiliser Surgeon pour supprimer le poids

        return self.model

    def run(self, inputs, targets):
        """
        Applique l'Optimal Brain Damage à un modèle Keras pour réduire sa taille.

        Args:
            inputs (Tensor): Les entrées du modèle.
            targets (Tensor): Les cibles pour calculer la perte.

        Returns:
            model: Le modèle avec les poids supprimés.
        """


        # Calculer la Hessienne
        print("Calcul de la Hessienne...")
        hessian = self.get_hessian(inputs,targets)


        # Pruner les poids
        print("Pruning des poids...")
        self.model = self.prune_weights(hessian)

        return self.model
