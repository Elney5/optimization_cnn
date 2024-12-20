import tensorflow as tf
import numpy as np
import tf_keras
from tf_keras.models import Model
from tf_keras import layers


def calculate_apoz(layer_outputs, precision=1e-5):
    print("Calculating APoZ...")
    # Identifier les activations nulles
    zero_activations = tf.less(tf.abs(layer_outputs), precision)
    # Compter le nombre d'activations nulles par filtre
    zero_count_per_filter = tf.reduce_sum(tf.cast(zero_activations, tf.float32), axis=[0, 1, 2])
    # Nombre total d'activations par filtre
    total_activations_per_filter = tf.cast(
        tf.reduce_prod(layer_outputs.shape[:3]), tf.float32
    )
    # Calculer l'APoZ (pourcentage d'activations nulles par filtre)
    apoz_per_filter = zero_count_per_filter / total_activations_per_filter
    return apoz_per_filter


class APoZ_Algorithm:
    def __init__(self, model, data, threshold):
        """
        Initialisation de l'algorithme APoZ.
        :param model: Modèle Keras à analyser.
        :param data: Données d'entrée pour calculer les activations.
        :param threshold: Seuil d'APoZ pour sélectionner les filtres à supprimer.
        """
        self.model = model
        self.data = data
        self.threshold = threshold

        # Configurer un modèle pour extraire les activations des couches convolutionnelles
        self.activation_model = Model(
            inputs=model.input,
            outputs=[layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
        )

    def analyze_layers(self):
        """
        Analyse les activations des couches convolutionnelles pour calculer l'APoZ.
        :return: Liste des filtres à supprimer pour chaque couche.
        """
        print("Analyzing layers for APoZ calculation...")

        # Extraire les images `x` du dataset
        # images = tf.concat([batch[0] for batch in self.data], axis=0)
        #
        # # Vérifier la forme des données
        # assert len(images.shape) == 4, f"Expected 4D input, but got shape {images.shape}"
        # print(f"Number of input images: {images.shape[0]}")
        #
        # # Normaliser les images
        # images = tf.cast(images, tf.float32) / 255.0
        # print("Normalized input data.")

        # Vérifier l'activation_model
        print("Activation model summary:")
        print(self.activation_model)
        self.activation_model.summary()

        # Obtenir les activations intermédiaires pour toutes les images
        print("Extracting activations from the model...")
        self.data = self.data = self.data.take(3)  # Prendre les 100 premiers échantillons
        print(f"Number of samples: {len(self.data)}")
        # Utiliser uniquement les 100 prem
        activations = self.activation_model.predict(self.data)

        # Vérifier que les activations ont été générées
        if not isinstance(activations, list) or len(activations) == 0:
            raise ValueError("The activation model did not return any outputs. Check the model configuration.")

        print(f"Number of layers analyzed: {len(activations)}")
        print("Calculating APoZ for each layer...")

        # Calculer l'APoZ pour chaque couche
        try:
            apoz_per_layer = [calculate_apoz(layer_output) for layer_output in activations]
        except Exception as e:
            raise ValueError(f"Error during APoZ calculation: {e}")

        # Afficher les valeurs APoZ pour déboguer
        for i, apoz in enumerate(apoz_per_layer):
            print(f"Layer {i}, APoZ shape: {apoz.shape}")

        # Identifier les filtres à supprimer dans chaque couche
        filters_to_prune = [np.where(apoz > self.threshold)[0] for apoz in apoz_per_layer]

        return filters_to_prune

    def prune_filters(self, filters_to_prune):
        """
        Supprime les filtres identifiés dans les couches du modèle.
        :param filters_to_prune: Liste des filtres à supprimer pour chaque couche.
        :return: Modèle avec les filtres supprimés.
        """
        print("Pruning filters based on APoZ analysis...")
        for layer, filters in zip(self.model.layers, filters_to_prune):
            if isinstance(layer, layers.Conv2D):
                # Récupérer les poids actuels de la couche
                weights, biases = layer.get_weights()

                # Supprimer les filtres (poids et biais) correspondant à APoZ élevé
                new_weights = np.delete(weights, filters, axis=-1)  # Supprimer les poids des filtres
                new_biases = np.delete(biases, filters, axis=0)  # Supprimer les biais des filtres

                # Mettre à jour les poids de la couche
                layer.set_weights([new_weights, new_biases])

        return self.model

    def run(self):
        print("Running APoZ algorithm...")
        # Étape 1 : Analyser les activations pour identifier les filtres à supprimer
        filters_to_prune = self.analyze_layers()
        # Étape 2 : Supprimer les filtres identifiés
        pruned_model = self.prune_filters(filters_to_prune)
        return pruned_model