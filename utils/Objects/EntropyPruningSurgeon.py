import numpy as np
from tf_keras import layers, Model, Input
from SurgeonPackage.Surgeon import Surgeon

class EntropyPruningSurgeon:
    """
    Classe pour effectuer le pruning basé sur l'entropie d'un modèle Keras.
    """
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def calculate_entropy(self, layer):
        weights, _ = layer.get_weights()
        num_filters = weights.shape[-1]  # Nombre de filtres

        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("Les poids contiennent des valeurs NaN ou Inf.")

        total_weight_sum = np.sum(np.abs(weights))
        if total_weight_sum == 0:
            return np.zeros(num_filters)

        # Probabilité normalisée des poids
        p_i = np.abs(weights) / total_weight_sum
        # Entropie calculée par -∑ p*log(p)
        entropies = -np.sum(p_i * np.log(p_i + 1e-10), axis=(0, 1, 2))  # Ajout de 1e-10 pour éviter log(0)
        return entropies

    def prune_model(self):
        """
        Effectue le pruning des filtres basés sur l'entropie.

        Returns:
            tf.keras.Model: Le modèle pruné.
        """
        surgeon = Surgeon(self.model)
        for layer in self.model.layers:
            if isinstance(layer, layers.Conv2D):
                # Calcul des entropies des filtres
                entropies = self.calculate_entropy(layer)

                # Trouver le filtre avec la plus petite entropie
                min_entropy_index = np.argmin(entropies)

                # Indices des filtres à supprimer
                pruned_indices = [i for i, e in enumerate(entropies) if e < self.threshold]


                # Vérifier que tous les filtres ne sont pas supprimés
                num_filters = entropies.shape[0]
                if len(pruned_indices) >= num_filters:
                    # Supprimer toute la couche
                    pruned_indices.remove(min_entropy_index)
                    surgeon.add_job("delete_channels", layer, channels=pruned_indices)
                else:
                    # Supprimer uniquement certains filtres
                    surgeon.add_job("delete_channels", layer, channels=pruned_indices)



        # Applique les modifications et retourne un nouveau modèle pruné
        pruned_model = surgeon.operate()
        return pruned_model

    def rebuild_model(self, pruned_model):
        """
        Reconstruit le modèle pruné pour éviter les problèmes de KerasTensors.

        Args:
            pruned_model (tf.keras.Model): Le modèle pruné.

        Returns:
            tf.keras.Model: Le modèle reconstruit.
        """
        # Entrée du modèle
        input_shape = pruned_model.input_shape[1:]  # Ignorer la batch size
        inputs = Input(shape=input_shape)

        # Passer les entrées à travers chaque couche pour reconstruire le modèle
        x = inputs
        for layer in pruned_model.layers:
            x = layer(x)

        # Crée le nouveau modèle fonctionnel
        rebuilt_model = Model(inputs=inputs, outputs=x)
        return rebuilt_model

    def run(self):
        """
        Exécute le pruning et retourne le modèle résultant.

        Returns:
            tf.keras.Model: Le modèle pruné et reconstruit.
        """
        # Prune le modèle
        pruned_model = self.prune_model()

        # Reconstruit le modèle pour éviter l'erreur des KerasTensors
        rebuilt_model = self.rebuild_model(pruned_model)

        return rebuilt_model
