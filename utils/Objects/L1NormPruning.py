import numpy as np
from tf_keras import layers, Model, Input
from SurgeonPackage.Surgeon import Surgeon

class L1NormPruning:
    def __init__(self, model, sparsity):
        self.model = model
        self.sparsity = sparsity

    def calculate_L1_norm(self, layer):
        weights, _ = layer.get_weights()
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("Les poids contiennent des valeurs NaN ou Inf.")

        # Calcul des normes L1 des filtres
        l1_norms = np.sum(np.abs(weights), axis=(0, 1, 2))  # Calcul L1 pour chaque filtre
        return l1_norms

    def prune_model(self):
        """
        Prune le modèle en fonction d'une sparsity cible

        """
        surgeon = Surgeon(self.model)

        # Parcours des couches du modèle
        for layer in self.model.layers:
            if isinstance(layer, layers.Conv2D):
                # Calcul des normes L1 des filtres
                l1_norms = self.calculate_L1_norm(layer)

                # Nombre total de filtres
                num_filters = len(l1_norms)

                # Nombre de filtres à supprimer en fonction de la sparsity
                num_to_prune = int(num_filters *self.sparsity)

                # Si le nombre de filtres à supprimer est trop faible, on ignore la couche
                if num_to_prune <= 0:
                    continue

                # Indices des filtres à supprimer (les plus petits selon L1)
                pruned_indices = np.argsort(l1_norms)[:num_to_prune]

                # Ajouter une tâche pour supprimer ces filtres
                surgeon.add_job("delete_channels", layer, channels=pruned_indices)
                print(f"Pruned {num_to_prune} filters ({self.sparsity * 100:.1f}% sparsity) in layer '{layer.name}'.")

        # Appliquer les changements au modèle
        pruned_model = surgeon.operate()
        return pruned_model

    # def prune_model(self):
    #     surgeon = Surgeon(self.model)
    #
    #     # Parcours de toutes les couches du modèle
    #     for layer in self.model.layers:
    #         if isinstance(layer, layers.Conv2D):
    #             # Calcul des normes L1 des filtres
    #             l1_norms = self.calculate_L1_norm(layer)
    #
    #             # Trouver le filtre avec la plus petite norme L1
    #             max_l1_index = np.argmax(l1_norms)
    #
    #             # Trouver les indices des filtres à supprimer (normes L1 inférieures au seuil)
    #             pruned_indices = [i for i, l1 in enumerate(l1_norms) if l1 < self.threshold]
    #
    #             # Si on doit supprimer tous les filtres, on garde uniquement celui avec la norme L1 la plus grande
    #             if len(pruned_indices) >= len(l1_norms):
    #                 # On supprime toute la couche sauf le filtre avec la norme L1 la plus grande
    #                 pruned_indices.remove(max_l1_index)  # Retirer l'indice du filtre "important"
    #                 print(
    #                     f"Suppression complète de la couche '{layer.name}' sauf le filtre avec la norme L1 la plus élevée.")
    #                 surgeon.add_job("delete_channels", layer, channels=pruned_indices)
    #             else:
    #                 # Supprimer les filtres dont la norme L1 est inférieure au seuil
    #                 surgeon.add_job("delete_channels", layer, channels=pruned_indices)
    #                 print(f"Filtres supprimés dans la couche '{layer.name}'.")
    #
    #     pruned_model = surgeon.operate()
    #     return pruned_model
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
