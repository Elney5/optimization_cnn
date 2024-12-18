
import tensorflow as tf
import tf_keras
from tf_keras import Sequential
from tf_keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth, RandomContrast
# Augmentation of the training data
# Définissez l'objet de data augmentation une fois, globalement
data_augmentation_layer = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomHeight(0.2),
    RandomWidth(0.2),
    RandomContrast(0.2)
])

# La fonction utilise l'objet déjà défini
def data_augmentation(image, label):
    return data_augmentation_layer(image), label
