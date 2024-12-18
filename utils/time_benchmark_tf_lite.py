import numpy as np

from configparser import ConfigParser
from time import time
import tensorflow as tf

from tf_keras import Model
from tensorflow.data import Dataset

def benchmark_tflite(val_data, tflite_model, class_names, image_size):
    # Initialisation du modèle TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    file_count = 0
    infer_times = []

    # Itération sur les données du Dataset
    for images, labels in val_data:
        # Vérification et traitement de chaque image du lot
        images = tf.image.resize(images, image_size)  # Redimensionner les images
        images = tf.cast(images, tf.float32)  # Convertir en float32
        images = images / 255.0  # Normalisation

        # Traiter chaque image dans le lot une par une
        for img in images:
            if file_count < 1:
                # Mesurer le temps d'initialisation pour la première image
                init_timer_start = time.time()
                img = np.expand_dims(img.numpy(), axis=0)  # Convertir en numpy et ajouter une dimension de lot
                interpreter.set_tensor(input_index, img)  # Passer à TensorFlow Lite
                interpreter.invoke()
                output = interpreter.tensor(output_index)
                pred_class = class_names[int(np.argmax(output()[0]))]
                init_timer_end = time.time()
                init_timer = init_timer_end - init_timer_start
                file_count += 1
            else:
                # Mesurer le temps d'inférence pour les images suivantes
                timer_start = time.time()
                img = np.expand_dims(img.numpy(), axis=0)  # Convertir en numpy et ajouter une dimension de lot
                interpreter.set_tensor(input_index, img)  # Passer à TensorFlow Lite
                interpreter.invoke()
                output = interpreter.tensor(output_index)
                pred_class = class_names[int(np.argmax(output()[0]))]
                timer_end = time.time()
                infer_times.append(timer_end - timer_start)
                file_count += 1

    # Calculer la moyenne et l'écart type des temps d'inférence
    avg_time = np.mean(infer_times) if infer_times else 0
    std_time = np.std(infer_times) if infer_times else 0

    return init_timer, avg_time, std_time