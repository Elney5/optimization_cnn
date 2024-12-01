import numpy as np

from configparser import ConfigParser
from time import time

from tf_keras import Model
from tensorflow.data import Dataset


def time_benchmark(model: Model, class_names: str, val_data: Dataset) -> tuple[float, float, float]:
    """
    Benchmark the time it takes to make predictions on the validation data
    :param model: Keras model to benchmark
    :param class_names: ex: "Horse or Human"
    :param val_data: Validation data to benchmark (ex: Horse or Human validation data)
    :return: time for first image, mean time for the rest of the images, standard deviation of the time
    """
    file_count = 0
    infer_times = []
    init_timer = 0

    for image in val_data:
        if file_count < 1:
            init_timer_start = time()
            pred = model.predict(image[0])
            pred_class = class_names[int(np.argmax(pred[0]))]
            init_timer_end = time()
            init_timer = init_timer_end - init_timer_start
            file_count += 1
        else:
            timer_start = time()
            pred = model.predict(image[0])
            pred_class = class_names[int(np.argmax(pred[0]))]
            timer_end = time()
            infer_times.append((timer_end - timer_start))
            file_count += 1

    init_time = init_timer
    avg_time = np.mean(infer_times)
    std = np.std(infer_times)

    print(f"The first image takes {init_time * 1000:.2f} ms")
    print(f"The average time taken per 99 images {avg_time * 1000:.2f} ms")
    print(f"The standard deviation of samples is {std * 1000:.2f} ms")

    return init_time, avg_time, std
