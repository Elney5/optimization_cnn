import tensorflow as tf

from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")

IMG_SIZE = (int(config["IMAGES"]["IMG_SIZE"]), int(config["IMAGES"]["IMG_SIZE"]))


def reformat_image(image, label, image_size=IMG_SIZE):
    image = tf.image.resize(image, image_size)
    return image, label
