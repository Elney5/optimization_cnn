import tensorflow as tf
IMAGE_SIZE = (224, 224)
def reformat_image(image, label,image_size=IMAGE_SIZE):
  image = tf.image.resize(image, image_size)
  return image,label
