
from tf_keras.callbacks import ModelCheckpoint
import os


# Create a function to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name):
  return ModelCheckpoint(filepath=model_name, monitor="val_loss", verbose=0, save_best_only=True)
