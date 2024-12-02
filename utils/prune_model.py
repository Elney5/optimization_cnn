import tempfile
import numpy as np

import tensorflow_model_optimization as tfmot

from tensorflow.data import Dataset
from tf_keras import optimizers, losses, Model

from utils.Objects import PrunedModel, PruneMetrics, EntropyPruningSurgeon,L1NormPruning


def unstructured_prune_model(model: Model,
                             final_sparsity: int,
                             epochs: int,
                             train_data: Dataset,
                             val_data: Dataset,
                             batch_size: int
                             ) -> PrunedModel:
    """
    Prune a model using unstructured pruning
    :param model: Keras model to prune
    :param final_sparsity: Sparsity level to prune the model to
    :param epochs: Number of epochs to train for
    :param train_data: Training data
    :param val_data: Validation data
    :param batch_size: Batch size for training
    :return: PrunedModel object containing the pruned model, metrics and logdir
    """

    # Create a tensorboard logfile
    logdir = tempfile.mkdtemp()
    # The end_step is the total number of iterations required for the training data which is basically the entire epochs over the length of the training data
    end_step = int(len(train_data) * epochs * 0.5)
    # Import the low-magnitude-pruning function
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Set the pruning params
    pruning_params = {

        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0,
                                                                 final_sparsity=final_sparsity,
                                                                 begin_step=0,
                                                                 end_step=end_step)

    }

    # Model for pruning
    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # Recompile
    model_for_pruning.compile(optimizer=optimizers.Adam(),
                              loss=losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
    # create callbacks
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                 tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)
                 ]

    # Fit the model
    model_for_pruning.fit(train_data,
                          validation_data=val_data,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks)

    # Evaluate the model
    score = model_for_pruning.evaluate(val_data, verbose=0)
    metric_dict = PruneMetrics(
        sparsity=final_sparsity,
        val_loss=np.round(score[0], 4),
        val_accuracy=np.round(score[1] * 100, 4)
    )
    return PrunedModel(model=model_for_pruning, metrics=metric_dict, logdir=logdir)


def entropy_prune_model(model, epochs, threshold, train_data, val_data, batch_size):
    # Create a tensorboard logfile
    logdir = tempfile.mkdtemp()

    # Model for pruning
    entropy_pruning_instance = EntropyPruningSurgeon(model=model, threshold=threshold)
    model_for_pruning = entropy_pruning_instance.run()

    # Recompile
    model_for_pruning.compile(optimizer=optimizers.Adam(),
                              loss=losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])

    # Fit the model
    model_for_pruning.fit(train_data,
                          validation_data=val_data,
                          batch_size=batch_size,
                          epochs=epochs)

    # Evaluate the model
    score = model_for_pruning.evaluate(val_data, verbose=0)
    metric_dict = {
        "threshold": threshold,
        "val_loss": np.round(score[0], 4),
        "val_accuracy": np.round(score[1] * 100, 4)
    }

    return PrunedModel(model=model_for_pruning, metrics=metric_dict, logdir=logdir)

def l1_norm_prune_model(model,epochs,threshold,train_data,val_data,batch_size):
    # Create a tensorboard logfile
    logdir = tempfile.mkdtemp()

    # Model for pruning
    l1_norm_pruning_instance = L1NormPruning(model=model, threshold=threshold)
    model_for_pruning = l1_norm_pruning_instance.run()

    # Recompile
    model_for_pruning.compile(optimizer=optimizers.Adam(),
                              loss=losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])

    # Fit the model
    model_for_pruning.fit(train_data,
                          validation_data=val_data,
                          batch_size=batch_size,
                          epochs=epochs)

    # Evaluate the model
    score = model_for_pruning.evaluate(val_data, verbose=0)
    metric_dict = {
        "threshold": threshold,
        "val_loss": np.round(score[0], 4),
        "val_accuracy": np.round(score[1] * 100, 4)
    }

    return PrunedModel(model=model_for_pruning, metrics=metric_dict, logdir=logdir)

