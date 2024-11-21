import tensorflow as tf
from tf_keras.callbacks import Callback

from sklearn.metrics import classification_report


class F1Score(Callback):
    def __init__(self, validation_data, labels):
        super().__init__()
        self.validation_data = validation_data
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        # Collect true labels and predictions
        true_classes = []
        predicted_classes = []

        for images, labels in self.validation_data:
            predictions = self.model.predict(images)
            predicted_classes.extend(tf.argmax(predictions, axis=1).numpy())
            true_classes.extend(labels.numpy())  # Use labels as they are

        # Compute confusion matrix
        confusion_matrix = tf.math.confusion_matrix(
            true_classes,
            predicted_classes,
            num_classes=len(self.labels)
        )

        print(f"\nConfusion Matrix at epoch {epoch + 1}:\n{confusion_matrix.numpy()}")

        # Compute F1-score, precision, and recall
        report = classification_report(
            true_classes,
            predicted_classes,
            target_names=self.labels,
            digits=4  # Customize the precision of displayed results
        )
        print(f"\nClassification Report at epoch {epoch + 1}:\n{report}")
        return report, confusion_matrix
