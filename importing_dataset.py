import tensorflow as tf
import tensorflow_datasets as tfds


def load_dataset(dataset_name: str, pourcentage_cross_validation: int) -> tuple:
    # Load the dataset
    (train_examples, validation_examples), info = tfds.load(
        dataset_name,
        split=(
            'train[:' + str(pourcentage_cross_validation) + '%]', 'train[' + str(pourcentage_cross_validation) + '%:]'),
        with_info=True,
        as_supervised=True
    )
    # Information about the dataset Horses or Humans
    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes
    class_names = info.features['label'].names
    return train_examples, validation_examples, num_examples, num_classes, class_names



