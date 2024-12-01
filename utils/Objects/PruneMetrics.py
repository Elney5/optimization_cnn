class PruneMetrics:
    """
    Class to store the metrics of the pruned model
    """

    def __init__(self, sparsity: int, val_loss: float, val_accuracy: float):
        # Add the keys to the class
        self.sparsity: int = sparsity
        self.val_loss: float = val_loss
        self.val_accuracy: float = val_accuracy

    def to_df(self):
        """
        Convert the metrics to a pandas DataFrame
        """
        import pandas as pd
        return pd.DataFrame({"Sparsity": [self.sparsity],
                             "Validation Loss": [self.val_loss],
                             "Validation Accuracy": [self.val_accuracy]})
