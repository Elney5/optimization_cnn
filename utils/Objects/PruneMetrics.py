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
        return pd.DataFrame({"sparsity": [self.sparsity],
                             "val_loss": [self.val_loss],
                             "val_accuracy": [self.val_accuracy]})
   