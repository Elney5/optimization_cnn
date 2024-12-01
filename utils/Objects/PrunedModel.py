from tf_keras.models import Model

from utils.Objects.PruneMetrics import PruneMetrics

class PrunedModel:
    """
    Class to store the pruned model, metrics and logdir
    """
    def __init__(self, model: Model, metric_dict: PruneMetrics, logdir: str):
        self.model: Model = model
        self.metrics: PruneMetrics = metric_dict
        self.logdir: str = logdir