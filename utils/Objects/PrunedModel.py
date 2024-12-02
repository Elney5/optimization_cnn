from tf_keras.models import Model

from utils.Objects.PruneMetrics import PruneMetrics

class PrunedModel:
    """
    Class to store the pruned model, metrics and logdir
    """
    def __init__(self, model: Model, metrics: PruneMetrics, logdir: str):
        self.model: Model = model
        self.metrics: PruneMetrics = metrics
        self.logdir: str = logdir