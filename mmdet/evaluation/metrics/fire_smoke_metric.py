from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from typing import Any, List, Optional, Sequence, Union, Dict


@METRICS.register_module()
class FireSmokeMetric(BaseMetric):
    def __init__(self,collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 ann_file= None,
                 metric=None,
                 backend_args: dict = None,) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        pass

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """


        pass
