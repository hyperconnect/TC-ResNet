from typing import List

import metrics.ops as mops
import metrics.parser as parser
from metrics.base import MetricManagerBase
from metrics.summaries import Summaries


class AudioMetricManager(MetricManagerBase):
    _metric_input_data_parser = parser.AudioDataParser

    def __init__(
            self,
            is_training: bool,
            use_class_metrics: bool,
            exclude_metric_names: List,
            summary: Summaries,
    ):
        super().__init__(exclude_metric_names, summary)
        self.register_metrics([
            # map
            mops.MAPMetricOp(),
            # accuracy
            mops.AccuracyMetricOp(),
            mops.Top5AccuracyMetricOp(),
            # misc
            mops.ClassificationReportMetricOp(),

            # tensor ops
            mops.LossesMetricOp(),
        ])

        if is_training:
            self.register_metrics([
                mops.WavSummaryOp(),
                mops.LearningRateSummaryOp()
            ])

        if use_class_metrics:
            # per-class
            self.register_metrics([
                mops.PrecisionMetricOp(),
                mops.RecallMetricOp(),
                mops.F1ScoreMetricOp(),
                mops.APMetricOp(),
            ])
