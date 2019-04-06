import numpy as np

import metrics.parser as parser
from metrics.ops.base_ops import TensorMetricOpBase
from metrics.summaries import BaseSummaries


class LossesMetricOp(TensorMetricOpBase):
    """ Loss Metric.
    """
    _properties = {
        "is_for_summary": True,
        "is_for_best_keep": True,
        "is_for_log": True,
        "valid_input_data_parsers": [
            parser.AudioDataParser,
        ],
        "summary_collection_key": BaseSummaries.KEY_TYPES.DEFAULT,
        "summary_value_type": BaseSummaries.VALUE_TYPES.PLACEHOLDER,
        "min_max_mode": "min",
    }

    def __str__(self):
        return "losses"

    def build_op(self, data):
        result = dict()

        for loss_name, loss_op in data.losses.items():
            key = f"metric_loss/{data.dataset_split_name}/{loss_name}"
            result[key] = loss_op

        return result

    def expectation_of(self, data: np.array):
        assert len(data.shape) == 2
        return np.mean(data)


class WavSummaryOp(TensorMetricOpBase):
    _properties = {
        "is_for_summary": True,
        "is_for_best_keep": False,
        "is_for_log": False,
        "valid_input_data_parsers": [
            parser.AudioDataParser,
        ],
        "summary_collection_key": BaseSummaries.KEY_TYPES.DEFAULT,
        "summary_value_type": BaseSummaries.VALUE_TYPES.AUDIO,
        "min_max_mode": None,
    }

    def __str__(self):
        return "summary_wav"

    def build_op(self, data: parser.AudioDataParser.OutputBuildData):
        return {
            f"wav/{data.dataset_split_name}": data.wavs
        }

    def expectation_of(self, data):
        pass
