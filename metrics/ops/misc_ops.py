from metrics.ops.base_ops import TensorMetricOpBase
from metrics.summaries import BaseSummaries
import metrics.parser as parser


class LearningRateSummaryOp(TensorMetricOpBase):
    """
    Learning rate summary.
    """
    _properties = {
        "is_for_summary": True,
        "is_for_best_keep": False,
        "is_for_log": True,
        "valid_input_data_parsers": [
            parser.AudioDataParser,
        ],
        "summary_collection_key": BaseSummaries.KEY_TYPES.DEFAULT,
        "summary_value_type": BaseSummaries.VALUE_TYPES.SCALAR,
        "min_max_mode": None,
    }

    def __str__(self):
        return "summary_lr"

    def build_op(self,
                 data):
        res = dict()
        if data.learning_rate is None:
            pass
        else:
            res[f"learning_rate/{data.dataset_split_name}"] = data.learning_rate

        return res

    def expectation_of(self, data):
        pass
