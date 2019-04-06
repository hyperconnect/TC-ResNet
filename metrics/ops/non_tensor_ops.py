from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from overload import overload

import metrics.parser as parser
from metrics.funcs import topN_accuracy
from metrics.ops.base_ops import NonTensorMetricOpBase
from metrics.summaries import BaseSummaries


class MAPMetricOp(NonTensorMetricOpBase):
    """
    Micro Mean Average Precision Metric.
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
        "min_max_mode": "max",
    }

    _average_fns = {
        "macro": lambda t, p: average_precision_score(t, p, average="macro"),
        "micro": lambda t, p: average_precision_score(t, p, average="micro"),
        "weighted": lambda t, p: average_precision_score(t, p, average="weighted"),
        "samples": lambda t, p: average_precision_score(t, p, average="samples"),
    }

    def __str__(self):
        return "mAP_metric"

    @overload
    def build_op(self,
                 data: parser.AudioDataParser.OutputBuildData):
        result = dict()

        for avg_name in self._average_fns:
            key = f"mAP/{data.dataset_split_name}/{avg_name}"
            result[key] = None

        return result

    @overload
    def evaluate(self,
                 data: parser.AudioDataParser.OutputNonTensorData):
        result = dict()

        for avg_name, avg_fn in self._average_fns.items():
            key = f"mAP/{data.dataset_split_name}/{avg_name}"
            result[key] = avg_fn(data.labels_onehot, data.predictions_onehot)

        return result


class AccuracyMetricOp(NonTensorMetricOpBase):
    """
    Accuracy Metric.
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
        "min_max_mode": "max",
    }

    def __str__(self):
        return "accuracy_metric"

    @overload
    def build_op(self,
                 data: parser.AudioDataParser.OutputBuildData):
        key = f"accuracy/{data.dataset_split_name}"

        return {
            key: None
        }

    @overload
    def evaluate(self,
                 data: parser.AudioDataParser.OutputNonTensorData):
        key = f"accuracy/{data.dataset_split_name}"

        metric = accuracy_score(data.labels, data.predictions)

        return {
            key: metric
        }


class Top5AccuracyMetricOp(NonTensorMetricOpBase):
    """
    Top 5 Accuracy Metric.
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
        "min_max_mode": "max",
    }

    def __str__(self):
        return "top5_accuracy_metric"

    @overload
    def build_op(self,
                 data: parser.AudioDataParser.OutputBuildData):
        key = f"top5_accuracy/{data.dataset_split_name}"

        return {
            key: None
        }

    @overload
    def evaluate(self,
                 data: parser.AudioDataParser.OutputNonTensorData):
        key = f"top5_accuracy/{data.dataset_split_name}"

        metric = topN_accuracy(y_true=data.labels,
                               y_pred_onehot=data.predictions_onehot,
                               N=5)

        return {
            key: metric
        }


class PrecisionMetricOp(NonTensorMetricOpBase):
    """
    Precision Metric.
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
        "min_max_mode": "max",
    }

    def __str__(self):
        return "precision_metric"

    @overload
    def build_op(self,
                 data: parser.AudioDataParser.OutputBuildData):
        result = dict()

        label_idxes = list(range(len(data.label_names)))

        for label_idx in label_idxes:
            label_name = data.label_names[label_idx]
            key = f"precision/{data.dataset_split_name}/{label_name}"
            result[key] = None

        return result

    @overload
    def evaluate(self,
                 data: parser.AudioDataParser.OutputNonTensorData):
        result = dict()

        label_idxes = list(range(len(data.label_names)))
        precisions = precision_score(data.labels, data.predictions, average=None, labels=label_idxes)

        for label_idx in label_idxes:
            label_name = data.label_names[label_idx]
            key = f"precision/{data.dataset_split_name}/{label_name}"
            metric = precisions[label_idx]
            result[key] = metric

        return result


class RecallMetricOp(NonTensorMetricOpBase):
    """
    Recall Metric.
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
        "min_max_mode": "max",
    }

    def __str__(self):
        return "recall_metric"

    @overload
    def build_op(self,
                 data: parser.AudioDataParser.OutputBuildData):
        result = dict()

        label_idxes = list(range(len(data.label_names)))

        for label_idx in label_idxes:
            label_name = data.label_names[label_idx]
            key = f"recall/{data.dataset_split_name}/{label_name}"
            result[key] = None

        return result

    @overload
    def evaluate(self,
                 data: parser.AudioDataParser.OutputNonTensorData):
        result = dict()

        label_idxes = list(range(len(data.label_names)))
        recalls = recall_score(data.labels, data.predictions, average=None, labels=label_idxes)

        for label_idx in label_idxes:
            label_name = data.label_names[label_idx]
            key = f"recall/{data.dataset_split_name}/{label_name}"
            metric = recalls[label_idx]
            result[key] = metric

        return result


class F1ScoreMetricOp(NonTensorMetricOpBase):
    """
    Per class F1-Score Metric.
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
        "min_max_mode": "max",
    }

    def __str__(self):
        return "f1_score_metric"

    @overload
    def build_op(self,
                 data: parser.AudioDataParser.OutputBuildData):
        result = dict()

        label_idxes = list(range(len(data.label_names)))

        for label_idx in label_idxes:
            label_name = data.label_names[label_idx]
            key = f"f1score/{data.dataset_split_name}/{label_name}"
            result[key] = None

        return result

    @overload
    def evaluate(self,
                 data: parser.AudioDataParser.OutputNonTensorData):
        result = dict()

        label_idxes = list(range(len(data.label_names)))
        f1_scores = f1_score(data.labels, data.predictions, average=None, labels=label_idxes)

        for label_idx in label_idxes:
            label_name = data.label_names[label_idx]
            key = f"f1score/{data.dataset_split_name}/{label_name}"
            metric = f1_scores[label_idx]
            result[key] = metric

        return result


class APMetricOp(NonTensorMetricOpBase):
    """
    Per class Average Precision Metric.
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
        "min_max_mode": "max",
    }

    def __str__(self):
        return "ap_score_metric"

    @overload
    def build_op(self,
                 data: parser.AudioDataParser.OutputBuildData):
        result = dict()

        label_idxes = list(range(len(data.label_names)))

        for label_idx in label_idxes:
            label_name = data.label_names[label_idx]
            key = f"ap/{data.dataset_split_name}/{label_name}"
            result[key] = None

        return result

    @overload
    def evaluate(self,
                 data: parser.AudioDataParser.OutputNonTensorData):
        result = dict()

        label_idxes = list(range(len(data.label_names)))
        aps = average_precision_score(data.labels_onehot, data.predictions_onehot, average=None)

        for label_idx in label_idxes:
            label_name = data.label_names[label_idx]
            key = f"ap/{data.dataset_split_name}/{label_name}"
            metric = aps[label_idx]
            result[key] = metric

        return result


class ClassificationReportMetricOp(NonTensorMetricOpBase):
    """
    Accuracy Metric.
    """
    _properties = {
        "is_for_summary": False,
        "is_for_best_keep": False,
        "is_for_log": True,
        "valid_input_data_parsers": [
            parser.AudioDataParser,
        ],
        "summary_collection_key": None,
        "summary_value_type": None,
        "min_max_mode": None,
    }

    def __str__(self):
        return "classification_report_metric"

    @overload
    def build_op(self,
                 data: parser.AudioDataParser.OutputBuildData):
        key = f"classification_report/{data.dataset_split_name}"

        return {
            key: None
        }

    @overload
    def evaluate(self,
                 data: parser.AudioDataParser.OutputNonTensorData):
        key = f"classification_report/{data.dataset_split_name}"

        label_idxes = list(range(len(data.label_names)))
        metric = classification_report(data.labels,
                                       data.predictions,
                                       labels=label_idxes,
                                       target_names=data.label_names)
        metric = f"[ClassificationReport]\n{metric}"

        return {
            key: metric
        }
