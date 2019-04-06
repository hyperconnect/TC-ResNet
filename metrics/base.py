from abc import ABC
from collections import defaultdict

from common.utils import get_logger
from common.utils import format_text
from common.utils import timer


class DataStructure(ABC):
    """
    Define inner data structure
    Should define `_keys`
    """
    _keys = None

    def __init__(self, data):
        keys = self.__class__.get_keys()
        data_keys = data.keys()

        if set(keys) != set(data_keys):
            raise ValueError(f"Keys defined in `_keys ({list(keys)})`"
                             f" should be appeared at "
                             f"`data ({list(data_keys)})`")
        for k in keys:
            setattr(self, k, data[k])

    def __str__(self):
        return f"<DataStructure: {self.to_dict()}>"

    def __repr__(self):
        return str(self)

    def to_dict(self):
        return {k: getattr(self, k) for k in self._keys}

    @classmethod
    def get_keys(cls):
        return cls._keys


class MetricAggregator:
    def __init__(self):
        self.step = None
        self.metrics_with_result = None

        self.init(-1)

    def init(self, step):
        self.step = step
        self.metrics_with_result = dict()

    def aggregate(self, metric, metric_result):
        assert metric not in self.metrics_with_result
        self.metrics_with_result[metric] = metric_result

    def iterate_metrics(self):
        for metric, metric_result in self.metrics_with_result.items():
            yield metric, metric_result

    def iterate_all(self):
        for metric, metric_result in self.iterate_metrics():
            for metric_key, value in metric_result.items():
                yield metric, metric_key, value

    def get_collection_summary_dict(self):
        # for_summary: Dict[str, Dict[MetricOp, List[Tuple(metric_key, tensor_op)]]]
        # collection_key -> metric -> List(summary_key, value)
        for_summary = defaultdict(lambda: defaultdict(list))
        for metric, metric_result in self.metrics_with_result.items():
            if metric.is_for_summary:
                for metric_key, value in metric_result.items():
                    for_summary[metric.summary_collection_key][metric].append((metric_key, value))

        return for_summary

    def get_tensor_metrics(self):
        """
        Get metric that would be fetched for session run.
        """
        tensor_metrics = dict()
        for metric, metric_result in self.metrics_with_result.items():
            if metric.is_tensor_metric:
                for metric_key, value in metric_result.items():
                    tensor_metrics[metric_key] = value

        return tensor_metrics

    def get_logs(self):
        logs = dict()
        for metric, metric_result in self.metrics_with_result.items():
            if metric.is_for_log:
                for metric_key, value in metric_result.items():
                    if isinstance(value, str):
                        msg = f"> {metric_key}\n{value}"
                    else:
                        msg = f"> {metric_key} : {value}"
                    logs[metric_key] = msg

        return logs


class MetricManagerBase(ABC):
    _metric_input_data_parser = None

    def __init__(self, exclude_metric_names, summary):
        self.log = get_logger("Metrics")
        self.build_op_aggregator = MetricAggregator()
        self.eval_metric_aggregator = MetricAggregator()

        self.summary = summary
        self.exclude_metric_names = exclude_metric_names

        self.metric_ops = []

    def register_metric(self, metric):
        # if metric is in exclude_metric_names ?
        if metric.__class__.__name__ in self.exclude_metric_names:
            self.log.info(f"{metric.__class__.__name__} is excluded by user setting.")
            return

        # assertion for this metric would be processable
        assert str(self._metric_input_data_parser) in map(lambda c: str(c), metric.valid_input_data_parsers), \
            f"{metric.__class__.__name__} cannot be parsed by {self._metric_input_data_parser}"

        # add one
        self.metric_ops.append(metric)
        self.log.info(f"{metric.__class__.__name__} is added.")

    def register_metrics(self, metrics: list):
        for metric in metrics:
            self.register_metric(metric)

    def build_metric_ops(self, data):
        """
        Define tensor metric operations
        1. call `build_op` of metrics, i.e. add operations to graph
        2. register summaries

        Return: Dict[str, Tensor]
            metric_key -> metric_op
        """
        output_build_data = self._metric_input_data_parser.parse_build_data(data)

        # get metric tf ops
        for metric in self.metric_ops:
            try:
                metric_build_ops = metric.build_op(output_build_data)
            except TypeError as e:
                raise TypeError(f"[{metric}]: {e}")
            self.build_op_aggregator.aggregate(metric, metric_build_ops)

        # if value is not None, it means it is defined with tensor
        metric_tf_ops = self.build_op_aggregator.get_tensor_metrics()

        # register summary
        collection_summary_dict = self.build_op_aggregator.get_collection_summary_dict()
        self.summary.register_summaries(collection_summary_dict)
        self.summary.setup_merged_summaries()

        return metric_tf_ops

    # def evaluate_non_tensor_metric(self, data, step):
    def evaluate_and_aggregate_metrics(self, non_tensor_data, eval_dict, step):
        """
        Run evaluation of non-tensor metrics
        Args:
            data: data passed from trainer / evaluator/ ...
        """
        non_tensor_data = self._metric_input_data_parser.parse_non_tensor_data(non_tensor_data)

        # aggregate metrics
        self.eval_metric_aggregator.init(step)

        # evaluate all metrics
        for metric, metric_key_op_dict in self.build_op_aggregator.iterate_metrics():
            if metric.is_tensor_metric:
                with timer(f"{metric}.expectation_of"):
                    # already aggregated - tensor ops
                    metric_result = dict()
                    for metric_key in metric_key_op_dict:
                        if metric_key in eval_dict:
                            exp_value = metric.expectation_of(eval_dict[metric_key])
                            metric_result[metric_key] = exp_value
            else:
                with timer(f"{metric}.evaluate"):
                    # need calculation - non tensor ops
                    metric_result = metric.evaluate(non_tensor_data)

            self.eval_metric_aggregator.aggregate(metric, metric_result)

    def write_tensor_summaries(self, step, summary_value):
        self.summary.write(summary_value, step)

    def write_evaluation_summaries(self, step, collection_keys):
        assert step == self.eval_metric_aggregator.step, \
            (f"step: {step} is different from aggregator's step: {self.eval_metric_aggregator.step}"
             f"`evaluate` function should be called before calling this function")

        collection_summary_dict = self.eval_metric_aggregator.get_collection_summary_dict()
        self.summary.write_evaluation_summaries(step=step,
                                                collection_keys=collection_keys,
                                                collection_summary_dict=collection_summary_dict)

    def log_metrics(self, step):
        """
        Logging metrics that are evaluated.
        """
        assert step == self.eval_metric_aggregator.step, \
            (f"step: {step} is different from aggregator's step: {self.eval_metric_aggregator.step}"
             f"`evaluate` function should be called before calling this function")

        log_dicts = dict()
        log_dicts.update(self.eval_metric_aggregator.get_logs())

        with format_text("green", ["bold"]) as fmt:
            for metric_key, log_str in log_dicts.items():
                self.log.info(fmt(log_str))

    def get_evaluation_result(self, step):
        """
        Retrun evaluation result regardless of metric type.
        """
        assert step == self.eval_metric_aggregator.step, \
            (f"step: {step} is different from aggregator's step: {self.eval_metric_aggregator.step}"
             f"`evaluate` function should be called before calling this function")

        eval_dict = dict()
        for metric, metric_key, value in self.eval_metric_aggregator.iterate_all():
            eval_dict[metric_key] = value

        return eval_dict

    def get_best_keep_metric_with_modes(self):
        metric_min_max_dict = dict()
        for metric, metric_key, _ in self.build_op_aggregator.iterate_all():
            if metric.is_for_best_keep:
                metric_min_max_dict[metric_key] = metric.min_max_mode

        return metric_min_max_dict

    def filter_best_keep_metric(self, eval_metric_dict):
        best_keep_metric_dict = dict()
        for metric, metric_key, _ in self.build_op_aggregator.iterate_all():
            if metric_key in eval_metric_dict and metric.is_for_best_keep:
                best_keep_metric_dict[metric_key] = eval_metric_dict[metric_key]

        return best_keep_metric_dict

    @staticmethod
    def add_arguments(parser):
        subparser = parser.add_argument_group(f"Metric Manager Arguments")
        subparser.add_argument("--exclude_metric_names",
                               nargs="*",
                               default=[],
                               type=str,
                               help="Name of metrics to be excluded")
        subparser.add_argument("--max_summary_outputs",
                               default=3,
                               type=int,
                               help="Number of maximum summary outputs for multimedia (ex: audio wav)")
