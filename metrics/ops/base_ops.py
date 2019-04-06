from abc import ABC, abstractmethod

from common.utils import get_logger
from metrics.summaries import BaseSummaries


class MetricOpBase(ABC):
    MIN_MAX_CHOICES = ["min", "max", None]

    _meta_properties = [
        "is_for_summary",
        "is_for_best_keep",
        "is_for_log",
        "valid_input_data_parsers",
        "summary_collection_key",
        "summary_value_type",
        "min_max_mode",
    ]
    _properties = dict()

    def __init__(self, **kwargs):
        self.log = get_logger("MetricOp")

        # init by _properties
        # custom values can be added as kwargs
        for attr in self._meta_properties:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                setattr(self, attr, self._properties[attr])

        # assertion
        assert self.min_max_mode in self.MIN_MAX_CHOICES

        if self.is_for_best_keep:
            assert self.min_max_mode is not None

        if self.is_for_summary:
            assert self.summary_collection_key in vars(BaseSummaries.KEY_TYPES).values()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def is_placeholder_summary(self):
        assert self.is_for_summary, f"DO NOT call `is_placeholder_summary` method if it is not summary metric"
        return self.summary_value_type == BaseSummaries.VALUE_TYPES.PLACEHOLDER

    @property
    @abstractmethod
    def is_tensor_metric(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def build_op(self, data):
        """ This class should be overloaded for
            all cases of `valid_input_data_parser`
        """
        raise NotImplementedError


class NonTensorMetricOpBase(MetricOpBase):
    @property
    def is_tensor_metric(self):
        return False

    @abstractmethod
    def evaluate(self, data):
        """ This class should be overloaded for
            all cases of `valid_input_data_parser`
        """
        raise NotImplementedError


class TensorMetricOpBase(MetricOpBase):
    @property
    def is_tensor_metric(self):
        return True

    @abstractmethod
    def expectation_of(self, data):
        """ If evaluate is done at tensor metric, it has to re-caculate the expectation of
            aggregated metric values.
            This function assumes that data is aggregated for all batches
            and retruns proper expectation value.
        """
        raise NotImplementedError
