from abc import ABC
from types import SimpleNamespace
from pathlib import Path

import tensorflow as tf
from overload import overload

from common.utils import get_logger


class BaseSummaries(ABC):
    KEY_TYPES = SimpleNamespace(
        DEFAULT="SUMMARY_DEFAULT",
        VERBOSE="SUMMARY_VERBOSE",
        FIRST_N="SUMMARY_FIRST_N",
    )

    VALUE_TYPES = SimpleNamespace(
        SCALAR="SCALAR",
        PLACEHOLDER="PLACEHOLDER",
        AUDIO="AUDIO",
        NONE="NONE",  # None is not used for summary
    )

    def __init__(
            self,
            session,
            train_dir,
            is_training,
            base_name="eval",
            max_summary_outputs=None,
    ):
        self.log = get_logger("Summary")

        self.session = session
        self.train_dir = train_dir
        self.max_summary_outputs = max_summary_outputs
        self.base_name = base_name
        self.merged_summaries = dict()

        self.summary_writer = None
        self._setup_summary_writer(is_training)

    def write(self, summary, global_step=0):
        self.summary_writer.add_summary(summary, global_step)

    def setup_experiment(self, config):
        """
        Args:
            config: Namespace
        """
        config = vars(config)

        sorted_config = [(k, str(v)) for k, v in sorted(config.items(), key=lambda x: x[0])]
        config = tf.summary.text("config", tf.convert_to_tensor(sorted_config))

        config_val = self.session.run(config)
        self.write(config_val)
        self.summary_writer.add_graph(tf.get_default_graph())

    def register_summaries(self, collection_summary_dict):
        """
        Args:
            collection_summary_dict: Dict[str, Dict[MetricOp, List[Tuple(metric_key, tensor_op)]]]
                                     collection_key -> metric -> List(summary_key, value)
        """
        for collection_key_suffix, metric_dict in collection_summary_dict.items():
            for metric, key_value_list in metric_dict.items():
                for summary_key, value in key_value_list:
                    self._routine_add_summary_op(summary_value_type=metric.summary_value_type,
                                                 summary_key=summary_key,
                                                 value=value,
                                                 collection_key_suffix=collection_key_suffix)

    def setup_merged_summaries(self):
        for collection_key_suffix in vars(self.KEY_TYPES).values():
            for collection_key in self._iterate_collection_keys(collection_key_suffix):
                merged_summary = tf.summary.merge_all(key=collection_key)
                self.merged_summaries[collection_key] = merged_summary

    def get_merged_summaries(self, collection_key_suffixes: list, is_tensor_summary: bool):
        summaries = []

        for collection_key_suffix in collection_key_suffixes:
            collection_key = self._build_collection_key(collection_key_suffix, is_tensor_summary)
            summary = self.merged_summaries[collection_key]

            if summary is not None:
                summaries.append(summary)

        if len(summaries) == 0:
            return None
        elif len(summaries) == 1:
            return summaries[0]
        else:
            return tf.summary.merge(summaries)

    def write_evaluation_summaries(self, step, collection_keys, collection_summary_dict):
        """
        Args:
            collection_summary_dict: Dict[str, Dict[MetricOp, List[Tuple(metric_key, tensor_op)]]]
                                     collection_key -> metric -> List(summary_key, value)
            collection_keys: List
        """
        for collection_key_suffix, metric_dict in collection_summary_dict.items():
            if collection_key_suffix in collection_keys:
                merged_summary_op = self.get_merged_summaries(collection_key_suffixes=[collection_key_suffix],
                                                              is_tensor_summary=False)
                feed_dict = dict()

                for metric, key_value_list in metric_dict.items():
                    if metric.is_placeholder_summary:
                        for summary_key, value in key_value_list:
                            # https://github.com/tensorflow/tensorflow/issues/3378
                            placeholder_name = self._build_placeholder_name(summary_key) + ":0"
                            feed_dict[placeholder_name] = value

                summary_value = self.session.run(merged_summary_op, feed_dict=feed_dict)
                self.write(summary_value, step)

    def _setup_summary_writer(self, is_training):
        summary_directory = self._build_summary_directory(is_training)
        self.log.info(f"Write summaries into : {summary_directory}")

        if is_training:
            self.summary_writer = tf.summary.FileWriter(summary_directory, self.session.graph)
        else:
            self.summary_writer = tf.summary.FileWriter(summary_directory)

    def _build_summary_directory(self, is_training):
        if is_training:
            return self.train_dir
        else:
            if Path(self.train_dir).is_dir():
                summary_directory = (Path(self.train_dir) / Path(self.base_name)).as_posix()
            else:
                summary_directory = (Path(self.train_dir).parent / Path(self.base_name)).as_posix()

            if not Path(summary_directory).exists():
                Path(summary_directory).mkdir(parents=True)

            return summary_directory

    def _routine_add_summary_op(self, summary_value_type, summary_key, value, collection_key_suffix):
        collection_key = self._build_collection_key(collection_key_suffix, summary_value_type)

        if summary_value_type == self.VALUE_TYPES.SCALAR:
            def register_fn(k, v):
                return tf.summary.scalar(k, v, collections=[collection_key])

        elif summary_value_type == self.VALUE_TYPES.AUDIO:
            def register_fn(k, v):
                return tf.summary.audio(k, v,
                                        sample_rate=16000,
                                        max_outputs=self.max_summary_outputs,
                                        collections=[collection_key])

        elif summary_value_type == self.VALUE_TYPES.PLACEHOLDER:
            def register_fn(k, v):
                return tf.summary.scalar(k, v, collections=[collection_key])
            value = self._build_placeholder(summary_key)

        else:
            raise NotImplementedError

        register_fn(summary_key, value)

    @classmethod
    def _build_placeholder(cls, summary_key):
        name = cls._build_placeholder_name(summary_key)
        return tf.placeholder(tf.float32, [], name=name)

    @staticmethod
    def _build_placeholder_name(summary_key):
        return f"non_tensor_summary_placeholder/{summary_key}"

    # Below two functions should be class method but defined as instance method
    # since it has bug in @overload
    # @classmethod
    @overload
    def _build_collection_key(self, collection_key_suffix, summary_value_type: str):
        if summary_value_type == self.VALUE_TYPES.PLACEHOLDER:
            prefix = "NON_TENSOR"
        else:
            prefix = "TENSOR"

        return f"{prefix}_{collection_key_suffix}"

    # @classmethod
    @_build_collection_key.add
    def _build_collection_key(self, collection_key_suffix, is_tensor_summary: bool):
        if not is_tensor_summary:
            prefix = "NON_TENSOR"
        else:
            prefix = "TENSOR"

        return f"{prefix}_{collection_key_suffix}"

    @classmethod
    def _iterate_collection_keys(cls, collection_key_suffix):
        for prefix in ["NON_TENSOR", "TENSOR"]:
            yield f"{prefix}_{collection_key_suffix}"


class Summaries(BaseSummaries):
    pass
