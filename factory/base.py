from abc import ABC
from abc import abstractmethod

import tensorflow as tf
import tensorflow.contrib.slim as slim

import common.tf_utils as tf_utils
from datasets.preprocessor_factory import _available_preprocessors


class TFModel(ABC):
    @staticmethod
    def add_arguments(parser):
        g_cnn = parser.add_argument_group("(CNNModel) Arguments")
        g_cnn.add_argument("--num_classes", type=int, default=None)
        g_cnn.add_argument("--checkpoint_path", default="", type=str)

        g_cnn.add_argument("--input_batch_size", type=int, default=1)
        g_cnn.add_argument("--output_name", type=str, required=True)

        g_cnn.add_argument("--preprocess_method", required=True, type=str,
                           choices=list(_available_preprocessors.keys()))

        g_cnn.add_argument("--no-ignore_missing_vars", dest="ignore_missing_vars", action="store_false")
        g_cnn.add_argument("--ignore_missing_vars", dest="ignore_missing_vars", action="store_true")
        g_cnn.set_defaults(ignore_missing_vars=False)

        g_cnn.add_argument("--checkpoint_exclude_scopes", default="", type=str,
                           help=("Prefix scopes that shoule be EXLUDED for restoring variables "
                                 "(comma separated)"))

        g_cnn.add_argument("--checkpoint_include_scopes", default="", type=str,
                           help=("Prefix scopes that should be INCLUDED for restoring variables "
                                 "(comma separated)"))
        g_cnn.add_argument("--weight_decay", default=1e-4, type=float)

    @abstractmethod
    def build_deployable_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def preprocess_input(self):
        pass

    @abstractmethod
    def build_output(self):
        pass

    @property
    @abstractmethod
    def audio(self):
        pass

    @property
    @abstractmethod
    def audio_original(self):
        pass

    @property
    @abstractmethod
    def total_loss(self):
        pass

    @property
    @abstractmethod
    def model_loss(self):
        pass
