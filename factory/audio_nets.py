from typing import Tuple
from typing import Dict
from types import SimpleNamespace
import types

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import preprocessor_factory
import common.tf_utils as tf_utils
import common.utils as utils
from factory.base import TFModel
from audio_nets import kws
from audio_nets import res
from audio_nets import ds_cnn
from audio_nets import tc_resnet


_available_nets = [
    "KWSModel",  # http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    "Res8Model",
    "Res8NarrowModel",
    "Res15Model",
    "Res15NarrowModel",
    "DSCNNSModel",
    "DSCNNMModel",
    "DSCNNLModel",
    "TCResNet8Model",
    "TCResNet14Model",
    "ResNet2D8Model",
    "ResNet2D8PoolModel",
]


class AudioNetModel(TFModel):
    def __init__(self, args, dataset):
        self.log = utils.get_logger("AudioNetModel")
        self.dataset = dataset
        self.args = args

    def build(self, wavs, labels, is_training):
        self._audio_original = wavs
        self.is_training = is_training
        self.labels = labels  # will be used in build_evaluation_fetch_ops

        self.preprocess_input()

        # -- * -- build_output: it includes build_inference
        self.inputs, self.logits, self._outputs, self.endpoints = self.build_output(
            self.audio,
            self.is_training,
            self.args.output_name
        )

        # -- * -- Build loss function: it can be different between each models
        self._total_loss, self._model_loss, self.endpoints_loss = self.build_loss(
            self.logits, self.outputs, self.labels
        )

        self.total_params = tf_utils.show_models(self.log)

    def preprocess_input(self, for_deploy=False):
        window_size_samples = int(self.args.sample_rate * self.args.window_size_ms / 1000)
        window_stride_samples = int(self.args.sample_rate * self.args.window_stride_ms / 1000)
        if for_deploy:
            self.args.sample_rate_const = tf.constant(self.args.sample_rate, dtype=tf.int32)

        audio_preprocessor = preprocessor_factory.factory(
            preprocess_method=self.args.preprocess_method,
            scope="input/audio/preprocessing",
            preprocessed_node_name="input/audio/preprocessed",
        )

        self._audio = audio_preprocessor.preprocess(
            self._audio_original,
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            for_deploy=for_deploy,
            **vars(self.args)
        )

        self.log.info(f"Update height/width to {self._audio.shape.as_list()}")
        self.args.height, self.args.width, self.args.channels = self._audio.shape.as_list()[1:4]

        self.input_preprocessors_for_tflite = [audio_preprocessor]

    def build_deployable_model(self, include_preprocess=True):
        if include_preprocess:
            desired_samples = int(self.args.sample_rate * self.args.clip_duration_ms / 1000)

            self._audio = tf.placeholder(
                tf.float32,
                shape=[self.args.input_batch_size, desired_samples, 1],
                name="input/audio/before_preprocessing",
            )
            self._audio_original = self._audio
            input_tensors = [self.audio]
            self.preprocess_input(for_deploy=True)

            _, _, output_tensor, _ = self.build_output(
                self.audio,
                False,
                self.args.output_name,
            )
        else:
            self.log.info("Build graph which excludes preprocessing for freezing!")
            # only use for profiling
            assert self.args.height > 0
            assert self.args.width > 0
            assert self.args.channels > 0

            inputs = tf.placeholder(
                tf.float32,
                shape=[1, self.args.height, self.args.width, self.args.channels],
                name="input",
            )

            input_tensors = [inputs]
            _, _, output_tensor, _ = self.build_output(
                inputs,
                False,
                self.args.output_name,
            )

        return input_tensors, output_tensor

    @property
    def model_loss(self):
        return self._model_loss

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def audio_original(self):
        return self._audio_original

    @property
    def audio(self):
        return self._audio

    @property
    def outputs(self):
        return self._outputs

    def build_output(
        self,
        inputs: tf.Tensor,
        is_training: tf.placeholder,
        output_name: str
    ) -> [tf.Tensor, tf.Tensor, tf.Tensor, Dict]:
        logits, endpoints = self.build_inference(inputs, is_training=is_training)
        output = slim.softmax(logits, scope=output_name + "/softmax")
        output = tf.identity(output, name=output_name)
        return inputs, logits, output, endpoints

    def build_inference(self, inputs, is_training=True):
        raise NotImplementedError

    def build_loss(
        self,
        logits: tf.Tensor,
        scores: tf.Tensor,
        labels: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, Dict]:
        endpoints_loss = {}
        model_loss = tf.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=labels,
            label_smoothing=self.args.label_smoothing,
            weights=1.0,
        )

        def exclude_batch_norm(name):
            return ("batch_normalization" not in name) and ("BatchNorm" not in name)

        l2_loss = self.args.weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
        )

        total_loss = model_loss + l2_loss
        return total_loss, model_loss, endpoints_loss

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--label_smoothing", default=0.0, type=float)


class GoogleKWS():
    def __init__(self, args):
        self.args = args

    def build_model_settings(self, inputs):
        model_settings = {}
        model_settings["fingerprint_width"] = inputs.shape.as_list()[2]
        model_settings["spectrogram_length"] = inputs.shape.as_list()[1]
        model_settings["fingerprint_size"] = model_settings["spectrogram_length"] * model_settings["fingerprint_width"]
        model_settings["label_count"] = self.args.num_classes
        model_settings["sample_rate"] = self.args.sample_rate
        model_settings["window_stride_samples"] = int(self.args.sample_rate * self.args.window_stride_ms / 1000)
        return model_settings


class KWSModel(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)
        self.google_kws = GoogleKWS(args)

    def build_inference(self, inputs, is_training=True):
        endpoints = {}
        logits = kws.create_model(
            inputs,
            model_settings=self.google_kws.build_model_settings(inputs),
            model_architecture=self.args.architecture,
            is_training=is_training,
        )
        return logits, endpoints

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--architecture", default="conv",
                            choices=["single_fc", "conv", "low_latency_conv", "low_latency_svdf", "tiny_conv",
                                     "one_fstride4", "trad_fpool3"])


class Res8Model(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.00001, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(res.Res_arg_scope(
            is_training=is_training,
            weight_decay=self.args.weight_decay)
        ):
            logits, endpoints = res.Res8(inputs, self.args.num_classes)

            return logits, endpoints


class Res8NarrowModel(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.00001, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(res.Res_arg_scope(
            is_training=is_training,
            weight_decay=self.args.weight_decay)
        ):
            logits, endpoints = res.Res8Narrow(inputs, self.args.num_classes)

            return logits, endpoints


class Res15Model(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.00001, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(res.Res_arg_scope(
            is_training=is_training,
            weight_decay=self.args.weight_decay)
        ):
            logits, endpoints = res.Res15(inputs, self.args.num_classes)

            return logits, endpoints


class Res15NarrowModel(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.00001, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(res.Res_arg_scope(
            is_training=is_training,
            weight_decay=self.args.weight_decay)
        ):
            logits, endpoints = res.Res15Narrow(inputs, self.args.num_classes)

            return logits, endpoints


class DSCNNSModel(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.0, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(ds_cnn.DSCNN_arg_scope(
            is_training=is_training)
        ):
            logits, endpoints = ds_cnn.DSCNN(
                inputs,
                self.args.num_classes,
                ds_cnn.S_NET_DEF,
            )

            return logits, endpoints


class DSCNNMModel(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.0, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(ds_cnn.DSCNN_arg_scope(
            is_training=is_training)
        ):
            logits, endpoints = ds_cnn.DSCNN(
                inputs,
                self.args.num_classes,
                ds_cnn.M_NET_DEF,
            )

            return logits, endpoints


class DSCNNLModel(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.0, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(ds_cnn.DSCNN_arg_scope(
            is_training=is_training)
        ):
            logits, endpoints = ds_cnn.DSCNN(
                inputs,
                self.args.num_classes,
                ds_cnn.L_NET_DEF,
            )

            return logits, endpoints


class TCResNet8Model(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.0001, type=float)
        parser.add_argument("--dropout_keep_prob", default=0.5, type=float)
        parser.add_argument("--width_multiplier", default=1.0, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tc_resnet.TCResNet_arg_scope(
            is_training=is_training,
            weight_decay=self.args.weight_decay,
            keep_prob=self.args.dropout_keep_prob)
        ):
            logits, endpoints = tc_resnet.TCResNet8(
                inputs,
                self.args.num_classes,
                width_multiplier=self.args.width_multiplier,
            )

            return logits, endpoints


class TCResNet14Model(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.0001, type=float)
        parser.add_argument("--dropout_keep_prob", default=0.5, type=float)
        parser.add_argument("--width_multiplier", default=1.0, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tc_resnet.TCResNet_arg_scope(
            is_training=is_training,
            weight_decay=self.args.weight_decay,
            keep_prob=self.args.dropout_keep_prob)
        ):
            logits, endpoints = tc_resnet.TCResNet14(
                inputs,
                self.args.num_classes,
                width_multiplier=self.args.width_multiplier,
            )

            return logits, endpoints


class ResNet2D8Model(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.0001, type=float)
        parser.add_argument("--dropout_keep_prob", default=0.5, type=float)
        parser.add_argument("--width_multiplier", default=1.0, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tc_resnet.TCResNet_arg_scope(
            is_training=is_training,
            weight_decay=self.args.weight_decay,
            keep_prob=self.args.dropout_keep_prob)
        ):
            logits, endpoints = tc_resnet.ResNet2D8(
                inputs,
                self.args.num_classes,
                width_multiplier=self.args.width_multiplier,
            )

            return logits, endpoints


class ResNet2D8PoolModel(AudioNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--weight_decay", default=0.0001, type=float)
        parser.add_argument("--dropout_keep_prob", default=0.5, type=float)
        parser.add_argument("--width_multiplier", default=1.0, type=float)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tc_resnet.TCResNet_arg_scope(
            is_training=is_training,
            weight_decay=self.args.weight_decay,
            keep_prob=self.args.dropout_keep_prob)
        ):
            logits, endpoints = tc_resnet.ResNet2D8Pool(
                inputs,
                self.args.num_classes,
                width_multiplier=self.args.width_multiplier
            )

        return logits, endpoints
