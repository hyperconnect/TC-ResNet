from abc import ABC
from abc import abstractmethod

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import tensorflow as tf

import const


class PreprocessorBase(ABC):
    def __init__(self, scope: str, preprocessed_node_name: str):
        self._scope = scope
        self._input_node = None
        self._preprocessed_node = None
        self._preprocessed_node_name = preprocessed_node_name

    @abstractmethod
    def preprocess(self, inputs, reuse):
        raise NotImplementedError

    @staticmethod
    def _uint8_to_float32(inputs):
        if inputs.dtype == tf.uint8:
            inputs = tf.cast(inputs, tf.float32)
        return inputs

    def _assign_input_node(self, inputs):
        # We include preprocessing part in TFLite float model so we need an input tensor before preprocessing.
        self._input_node = inputs

    def _make_node_after_preprocessing(self, inputs):
        node = tf.identity(inputs, name=self._preprocessed_node_name)
        self._preprocessed_node = node
        return node

    @property
    def input_node(self):
        return self._input_node

    @property
    def preprocessed_node(self):
        return self._preprocessed_node


class NoOpPreprocessor(PreprocessorBase):
    def preprocess(self, inputs, reuse=False):
        self._assign_input_node(inputs)
        inputs = self._make_node_after_preprocessing(inputs)
        return inputs


# For Audio
class AudioPreprocessorBase(PreprocessorBase):
    def preprocess(self, inputs, window_size_samples, window_stride_samples, for_deploy, **kwargs):
        self._assign_input_node(inputs)
        with tf.variable_scope(self._scope):
            if for_deploy:
                inputs = self._preprocess_for_deploy(inputs, window_size_samples, window_stride_samples, **kwargs)
            else:
                inputs = self._preprocess(inputs, window_size_samples, window_stride_samples, **kwargs)
        inputs = self._make_node_after_preprocessing(inputs)
        return inputs

    def _log_mel_spectrogram(self, audio, window_size_samples, window_stride_samples,
                             magnitude_squared, **kwargs):
        # only accept single channels
        audio = tf.squeeze(audio, -1)
        stfts = tf.contrib.signal.stft(audio,
                                       frame_length=window_size_samples,
                                       frame_step=window_stride_samples)

        # If magnitude_squared = True(power_spectrograms)#, tf.real(stfts * tf.conj(stfts))
        # If magnitude_squared = False(magnitude_spectrograms), tf.abs(stfts)
        if magnitude_squared:
            spectrograms = tf.real(stfts * tf.conj(stfts))
        else:
            spectrograms = tf.abs(stfts)

        num_spectrogram_bins = spectrograms.shape[-1].value
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            kwargs["num_mel_bins"],
            num_spectrogram_bins,
            kwargs["sample_rate"],
            kwargs["lower_edge_hertz"],
            kwargs["upper_edge_hertz"],
        )

        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )

        log_offset = 1e-6
        log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

        return log_mel_spectrograms

    def _single_spectrogram(self, audio, window_size_samples, window_stride_samples, magnitude_squared):
        # only accept single batch
        audio = tf.squeeze(audio, 0)

        spectrogram = contrib_audio.audio_spectrogram(
            audio,
            window_size=window_size_samples,
            stride=window_stride_samples,
            magnitude_squared=magnitude_squared
        )

        return spectrogram

    def _single_mfcc(self, audio, window_size_samples, window_stride_samples, magnitude_squared,
                     **kwargs):
        spectrogram = self._single_spectrogram(audio, window_size_samples, window_stride_samples, magnitude_squared)

        mfcc = contrib_audio.mfcc(
            spectrogram,
            kwargs["sample_rate_const"],
            upper_frequency_limit=kwargs["upper_edge_hertz"],
            lower_frequency_limit=kwargs["lower_edge_hertz"],
            filterbank_channel_count=kwargs["num_mel_bins"],
            dct_coefficient_count=kwargs["num_mfccs"],
        )

        return mfcc

    def _get_mel_matrix(self, num_mel_bins, num_spectrogram_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz):
        if num_mel_bins == 64 and num_spectrogram_bins == 257 and sample_rate == 16000 \
                and lower_edge_hertz == 80.0 and upper_edge_hertz == 7600.0:
            return tf.constant(const.MEL_WEIGHT_64_257_16000_80_7600, dtype=tf.float32, name="mel_weight_matrix")
        elif num_mel_bins == 64 and num_spectrogram_bins == 513 and sample_rate == 16000 \
                and lower_edge_hertz == 80.0 and upper_edge_hertz == 7600.0:
            return tf.constant(const.MEL_WEIGHT_64_513_16000_80_7600, dtype=tf.float32, name="mel_weight_matrix")
        else:
            setting = (num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
            raise ValueError(f"Target setting is not defined: {setting}")

    def _single_log_mel_spectrogram(self, audio, window_size_samples, window_stride_samples,
                                    magnitude_squared, **kwargs):
        spectrogram = self._single_spectrogram(audio, window_size_samples, window_stride_samples, magnitude_squared)
        spectrogram = tf.squeeze(spectrogram, 0)

        num_spectrogram_bins = spectrogram.shape[-1].value
        linear_to_mel_weight_matrix = self._get_mel_matrix(
            kwargs["num_mel_bins"],
            num_spectrogram_bins,
            kwargs["sample_rate"],
            kwargs["lower_edge_hertz"],
            kwargs["upper_edge_hertz"],
        )

        mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)

        log_offset = 1e-6
        log_mel_spectrograms = tf.log(mel_spectrogram + log_offset)
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 0)

        return log_mel_spectrograms


class LogMelSpectrogramPreprocessor(AudioPreprocessorBase):
    def _preprocess(self, audio, window_size_samples, window_stride_samples, **kwargs):
        # When calculate log mel spectogram, set magnitude_squared False
        log_mel_spectrograms = self._log_mel_spectrogram(audio,
                                                         window_size_samples,
                                                         window_stride_samples,
                                                         False,
                                                         **kwargs)
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, axis=-1)
        return log_mel_spectrograms

    def _preprocess_for_deploy(self, audio, window_size_samples, window_stride_samples, **kwargs):
        log_mel_spectrogram = self._single_log_mel_spectrogram(audio,
                                                               window_size_samples,
                                                               window_stride_samples,
                                                               False,
                                                               **kwargs)
        log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)
        return log_mel_spectrogram


class MFCCPreprocessor(AudioPreprocessorBase):
    def _preprocess(self, audio, window_size_samples, window_stride_samples, **kwargs):
        # When calculate log mel spectogram, set magnitude_squared True
        log_mel_spectrograms = self._log_mel_spectrogram(audio,
                                                         window_size_samples,
                                                         window_stride_samples,
                                                         True,
                                                         **kwargs)

        mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        mfccs = mfccs[..., :kwargs["num_mfccs"]]
        mfccs = tf.expand_dims(mfccs, axis=-1)
        return mfccs

    def _preprocess_for_deploy(self, audio, window_size_samples, window_stride_samples, **kwargs):
        mfcc = self._single_mfcc(audio,
                                 window_size_samples,
                                 window_stride_samples,
                                 True,
                                 **kwargs)
        mfcc = tf.expand_dims(mfcc, axis=-1)
        return mfcc
