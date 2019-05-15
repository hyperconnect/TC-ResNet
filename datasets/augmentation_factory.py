from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import tensorflow as tf


_available_audio_augmentation_methods = [
    "anchored_slice_or_pad",
    "anchored_slice_or_pad_with_shift",
    "no_augmentation_audio",
]


_available_augmentation_methods = (
    _available_audio_augmentation_methods +
    ["no_augmentation"]
)


def no_augmentation(x):
    return x


def _gen_random_from_zero(maxval, dtype=tf.float32):
    return tf.random.uniform([], maxval=maxval, dtype=dtype)


def _gen_empty_audio(desired_samples):
    return tf.zeros([desired_samples, 1], dtype=tf.float32)


def _mix_background(
        audio,
        desired_samples,
        background_data,
        is_silent,
        is_training,
        background_frequency,
        background_max_volume,
        naive_version=True,
        **kwargs
):
    """
    Args:
        audio: Tensor of audio.
        desired_samples: int value of desired length.
        background_data: List of background audios.
        is_silent: Tensor[Bool].
        is_training: Tensor[Bool].
        background_frequency: probability of mixing background. [0.0, 1.0]
        background_max_volume: scaling factor of mixing background. [0.0, 1.0]
    """
    foreground_wav = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(desired_samples),
        false_fn=lambda: tf.identity(audio)
    )

    # sampling background
    random_background_data_idx = _gen_random_from_zero(
        len(background_data),
        dtype=tf.int32
    )
    background_wav = tf.case({
        tf.equal(background_data_idx, random_background_data_idx): 
            lambda tensor=wav: tensor
        for background_data_idx, wav in enumerate(background_data)
    }, exclusive=True)
    background_wav = tf.random_crop(background_wav, [desired_samples, 1])

    if naive_version:
        # Version 1
        # https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/input_data.py#L461
        if is_training:
            background_volume = tf.cond(
                tf.less(_gen_random_from_zero(1.0), background_frequency),
                true_fn=lambda: _gen_random_from_zero(background_max_volume),
                false_fn=lambda: 0.0,
            )
        else:
            background_volume = 0.0
    else:
        # Version 2
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py#L570
        background_volume = tf.cond(
            tf.logical_or(is_training, is_silent),
            true_fn=lambda: tf.cond(
                is_silent,
                true_fn=lambda: _gen_random_from_zero(1.0),
                false_fn=lambda: tf.cond(
                    tf.less(_gen_random_from_zero(1.0), background_frequency),
                    true_fn=lambda: _gen_random_from_zero(background_max_volume),
                    false_fn=lambda: 0.0,
                ),
            ),
            false_fn=lambda: 0.0,
        )

    background_wav = tf.multiply(background_wav, background_volume)
    background_added = tf.add(background_wav, foreground_wav)
    augmented_audio = tf.clip_by_value(background_added, -1.0, 1.0)

    return augmented_audio


def _shift_audio(audio, desired_samples, shift_ratio=0.1):
    time_shift = int(desired_samples * shift_ratio)
    time_shift_amount = tf.random.uniform(
        [],
        minval=-time_shift,
        maxval=time_shift,
        dtype=tf.int32
    )

    time_shift_abs = tf.abs(time_shift_amount)

    def _pos_padding():
        return [[time_shift_amount, 0], [0, 0]]

    def _pos_offset():
        return [0, 0]

    def _neg_padding():
        return [[0, time_shift_abs], [0, 0]]

    def _neg_offset():
        return [time_shift_abs, 0]

    padded_audio = tf.pad(
        audio,
        tf.cond(tf.greater_equal(time_shift_amount, 0),
                true_fn=_pos_padding,
                false_fn=_neg_padding),
        mode="CONSTANT",
    )

    sliced_audio = tf.slice(
        padded_audio,
        tf.cond(tf.greater_equal(time_shift_amount, 0),
                true_fn=_pos_offset,
                false_fn=_neg_offset),
        [desired_samples, 1],
    )

    return sliced_audio


def _load_wav_file(filename, desired_samples, file_format):
    if file_format == "wav":
        wav_decoder = contrib_audio.decode_wav(
            tf.read_file(filename),
            desired_channels=1,
            # If desired_samples is set, then the audio will be
            # cropped or padded with zeroes to the requested length.
            desired_samples=desired_samples,
        )
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return wav_decoder.audio


def no_augmentation_audio(
        filename,
        desired_samples,
        file_format,
        sample_rate,
        **kwargs
):
    return _load_wav_file(filename, desired_samples, file_format)


def anchored_slice_or_pad(
        filename,
        desired_samples,
        file_format,
        sample_rate,
        **kwargs,
):
    is_silent = tf.equal(tf.strings.length(filename), 0)

    audio = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(desired_samples),
        false_fn=lambda: _load_wav_file(filename, desired_samples, file_format)
    )

    if "background_data" in kwargs:
        audio = _mix_background(audio, desired_samples, is_silent=is_silent, **kwargs)

    return audio


def anchored_slice_or_pad_with_shift(
        filename,
        desired_samples,
        file_format,
        sample_rate,
        **kwargs
):
    is_silent = tf.equal(tf.strings.length(filename), 0)

    audio = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(desired_samples),
        false_fn=lambda: _load_wav_file(filename, desired_samples, file_format)
    )
    audio = _shift_audio(audio, desired_samples, shift_ratio=0.1)

    if "background_data" in kwargs:
        audio = _mix_background(audio, desired_samples, is_silent=is_silent, **kwargs)

    return audio


def get_audio_augmentation_fn(name):
    if name not in _available_audio_augmentation_methods:
        raise ValueError(f"Augmentation name [{name}] was not recognized")
    return eval(name)
