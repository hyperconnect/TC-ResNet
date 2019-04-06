from pathlib import Path

import tensorflow as tf

import const
from datasets.data_wrapper_base import DataWrapperBase
from datasets.augmentation_factory import get_audio_augmentation_fn


class AudioDataWrapper(DataWrapperBase):
    def __init__(
        self, args, session, dataset_split_name, is_training, name: str="AudioDataWrapper"
    ):
        super().__init__(args, dataset_split_name, is_training, name)
        self.setup()

        self.setup_dataset(self.placeholders)
        self.setup_iterator(
            session,
            self.placeholders,
            self.data,
        )

    def set_fileformat(self, filenames):
        file_formats = set(Path(fn).suffix for fn in filenames)
        assert len(file_formats) == 1
        self.file_format = file_formats.pop()[1:]  # decode_audio receives without .(dot)

    @property
    def num_samples(self):
        return self._num_samples

    def augment_audio(self, filename, desired_samples, file_format, sample_rate, **kwargs):
        aug_fn = get_audio_augmentation_fn(self.args.augmentation_method)
        return aug_fn(filename, desired_samples, file_format, sample_rate, **kwargs)

    def _parse_function(self, filename, label):
        """
        `filename` tensor holds full path of input
        `label` tensor is holding index of class to which it belongs to
        """
        desired_samples = int(self.args.sample_rate * self.args.clip_duration_ms / 1000)

        # augment
        augmented_audio = self.augment_audio(
            filename,
            desired_samples,
            self.file_format,
            self.args.sample_rate,
            background_data=self.background_data,
            is_training=self.is_training,
            background_frequency=self.background_frequency,
            background_max_volume=self.background_max_volume,
        )

        label_parsed = self.parse_label(label)

        return augmented_audio, label_parsed

    @staticmethod
    def add_arguments(parser):
        g = parser.add_argument_group("(AudioDataWrapper) Arguments for Audio DataWrapper")
        g.add_argument(
            "--sample_rate",
            type=int,
            default=16000,
            help="Expected sample rate of the wavs",)
        g.add_argument(
            "--clip_duration_ms",
            type=int,
            default=1000,
            help=("Expected duration in milliseconds of the wavs"
                  "the audio will be cropped or padded with zeroes based on this value"),)
        g.add_argument(
            "--window_size_ms",
            type=float,
            default=30.0,
            help="How long each spectrogram timeslice is.",)
        g.add_argument(
            "--window_stride_ms",
            type=float,
            default=10.0,
            help="How far to move in time between spectogram timeslices.",)

        # {{ -- Arguments for log-mel spectrograms
        # Default values are coming from tensorflow official tutorial
        g.add_argument("--lower_edge_hertz", type=float, default=80.0)
        g.add_argument("--upper_edge_hertz", type=float, default=7600.0)
        g.add_argument("--num_mel_bins", type=int, default=64)
        # Arguments for log-mel spectrograms -- }}

        # {{ -- Arguments for mfcc
        # Google speech_commands sample uses num_mfccs=40 as a default value
        # Official signal processing tutorial uses num_mfccs=13 as a default value
        g.add_argument("--num_mfccs", type=int, default=40)
        # Arguments for mfcc -- }}

        g.add_argument("--input_file", default=None, type=str)
        g.add_argument("--description_file", default=None, type=str)
        g.add_argument("--num_partitions", default=2, type=int,
                       help=("Number of partition to which is input csv file split"
                             "and parallely processed"))

        # background noise
        g.add_argument("--background_max_volume", default=0.1, type=float,
                       help="How loud the background noise should be, between 0 and 1.")
        g.add_argument("--background_frequency", default=0.8, type=float,
                       help="How many of the training samples have background noise mixed in.")
        g.add_argument("--num_silent", default=-1, type=int,
                       help="How many silent data should be added. -1 means automatically calculated.")


class SingleLabelAudioDataWrapper(AudioDataWrapper):
    def parse_label(self, label):
        return tf.sparse_to_dense(sparse_indices=tf.cast(label, tf.int32),
                                  sparse_values=tf.ones([1], tf.float32),
                                  output_shape=[self.num_labels],
                                  validate_indices=False)

    def setup(self):
        dataset_paths = self.get_all_dataset_paths()
        self.label_names, self.num_labels = self.get_label_names(dataset_paths)
        assert const.NULL_CLASS_LABEL in self.label_names
        assert self.args.num_classes == self.num_labels

        self.filenames, self.labels = self.get_filenames_labels(dataset_paths)
        self.set_fileformat(self.filenames)

        # add dummy data for silent class
        self.background_max_volume = tf.constant(self.args.background_max_volume)
        self.background_frequency = tf.constant(self.args.background_frequency)
        self.background_data = self.prepare_silent_data(dataset_paths)
        self.add_silent_data()
        self._num_samples = self.count_samples(self.filenames)

        self.data = (self.filenames, self.labels)

        self.filenames_placeholder = tf.placeholder(tf.string, self._num_samples)
        self.labels_placeholder = tf.placeholder(tf.int32, self._num_samples)
        self.placeholders = (self.filenames_placeholder, self.labels_placeholder)

        # shuffle
        if self.shuffle:
            self.data = self.do_shuffle(*self.data)

    def prepare_silent_data(self, dataset_paths):
        def _gen(filename):
            filename = tf.constant(filename, dtype=tf.string)
            read_fn = get_audio_augmentation_fn("no_augmentation_audio")
            desired_samples = -1  # read all
            wav_data = read_fn(filename, desired_samples, self.file_format, self.args.sample_rate)
            return wav_data

        background_data = list()
        for dataset_path in dataset_paths:
            for label_path in dataset_path.iterdir():
                if label_path.name == const.BACKGROUND_NOISE_DIR_NAME:
                    for wav_fullpath in label_path.glob("*.wav"):
                        background_data.append(_gen(str(wav_fullpath)))

        self.log.info(f"{len(background_data)} background files are loaded.")
        return background_data

    def add_silent_data(self):
        num_silent = self.args.num_silent
        if self.args.num_silent < 0:
            num_samples = self.count_samples(self.filenames)
            num_silent = num_samples // self.num_labels

        label_idx = self.label_names.index(const.NULL_CLASS_LABEL)
        for _ in range(num_silent):
            self.filenames.append("")
            self.labels.append(label_idx)
        self.log.info(f"{num_silent} silent samples will be added.")
