from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Tuple
from typing import List
from collections import defaultdict
import random

import tensorflow as tf
import pandas as pd
from termcolor import colored

import common.utils as utils
import const
from datasets.augmentation_factory import _available_augmentation_methods


class DataWrapperBase(ABC):
    def __init__(
        self,
        args,
        dataset_split_name: str,
        is_training: bool,
        name: str,
    ):
        self.name = name
        self.args = args
        self.dataset_split_name = dataset_split_name
        self.is_training = is_training

        self.shuffle = self.args.shuffle

        self.log = utils.get_logger(self.name)
        self.timer = utils.Timer(self.log)
        self.dataset_path = Path(self.args.dataset_path)
        self.dataset_path_with_split_name = self.dataset_path / self.dataset_split_name

        with utils.format_text("yellow", ["underline"]) as fmt:
            self.log.info(self.name)
            self.log.info(fmt(f"dataset_path_with_split_name: {self.dataset_path_with_split_name}"))
            self.log.info(fmt(f"dataset_split_name: {self.dataset_split_name}"))

    @property
    @abstractmethod
    def num_samples(self):
        pass

    @property
    def batch_size(self):
        try:
            return self._batch_size
        except AttributeError:
            self._batch_size = 0

    @batch_size.setter
    def batch_size(self, val):
        self._batch_size = val

    def setup_dataset(
        self,
        placeholders: Tuple[tf.placeholder, tf.placeholder],
        batch_size: int=None,
    ):
        self.batch_size = self.args.batch_size if batch_size is None else batch_size

        # single-GPU: prefetch before batch-shuffle-repeat
        dataset = tf.data.Dataset.from_tensor_slices(placeholders)
        if self.shuffle:
            dataset = dataset.shuffle(self.num_samples)  # Follow tf.data.Dataset.list_files
        dataset = dataset.map(self._parse_function, num_parallel_calls=self.args.num_threads)

        if hasattr(tf.contrib.data, "AUTOTUNE"):
            dataset = dataset.prefetch(
                buffer_size=tf.contrib.data.AUTOTUNE
            )
        else:
            dataset = dataset.prefetch(
                buffer_size=self.args.prefetch_factor * self.batch_size
            )

        dataset = dataset.batch(self.batch_size)
        if self.is_training and self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.args.buffer_size, reshuffle_each_iteration=True).repeat(-1)
        elif self.is_training and not self.shuffle:
            dataset = dataset.repeat(-1)

        self.dataset = dataset
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_elem = self.iterator.get_next()

    def setup_iterator(
            self,
            session: tf.Session,
            placeholders: Tuple[tf.placeholder, ...],
            variables: Tuple[tf.placeholder, ...],
    ):
        assert len(placeholders) == len(variables), "Length of placeholders and variables differ!"
        with self.timer(colored("Initialize data iterator.", "yellow")):
            session.run(self.iterator.initializer,
                        feed_dict={placeholder: variable for placeholder, variable in zip(placeholders, variables)})

    def get_input_and_output_op(self):
        return self.next_elem

    def __str__(self):
        return f"path: {self.args.dataset_path}, split: {self.args.dataset_split_name} data size: {self._num_samples}"

    def get_all_dataset_paths(self) -> List[str]:
        if self.args.has_sub_dataset:
            return sorted([p for p in self.dataset_path_with_split_name.glob("*/") if p.is_dir()])
        else:
            return [self.dataset_path_with_split_name]

    def get_label_names(
        self,
        dataset_paths: List[str],
    ):
        """Get all label names (either from one or all subdirectories if subdatasets are defined)
        and check consistency of names.

        Args:
            dataset_paths: List of paths to datasets.

        Returns:
            name_labels: Names of labels.
            num_labels: Number of all labels.
        """
        tmp_label_names = []
        for dataset_path in dataset_paths:
            dataset_label_names = []

            if self.args.add_null_class:
                dataset_label_names.append(const.NULL_CLASS_LABEL)

            for name in sorted([c.name for c in dataset_path.glob("*")]):
                if name[0] != "_":
                    dataset_label_names.append(name)
            tmp_label_names.append(dataset_label_names)

        assert len(set(map(tuple, tmp_label_names))) == 1, "Different labels for each sub-dataset directory"

        name_labels = tmp_label_names[0]
        num_labels = len(name_labels)
        assert num_labels > 0, f"There're no label directories in {dataset_paths}"
        return name_labels, num_labels

    def get_filenames_labels(
        self,
        dataset_paths: List[str],
    ) -> [List[str], List[str]]:
        """Get paths to all inputs and their labels.

        Args:
            dataset_paths: List of paths to datasets.

        Returns:
            filenames: List of paths to all inputs.
            labels: List of label indexes with corresponding to filenames.
        """
        if self.args.cache_dataset and self.args.cache_dataset_path is None:
            cache_directory = self.dataset_path / "_metainfo"
            cache_directory.mkdir(parents=True, exist_ok=True)
            cache_dataset_path = cache_directory / f"{self.dataset_split_name}.csv"
        else:
            cache_dataset_path = self.args.cache_dataset_path

        if self.args.cache_dataset and cache_dataset_path.exists():
            dataset_df = pd.read_csv(cache_dataset_path)

            filenames = list(dataset_df["filenames"])
            labels = list(dataset_df["labels"])
        else:
            filenames = []
            labels = []
            for label_idx, class_name in enumerate(self.label_names):
                for dataset_path in dataset_paths:
                    for class_filename in dataset_path.joinpath(class_name).glob("*"):
                        filenames.append(str(class_filename))
                        labels.append(label_idx)

            if self.args.cache_dataset:
                pd.DataFrame({
                    "filenames": filenames,
                    "labels": labels,
                }).to_csv(cache_dataset_path, index=False)

        assert len(filenames) > 0
        if self.shuffle:
            filenames, labels = self.do_shuffle(filenames, labels)

        return filenames, labels

    def do_shuffle(self, *args):
        shuffled_data = list(zip(*args))
        random.shuffle(shuffled_data)
        result = tuple(map(lambda l: list(l), zip(*shuffled_data)))

        self.log.info(colored("Data shuffled!", "red"))
        return result

    def count_samples(
        self,
        samples: List,
    ) -> int:
        """Count number of samples in dataset.

        Args:
            samples: List of samples (e.g. filenames, labels).

        Returns:
            Number of samples.
        """
        num_samples = len(samples)
        with utils.format_text("yellow", ["underline"]) as fmt:
            self.log.info(fmt(f"number of data: {num_samples}"))

        return num_samples

    def oversampling(self, data, labels):
        """Doing oversampling based on labels.
        data: list of data.
        labels: list of labels.
        """
        assert self.args.oversampling_ratio is not None, (
            "When `--do_oversampling` is set, it also needs a proper value for `--oversampling_ratio`.")

        samples_of_label = defaultdict(list)
        for sample, label in zip(data, labels):
            samples_of_label[label].append(sample)

        num_samples_of_label = {label: len(lst) for label, lst in samples_of_label.items()}
        max_num_samples = max(num_samples_of_label.values())
        min_num_samples = int(max_num_samples * self.args.oversampling_ratio)

        self.log.info(f"Log for oversampling!")
        for label, num_samples in sorted(num_samples_of_label.items()):
            # for approximation issue, let's put them at least `n` times
            n = 5
            # ratio = int(max(min_num_samples / num_samples, 1.0) * n / n + 0.5)
            ratio = int(max(min_num_samples / num_samples, 1.0) * n + 0.5)

            self.log.info(f"{label}: {num_samples} x {ratio} => {num_samples * ratio}")

            for i in range(ratio - 1):
                data.extend(samples_of_label[label])
                labels.extend(label for _ in range(num_samples))

        return data, labels

    @staticmethod
    def add_arguments(parser):
        g_common = parser.add_argument_group("(DataWrapperBase) Common Arguments for all data wrapper.")
        g_common.add_argument("--dataset_path", required=True, type=str, help="The name of the dataset to load.")
        g_common.add_argument("--dataset_split_name", required=True, type=str, nargs="*",
                              help="The name of the train/test split. Support multiple splits")
        g_common.add_argument("--no-has_sub_dataset", dest="has_sub_dataset", action="store_false")
        g_common.add_argument("--has_sub_dataset", dest="has_sub_dataset", action="store_true")
        g_common.set_defaults(has_sub_dataset=False)
        g_common.add_argument("--no-add_null_class", dest="add_null_class", action="store_false",
                              help="Support null class for idx 0")
        g_common.add_argument("--add_null_class", dest="add_null_class", action="store_true")
        g_common.set_defaults(add_null_class=True)

        g_common.add_argument("--batch_size", default=32, type=utils.positive_int,
                              help="The number of examples in batch.")
        g_common.add_argument("--no-shuffle", dest="shuffle", action="store_false")
        g_common.add_argument("--shuffle", dest="shuffle", action="store_true")
        g_common.set_defaults(shuffle=True)

        g_common.add_argument("--cache_dataset", dest="cache_dataset", action="store_true",
                              help=("If True generates/loads csv file with paths to all inputs. "
                                    "It accelerates loading of large datasets."))
        g_common.add_argument("--no-cache_dataset", dest="cache_dataset", action="store_false")
        g_common.set_defaults(cache_dataset=False)
        g_common.add_argument("--cache_dataset_path", default=None, type=lambda p: Path(p),
                              help=("Path to cached csv files containing paths to all inputs. "
                                    "If not given, csv file will be generated in the "
                                    "root data directory. This argument is used only if"
                                    "--cache_dataset is used."))

        g_common.add_argument("--width", type=int, default=-1)
        g_common.add_argument("--height", type=int, default=-1)
        g_common.add_argument("--augmentation_method", type=str, required=True,
                              choices=_available_augmentation_methods)
        g_common.add_argument("--num_threads", default=8, type=int,
                              help="We recommend using the number of available CPU cores for its value.")
        g_common.add_argument("--buffer_size", default=1000, type=int)
        g_common.add_argument("--prefetch_factor", default=100, type=int)
