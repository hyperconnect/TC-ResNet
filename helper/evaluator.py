import csv
import sys
from pathlib import Path
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import common.tf_utils as tf_utils
import metrics.manager as metric_manager
from common.model_loader import Ckpt
from common.utils import format_text
from common.utils import get_logger
from helper.base import AudioBase
from metrics.summaries import BaseSummaries
from metrics.summaries import Summaries


class Evaluator(object):
    def __init__(self, model, session, args, dataset, dataset_name, name):
        self.log = get_logger(name)

        self.model = model
        self.session = session
        self.args = args
        self.dataset = dataset
        self.dataset_name = dataset_name

        if Path(self.args.checkpoint_path).is_dir():
            latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_path)
            if latest_checkpoint is not None:
                self.args.checkpoint_path = latest_checkpoint
            self.log.info(f"Get latest checkpoint and update to it: {self.args.checkpoint_path}")

        self.watch_path = self._build_watch_path()

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        self.ckpt_loader = Ckpt(
            session=session,
            include_scopes=args.checkpoint_include_scopes,
            exclude_scopes=args.checkpoint_exclude_scopes,
            ignore_missing_vars=args.ignore_missing_vars,
            use_ema=self.args.use_ema,
            ema_decay=self.args.ema_decay,
        )

    @abstractmethod
    def setup_metric_manager(self):
        raise NotImplementedError

    @abstractmethod
    def setup_metric_ops(self):
        raise NotImplementedError

    @abstractmethod
    def build_non_tensor_data_from_eval_dict(self, eval_dict, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def setup_dataset_iterator(self):
        raise NotImplementedError

    def _build_watch_path(self):
        if Path(self.args.checkpoint_path).is_dir():
            return Path(self.args.checkpoint_path)
        else:
            return Path(self.args.checkpoint_path).parent

    def build_evaluation_step(self, checkpoint_path):
        if "-" in checkpoint_path and checkpoint_path.split("-")[-1].isdigit():
            return int(checkpoint_path.split("-")[-1])
        else:
            return 0

    def build_checkpoint_paths(self, checkpoint_path):
        checkpoint_glob = Path(checkpoint_path + "*")
        checkpoint_path = Path(checkpoint_path)

        return checkpoint_glob, checkpoint_path

    def build_miscellaneous_path(self, name):
        target_dir = self.watch_path / "miscellaneous" / self.dataset_name / name

        if not target_dir.exists():
            target_dir.mkdir(parents=True)

        return target_dir

    def setup_best_keeper(self):
        metric_with_modes = self.metric_manager.get_best_keep_metric_with_modes()
        self.log.debug(metric_with_modes)
        self.best_keeper = tf_utils.BestKeeper(
            metric_with_modes,
            self.dataset_name,
            self.watch_path,
            self.log,
        )

    def evaluate_once(self, checkpoint_path):
        self.log.info("Evaluation started")
        self.setup_dataset_iterator()
        self.ckpt_loader.load(checkpoint_path)

        step = self.build_evaluation_step(checkpoint_path)
        checkpoint_glob, checkpoint_path = self.build_checkpoint_paths(checkpoint_path)
        self.session.run(tf.local_variables_initializer())

        eval_metric_dict = self.run_evaluation(step, is_training=False)
        best_keep_metric_dict = self.metric_manager.filter_best_keep_metric(eval_metric_dict)
        is_keep, metrics_keep = self.best_keeper.monitor(self.dataset_name, best_keep_metric_dict)

        if self.args.save_best_keeper:
            meta_info = {
                "step": step,
                "model_size": self.model.total_params,
            }
            self.best_keeper.remove_old_best(self.dataset_name, metrics_keep)
            self.best_keeper.save_best(self.dataset_name, metrics_keep, checkpoint_glob)
            self.best_keeper.remove_temp_dir()
            self.best_keeper.save_scores(self.dataset_name, metrics_keep, best_keep_metric_dict, meta_info)

        self.metric_manager.write_evaluation_summaries(step=step,
                                                       collection_keys=[BaseSummaries.KEY_TYPES.DEFAULT])
        self.metric_manager.log_metrics(step=step)

        self.log.info("Evaluation finished")

        if step >= self.args.max_step_from_restore:
            self.log.info("Evaluation stopped")
            sys.exit()

    def build_train_directory(self):
        if Path(self.args.checkpoint_path).is_dir():
            return str(self.args.checkpoint_path)
        else:
            return str(Path(self.args.checkpoint_path).parent)

    @staticmethod
    def add_arguments(parser):
        g = parser.add_argument_group("(Evaluator) arguments")

        g.add_argument("--valid_type", default="loop", type=str, choices=["loop", "once"])
        g.add_argument("--max_outputs", default=5, type=int)

        g.add_argument("--maximum_num_labels_for_metric", default=10, type=int,
                       help="Maximum number of labels for using class-specific metrics(e.g. precision/recall/f1score)")

        g.add_argument("--no-save_best_keeper", dest="save_best_keeper", action="store_false")
        g.add_argument("--save_best_keeper", dest="save_best_keeper", action="store_true")
        g.set_defaults(save_best_keeper=True)

        g.add_argument("--no-flatten_output", dest="flatten_output", action="store_false")
        g.add_argument("--flatten_output", dest="flatten_output", action="store_true")
        g.set_defaults(flatten_output=False)

        g.add_argument("--max_step_from_restore", default=1e20, type=int)


class SingleLabelAudioEvaluator(Evaluator, AudioBase):

    def __init__(self, model, session, args, dataset, dataset_name):
        super().__init__(model, session, args, dataset, dataset_name, "SingleLabelAudioEvaluator")
        self.setup_dataset_related_attr()
        self.setup_metric_manager()
        self.setup_metric_ops()
        self.setup_best_keeper()

    def setup_dataset_related_attr(self):
        assert len(self.dataset.label_names) == self.args.num_classes
        self.use_class_metrics = len(self.dataset.label_names) < self.args.maximum_num_labels_for_metric

    def setup_metric_manager(self):
        self.metric_manager = metric_manager.AudioMetricManager(
            is_training=False,
            use_class_metrics=self.use_class_metrics,
            exclude_metric_names=self.args.exclude_metric_names,
            summary=Summaries(
                session=self.session,
                train_dir=self.build_train_directory(),
                is_training=False,
                base_name=self.dataset.dataset_split_name,
                max_summary_outputs=self.args.max_summary_outputs,
            ),
        )

    def setup_metric_ops(self):
        losses = self.build_basic_loss_ops()
        self.metric_tf_op = self.metric_manager.build_metric_ops({
            "dataset_split_name": self.dataset_name,
            "label_names": self.dataset.label_names,
            "losses": losses,
            "learning_rate": None,
            "wavs": self.model.audio_original,
        })

    def build_non_tensor_data_from_eval_dict(self, eval_dict, **kwargs):
        return {
            "dataset_split_name": self.dataset.dataset_split_name,
            "label_names": self.dataset.label_names,
            "predictions_onehot": eval_dict["predictions_onehot"],
            "labels_onehot": eval_dict["labels_onehot"],
        }

    def setup_dataset_iterator(self):
        self.dataset.setup_iterator(
            self.session,
            self.dataset.placeholders,
            self.dataset.data,
        )
