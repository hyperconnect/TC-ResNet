import sys
import time
from pathlib import Path
from typing import Dict
from contextlib import contextmanager
from abc import ABC
from abc import abstractmethod

import humanfriendly as hf
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from termcolor import colored

import common.tf_utils as tf_utils
import common.utils as utils
import metrics.manager as metric_manager
from helper.base import AudioBase
from common.model_loader import Ckpt
from metrics.summaries import BaseSummaries
from metrics.summaries import Summaries


class TrainerBase(ABC):
    """method prefix naming convention

    If method starts with `setup_`, it returns nothing and internally sets some fields (not recommended)
    If method starts with `build_`, it returns something and doesn't internally set any fields
    If method starts with `routine_`, it returns nothing and doesn't internally set any fields
    """
    def __init__(self, model, session, args, dataset, dataset_name, name):
        self.model = model
        self.session = session
        self.args = args
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.log = utils.get_logger(name)
        self.timer = utils.Timer(self.log)

        self.info_red = utils.format_log(self.log.info, "red")
        self.info_cyan = utils.format_log(self.log.info, "cyan")
        self.info_magenta = utils.format_log(self.log.info, "magenta")
        self.info_magenta_reverse = utils.format_log(self.log.info, "magenta", attrs=["reverse"])
        self.info_cyan_underline = utils.format_log(self.log.info, "cyan", attrs=["underline"])
        self._saver = None
        self.input_shape = self.model.audio.get_shape().as_list()

        # used in `log_step_message` method
        self.last_loss = dict()

    @property
    def etc_fetch_namespace(self):
        return utils.MLNamespace(
            step_op="step_op",
            global_step="global_step",
            summary="summary",
        )

    @property
    def loss_fetch_namespace(self):
        return utils.MLNamespace(
            total_loss="total_loss",
            model_loss="model_loss",
        )

    @property
    def summary_fetch_namespace(self):
        return utils.MLNamespace(
            merged_summaries="merged_summaries",
            merged_verbose_summaries="merged_verbose_summaries",
            merged_first_n_summaries="merged_first_n_summaries",
        )

    @property
    def after_fetch_namespace(self):
        return utils.MLNamespace(
            single_step="single_step",
            single_step_per_instance="single_step_per_instance",
        )

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)
        return self._saver

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
    def build_evaluate_iterations(self, iters: int):
        raise NotImplementedError

    def routine_experiment_summary(self):
        self.metric_manager.summary.setup_experiment(self.args)

    def setup_essentials(self, max_to_keep=5):
        self.no_op = tf.no_op()
        self.args.checkpoint_path = tf_utils.resolve_checkpoint_path(
            self.args.checkpoint_path, self.log, is_training=True
        )
        self.train_dir_name = (Path.cwd() / Path(self.args.train_dir)).resolve()

        # We use this global step for shift boundaries for piecewise_constant learning rate
        # We cannot use global step from checkpoint file before restore from checkpoint
        # For restoring, we needs to initialize all operations including optimizer
        self.global_step_from_checkpoint = tf_utils.get_global_step_from_checkpoint(self.args.checkpoint_path)
        self.global_step = tf.Variable(self.global_step_from_checkpoint, name="global_step", trainable=False)

        if self.args.boundaries_epoch:
            boundaries = [b * self.dataset.num_samples // self.dataset.batch_size for b in self.args.boundaries]
        else:
            boundaries = self.args.boundaries

        if self.args.relative:
            self.boundaries = [self.global_step_from_checkpoint + b for b in boundaries]
            self.logger.info(colored(
                (f"global_step starts with {self.global_step_from_checkpoint},"
                 f" so, update boundaries {boundaries} to {self.boundaries}"),
                "yellow",
                attrs=["underline"]))
        else:
            self.boundaries = boundaries

        self.learning_rate_placeholder = tf.train.piecewise_constant(
            self.global_step, self.boundaries, self.args.lr_list
        )

    def routine_restore_and_initialize(self, checkpoint_path=None):
        """Read various loading methods for tensorflow
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py#L121
        """
        if checkpoint_path is None:
            checkpoint_path = self.args.checkpoint_path
        var_names_to_values = getattr(self.model, "var_names_to_values", None)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())  # for metrics

        if var_names_to_values is not None:
            init_assign_op, init_feed_dict = slim.assign_from_values(var_names_to_values)
            # Create an initial assignment function.
            self.session.run(init_assign_op, init_feed_dict)
            self.log.info(colored("Restore from Memory(usually weights from caffe!)",
                                  "cyan", attrs=["bold", "underline"]))
        elif checkpoint_path == "" or checkpoint_path is None:
            self.log.info(colored("Initialize global / local variables", "cyan", attrs=["bold", "underline"]))
        else:
            ckpt_loader = Ckpt(
                session=self.session,
                include_scopes=self.args.checkpoint_include_scopes,
                exclude_scopes=self.args.checkpoint_exclude_scopes,
                ignore_missing_vars=self.args.ignore_missing_vars,
            )
            ckpt_loader.load(checkpoint_path)

    def routine_logging_checkpoint_path(self):
        self.log.info(colored("Watch Validation Through TensorBoard !", "yellow", attrs=["underline", "bold"]))
        self.log.info(colored("--checkpoint_path {}".format(self.train_dir_name),
                              "yellow", attrs=["underline", "bold"]))

    def build_optimizer(self, optimizer, learning_rate, momentum=None, decay=None, epsilon=None):
        kwargs = {
            "learning_rate": learning_rate
        }
        if momentum:
            kwargs["momentum"] = momentum
        if decay:
            kwargs["decay"] = decay
        if epsilon:
            kwargs["epsilon"] = epsilon

        if optimizer == "gd":
            opt = tf.train.GradientDescentOptimizer(**kwargs)
            self.log.info("Use GradientDescentOptimizer")
        elif optimizer == "adam":
            opt = tf.train.AdamOptimizer(**kwargs)
            self.log.info("Use AdamOptimizer")
        elif optimizer == "mom":
            opt = tf.train.MomentumOptimizer(**kwargs)
            self.log.info("Use MomentumOptimizer")
        elif optimizer == "rmsprop":
            opt = tf.train.RMSPropOptimizer(**kwargs)
            self.log.info("Use RMSPropOptimizer")
        else:
            self.log.error("Unknown optimizer: {}".format(optimizer))
            raise NotImplementedError
        return opt

    def build_train_op(self, total_loss, optimizer, trainable_scopes, global_step, gradient_multipliers=None):
        # If you use `slim.batch_norm`, then you should include train_op in slim.
        # https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-280325584
        variables_to_train = tf_utils.get_variables_to_train(trainable_scopes, logger=self.log)

        if variables_to_train:
            train_op = slim.learning.create_train_op(
                total_loss,
                optimizer,
                global_step=global_step,
                variables_to_train=variables_to_train,
                gradient_multipliers=gradient_multipliers,
            )

            if self.args.use_ema:
                self.ema = tf.train.ExponentialMovingAverage(decay=self.args.ema_decay)

                with tf.control_dependencies([train_op]):
                    train_op = self.ema.apply(variables_to_train)
        else:
            self.log.info("Empty variables_to_train")
            train_op = tf.no_op()

        return train_op

    def build_epoch(self, step):
        return (step * self.dataset.batch_size) / self.dataset.num_samples

    def build_info_step_message(self, info: Dict, float_format, delimiter: str=" / "):
        keys = list(info.keys())
        desc = delimiter.join(keys)
        val = delimiter.join([str(float_format.format(info[k])) for k in keys])
        return desc, val

    def build_duration_step_message(self, header: Dict, delimiter: str=" / "):
        def convert_to_string(number):
            type_number = type(number)
            if type_number == int or type_number == np.int32 or type_number == np.int64:
                return str(f"{number:8d}")
            elif type_number == float or type_number == np.float64:
                return str(f"{number:3.3f}")
            else:
                raise TypeError("Unrecognized type of input number")

        keys = list(header.keys())
        header_desc = delimiter.join(keys)
        header_val = delimiter.join([convert_to_string(header[k]) for k in keys])

        return header_desc, header_val

    def log_evaluation(
        self, dataset_name, epoch_from_restore, step_from_restore, global_step, eval_scores,
    ):
        self.info_cyan_underline(
            f"[{dataset_name}-Evaluation] global_step / step_from_restore / epoch_from_restore: "
            f"{global_step:8d} / {step_from_restore:5d} / {epoch_from_restore:3.3f}"
        )
        self.metric_manager.log_metrics(global_step)

    def log_step_message(self, header, losses, speeds, comparative_loss, batch_size, is_training, tag=""):
        def get_loss_color(old_loss: float, new_loss: float):
            if old_loss < new_loss:
                return "red"
            else:
                return "green"

        def get_log_color(is_training: bool):
            if is_training:
                return {"color": "blue",
                        "attrs": ["bold"]}
            else:
                return {"color": "yellow",
                        "attrs": ["underline"]}

        self.last_loss.setdefault(tag, comparative_loss)
        loss_color = get_loss_color(self.last_loss.get(tag, 0), comparative_loss)
        self.last_loss[tag] = comparative_loss

        model_size = hf.format_size(self.model.total_params*4)
        total_params = hf.format_number(self.model.total_params)

        loss_desc, loss_val = self.build_info_step_message(losses, "{:7.4f}")
        header_desc, header_val = self.build_duration_step_message(header)
        speed_desc, speed_val = self.build_info_step_message(speeds, "{:4.0f}")

        with utils.format_text(loss_color) as fmt:
            loss_val_colored = fmt(loss_val)
            msg = (
                f"[{tag}] {header_desc}: {header_val}\t"
                f"{speed_desc}: {speed_val} ({self.input_shape};{batch_size})\t"
                f"{loss_desc}: {loss_val_colored} "
                f"| {model_size} {total_params}")

            with utils.format_text(**get_log_color(is_training)) as fmt:
                self.log.info(fmt(msg))

    def setup_trainer(self):
        self.setup_essentials(self.args.max_to_keep)
        self.optimizer = self.build_optimizer(self.args.optimizer,
                                              learning_rate=self.learning_rate_placeholder,
                                              momentum=self.args.momentum,
                                              decay=self.args.optimizer_decay,
                                              epsilon=self.args.optimizer_epsilon)
        self.train_op = self.build_train_op(total_loss=self.model.total_loss,
                                            optimizer=self.optimizer,
                                            trainable_scopes=self.args.trainable_scopes,
                                            global_step=self.global_step)
        self.routine_restore_and_initialize()

    def append_if_value_is_not_none(self, key_and_op, fetch_ops):
        if key_and_op[1] is not None:
            fetch_ops.append(key_and_op)

    def run_single_step(self, fetch_ops: Dict, feed_dict: Dict=None):
        st = time.time()
        fetch_vals = self.session.run(fetch_ops, feed_dict=feed_dict)
        step_time = (time.time() - st) * 1000
        step_time_per_instance = step_time / self.dataset.batch_size

        fetch_vals[self.after_fetch_namespace.single_step] = step_time
        fetch_vals[self.after_fetch_namespace.single_step_per_instance] = step_time_per_instance

        return fetch_vals

    def log_summaries(self, fetch_vals):
        summary_keys = (
            set(fetch_vals.keys()) -
            set(self.loss_fetch_namespace.unordered_values()) -
            set(self.etc_fetch_namespace.unordered_values()) -
            set(self.after_fetch_namespace.unordered_values())
        )
        if len(summary_keys) > 0:
            self.info_magenta(f"Above step includes saving {summary_keys} summaries to {self.train_dir_name}")

    def run_with_logging(self, summary_op, metric_op_dict, feed_dict):
        fetch_ops = {
            self.etc_fetch_namespace.step_op: self.train_op,
            self.etc_fetch_namespace.global_step: self.global_step,
            self.loss_fetch_namespace.total_loss: self.model.total_loss,
            self.loss_fetch_namespace.model_loss: self.model.model_loss,
        }
        if summary_op is not None:
            fetch_ops.update({self.etc_fetch_namespace.summary: summary_op})
        if metric_op_dict is not None:
            fetch_ops.update(metric_op_dict)

        fetch_vals = self.run_single_step(fetch_ops=fetch_ops, feed_dict=feed_dict)

        global_step = fetch_vals[self.etc_fetch_namespace.global_step]
        step_from_restore = global_step - self.global_step_from_checkpoint
        epoch_from_restore = self.build_epoch(step_from_restore)

        self.log_step_message(
            {"GlobalStep": global_step,
             "StepFromRestore": step_from_restore,
             "EpochFromRestore": epoch_from_restore},
            {"TotalLoss": fetch_vals[self.loss_fetch_namespace.total_loss],
             "ModelLoss": fetch_vals[self.loss_fetch_namespace.model_loss]},
            {"SingleStepPerInstance(ms)": fetch_vals[self.after_fetch_namespace.single_step_per_instance],
             "SingleStep(ms)": fetch_vals[self.after_fetch_namespace.single_step]},
            comparative_loss=fetch_vals[self.loss_fetch_namespace.total_loss],
            batch_size=self.dataset.batch_size,
            tag=self.dataset_name,
            is_training=True
        )

        return fetch_vals, global_step, step_from_restore, epoch_from_restore

    def train(self, name: str="Training"):
        self.log.info(f"{name} started")

        global_step, step_from_restore, epoch_from_restore = 0, 0, 0
        while True:
            try:
                feed_dict = self.get_feed_dict(is_training=True)
                valid_collection_keys = []

                # collect valid collection keys
                if step_from_restore >= self.args.step_min_summaries and \
                        step_from_restore % self.args.step_save_summaries == 0:
                    valid_collection_keys.append(BaseSummaries.KEY_TYPES.DEFAULT)

                if step_from_restore % self.args.step_save_verbose_summaries == 0:
                    valid_collection_keys.append(BaseSummaries.KEY_TYPES.VERBOSE)

                if step_from_restore <= self.args.step_save_first_n_summaries:
                    valid_collection_keys.append(BaseSummaries.KEY_TYPES.FIRST_N)

                # merge it to single one
                summary_op = self.metric_manager.summary.get_merged_summaries(
                    collection_key_suffixes=valid_collection_keys,
                    is_tensor_summary=True
                )

                # send metric op
                # run it only when evaluate
                if step_from_restore % self.args.step_evaluation == 0:
                    metric_op_dict = self.metric_tf_op
                else:
                    metric_op_dict = None

                # Session.Run!
                fetch_vals, global_step, step_from_restore, epoch_from_restore = self.run_with_logging(
                    summary_op, metric_op_dict, feed_dict)
                self.log_summaries(fetch_vals)

                # Save
                if step_from_restore % self.args.step_save_checkpoint == 0:
                    with self.timer(f"save checkpoint: {self.train_dir_name}", self.info_magenta_reverse):
                        self.saver.save(self.session,
                                        str(Path(self.args.train_dir) / self.args.model),
                                        global_step=global_step)
                        if self.args.write_pbtxt:
                            tf.train.write_graph(
                                self.session.graph_def, self.args.train_dir, self.args.model + ".pbtxt"
                            )

                if step_from_restore % self.args.step_evaluation == 0:
                    self.evaluate(epoch_from_restore, step_from_restore, global_step, self.dataset_name)

                if epoch_from_restore >= self.args.max_epoch_from_restore:
                    self.info_red(f"Reached {self.args.max_epoch_from_restore} epochs from restore.")
                    break

                if step_from_restore >= self.args.max_step_from_restore:
                    self.info_red(f"Reached {self.args.max_step_from_restore} steps from restore.")
                    break

                global_step += 1
                step_from_restore = global_step - self.global_step_from_checkpoint
                epoch_from_restore = self.build_epoch(step_from_restore)
            except tf.errors.InvalidArgumentError as e:
                utils.format_log(self.log.error, "red")(f"Invalid instance is detected: {e}")
                continue

        self.log.info(f"{name} finished")

    def evaluate(
        self,
        epoch_from_restore: float,
        step_from_restore: int,
        global_step: int,
        dataset_name: str,
        iters: int=None,
    ):
        # calculate number of iterations
        iters = self.build_evaluate_iterations(iters)

        with self.timer(f"run_evaluation (iterations: {iters})", self.info_cyan):
            # evaluate metrics
            eval_dict = self.run_inference(global_step, iters=iters, is_training=True)

            non_tensor_data = self.build_non_tensor_data_from_eval_dict(eval_dict)

            self.metric_manager.evaluate_and_aggregate_metrics(step=global_step,
                                                               non_tensor_data=non_tensor_data,
                                                               eval_dict=eval_dict)

        self.metric_manager.write_evaluation_summaries(step=global_step,
                                                       collection_keys=[BaseSummaries.KEY_TYPES.DEFAULT])

        self.log_evaluation(dataset_name, epoch_from_restore, step_from_restore, global_step, eval_scores=None)

    @staticmethod
    def add_arguments(parser, name: str="TrainerBase"):
        g_optimize = parser.add_argument_group(f"({name}) Optimizer Arguments")
        g_optimize.add_argument("--optimizer", default="adam", type=str,
                                choices=["gd", "adam", "mom", "rmsprop"],
                                help="name of optimizer")
        g_optimize.add_argument("--momentum", default=None, type=float)
        g_optimize.add_argument("--optimizer_decay", default=None, type=float)
        g_optimize.add_argument("--optimizer_epsilon", default=None, type=float)

        g_rst = parser.add_argument_group(f"({name}) Saver(Restore) Arguments")
        g_rst.add_argument("--trainable_scopes", default="", type=str,
                           help=(
                               "Prefix scopes for training variables (comma separated)\n"
                               "Usually Logits e.g. InceptionResnetV2/Logits/Logits,InceptionResnetV2/AuxLogits/Logits"
                               "For default value, trainable_scopes='' means training 'all' variable"
                               "If you don't want to train(e.g. validation only), "
                               "you should give unmatched random string"
                           ))

        g_options = parser.add_argument_group(f"({name}) Training options(step, batch_size, path) Arguments")

        g_options.add_argument("--train_dir", required=True, type=str,
                               help="Directory where to write event logs and checkpoint.")
        g_options.add_argument("--step_save_summaries", default=10, type=int)
        g_options.add_argument("--step_save_verbose_summaries", default=2000, type=int)
        g_options.add_argument("--step_save_first_n_summaries", default=30, type=int)
        g_options.add_argument("--step_save_checkpoint", default=500, type=int)
        g_options.add_argument("--step_evaluation", default=500, type=utils.positive_int)

        g_options.add_argument("--no-write_pbtxt", dest="write_pbtxt", action="store_false")
        g_options.add_argument("--write_pbtxt", dest="write_pbtxt", action="store_true",
                               help="write_pbtxt model parameters")
        g_options.set_defaults(write_pbtxt=True)

        g_options.add_argument("--max_to_keep", default=5, type=utils.positive_int)
        g_options.add_argument("--max_outputs", default=5, type=utils.positive_int)
        g_options.add_argument("--max_epoch_from_restore", default=50000, type=float,
                               help=(
                                   "max epoch(1 epoch = whole data): "
                               ))
        g_options.add_argument("--step_min_summaries", default=0, type=int)
        g_options.add_argument("--max_step_from_restore", default=sys.maxsize, type=int,
                               help="Stop training when reaching given step value.")

        g_options.add_argument("--class_sampling_factor", default=20, type=int,
                               help="Sampling (class_sampling_factor * num_classes) data for evaluation")
        g_options.add_argument("--maximum_num_labels_for_metric", default=10, type=int,
                               help=("Maximum number of labels for using class-specific metrics"
                                     "e.g. precision/recall/f1score)"))

        g_lr = parser.add_argument_group("Learning Rate Scheduling Arguments")
        g_lr.add_argument("--learning_rate", default=1e-4, type=float, help="Initial learning rate for gradient update")

        g_lr.add_argument("--boundaries", default=[100000, 200000], type=int, nargs="*",
                                     help=("decay_method: predefined > global_step boundaries"
                                           "for piecewise_constant decay. If restoring from checkpoint, "
                                           "boundaries is shifted by checkpoint's global_step"))
        g_lr.add_argument("--boundaries_epoch", dest="boundaries_epoch", action="store_true",
                                     help="Given boundaries are implicitly considered in epoch units. "
                                     "If you want to use step boundaries, apply --no-boundaries-epoch "
                                     "argument.")
        g_lr.add_argument("--no-boundaries_epoch", dest="boundaries_epoch", action="store_false")
        g_lr.add_argument("--lr_list", default=[1e-3, 1e-4, 1e-5], type=float, nargs="*",
                                     help="decay_method: predefined > learning rate values for each region")
        g_lr.add_argument("--relative_schedule", dest="relative", action="store_true")
        g_lr.add_argument("--absolute_schedule", dest="relative", action="store_false",
                                     help="decay_method: predefined > choose whether to use the "
                                          "boundary values in relative (to predefined global step) "
                                          "or absolute manner. For example, given boundaries=[10, 20] "
                                          "and global_step_from_checkpoint=10, with --relative_schedule "
                                          "the schedule will shifted to [20, 30] in global manner "
                                          "where with --absolute_schedule the boundary remains to [10, 20] "
                                          "so the first scheudle begins immediately.")
        g_lr.set_defaults(relative=True, boundaries_epoch=True)


class SingleLabelAudioTrainer(TrainerBase, AudioBase):
    def __init__(self, model, session, args, dataset, dataset_name, name="AudioTrainer"):
        super().__init__(model, session, args, dataset, dataset_name, name)
        self.setup_dataset_related_attr()

        self.setup_trainer()
        self.setup_metric_manager()
        self.setup_metric_ops()

        self.routine_experiment_summary()
        self.routine_logging_checkpoint_path()

    def setup_dataset_related_attr(self):
        self.label_names = self.dataset.label_names
        assert len(self.label_names) == self.args.num_classes
        self.use_class_metrics = len(self.label_names) < self.args.maximum_num_labels_for_metric

    def setup_metric_ops(self):
        losses = self.build_basic_loss_ops()
        self.metric_tf_op = self.metric_manager.build_metric_ops({
            "dataset_split_name": self.dataset_name,
            "label_names": self.dataset.label_names,
            "losses": losses,
            "learning_rate": self.learning_rate_placeholder,
            "wavs": self.model.audio_original,
        })

    def setup_metric_manager(self):
        self.metric_manager = metric_manager.AudioMetricManager(
            is_training=True,
            use_class_metrics=self.use_class_metrics,
            exclude_metric_names=self.args.exclude_metric_names,
            summary=Summaries(
                session=self.session,
                train_dir=self.args.train_dir,
                is_training=True,
                max_summary_outputs=self.args.max_summary_outputs
            ),
        )

    def build_evaluate_iterations(self, iters):
        if iters is not None:
            iters = iters
        elif self.args.evaluation_iterations is not None:
            iters = self.args.evaluation_iterations
        else:
            num_classes = self.args.num_classes if self.args.num_classes else self.args.max_num_labels
            iters = max((self.args.class_sampling_factor * num_classes) // self.args.batch_size, 1)
        return iters

    def build_non_tensor_data_from_eval_dict(self, eval_dict, **kwargs):
        return {
            "dataset_split_name": self.dataset.dataset_split_name,
            "label_names": self.dataset.label_names,
            "predictions_onehot": eval_dict["predictions_onehot"],
            "labels_onehot": eval_dict["labels_onehot"],
        }

    @staticmethod
    def add_arguments(parser):
        pass
