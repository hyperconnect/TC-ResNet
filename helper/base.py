import time

import tensorflow as tf
import numpy as np
from abc import ABC
from abc import abstractmethod
from termcolor import colored

from common.utils import Timer
from common.utils import get_logger
from common.utils import format_log
from common.utils import format_text
from metrics.summaries import BaseSummaries


class Base(ABC):
    def __init__(self):
        self.log = get_logger("Base")
        self.timer = Timer(self.log)

    def get_feed_dict(self, is_training: bool=False):
        feed_dict = dict()
        return feed_dict

    def check_batch_size(
        self,
        batch_size: int,
        terminate: bool=False,
    ):
        if batch_size != self.args.batch_size:
            self.log.info(colored(f"Batch size: required {self.args.batch_size}, obtained {batch_size}", "red"))
            if terminate:
                raise tf.errors.OutOfRangeError(None, None, "Finished looping dataset.")

    def build_iters_from_batch_size(self, num_samples, batch_size):
        iters = self.dataset.num_samples // self.args.batch_size
        num_ignored_samples = self.dataset.num_samples % self.args.batch_size
        if num_ignored_samples > 0:
            with format_text("red", attrs=["bold"]) as fmt:
                msg = (
                    f"Number of samples cannot be divided by batch_size, "
                    f"so it ignores some data examples in evaluation: "
                    f"{self.dataset.num_samples} % {self.args.batch_size} = {num_ignored_samples}"
                )
                self.log.warning(fmt(msg))
        return iters

    @abstractmethod
    def build_evaluation_fetch_ops(self, do_eval):
        raise NotImplementedError

    def run_inference(
        self,
        global_step: int,
        iters: int=None,
        is_training: bool=False,
        do_eval: bool=True,
    ):
        """
        Return: Dict[metric_key] -> np.array
            array is stacked values for all batches
        """
        feed_dict = self.get_feed_dict(is_training=is_training)

        is_first_batch = True

        if iters is None:
            iters = self.build_iters_from_batch_size(self.dataset.num_samples, self.args.batch_size)

        # Get summary ops which should be evaluated by session.run
        # For example, segmentation task has several loss(GRAD/MAD/MSE) metrics
        # And these losses are now calculated by TensorFlow(not numpy)
        merged_tensor_type_summaries = self.metric_manager.summary.get_merged_summaries(
            collection_key_suffixes=[BaseSummaries.KEY_TYPES.DEFAULT],
            is_tensor_summary=True
        )

        fetch_ops = self.build_evaluation_fetch_ops(do_eval)

        aggregator = {key: list() for key in fetch_ops}
        aggregator.update({
            "batch_infer_time": list(),
            "unit_infer_time": list(),
        })

        for i in range(iters):
            try:
                st = time.time()

                is_running_summary = do_eval and is_first_batch and merged_tensor_type_summaries is not None
                if is_running_summary:
                    fetch_ops_with_summary = {"summary": merged_tensor_type_summaries}
                    fetch_ops_with_summary.update(fetch_ops)

                    fetch_vals = self.session.run(fetch_ops_with_summary, feed_dict=feed_dict)

                    # To avoid duplicated code of session.run, we evaluate merged_sum
                    # Because we run multiple batches within single global_step,
                    # merged_summaries can have duplicated values.
                    # So we write only when the session.run is first
                    self.metric_manager.write_tensor_summaries(global_step, fetch_vals["summary"])
                    is_first_batch = False
                else:
                    fetch_vals = self.session.run(fetch_ops, feed_dict=feed_dict)

                batch_infer_time = (time.time() - st) * 1000  # use milliseconds

                # aggregate
                for key, fetch_val in fetch_vals.items():
                    if key in aggregator:
                        aggregator[key].append(fetch_val)

                # add inference time
                aggregator["batch_infer_time"].append(batch_infer_time)
                aggregator["unit_infer_time"].append(batch_infer_time / self.args.batch_size)

            except tf.errors.OutOfRangeError:
                format_log(self.log.info, "yellow")(f"Reach end of the dataset.")
                break
            except tf.errors.InvalidArgumentError as e:
                format_log(self.log.error, "red")(f"Invalid instance is detected: {e}")
                continue

        aggregator = {k: np.vstack(v) for k, v in aggregator.items()}
        return aggregator

    def run_evaluation(
        self,
        global_step: int,
        iters: int=None,
        is_training: bool=False,
    ):
        eval_dict = self.run_inference(global_step, iters, is_training, do_eval=True)

        non_tensor_data = self.build_non_tensor_data_from_eval_dict(eval_dict, step=global_step)

        self.metric_manager.evaluate_and_aggregate_metrics(step=global_step,
                                                           non_tensor_data=non_tensor_data,
                                                           eval_dict=eval_dict)

        eval_metric_dict = self.metric_manager.get_evaluation_result(step=global_step)

        return eval_metric_dict

    @staticmethod
    def add_arguments(parser):
        g_base = parser.add_argument_group("Base")
        g_base.add_argument("--no-use_ema", dest="use_ema", action="store_false")
        g_base.add_argument("--use_ema", dest="use_ema", action="store_true",
                            help="Exponential Moving Average. It may take more memory.")
        g_base.set_defaults(use_ema=False)
        g_base.add_argument("--ema_decay", default=0.999, type=float,
                            help=("Exponential Moving Average decay.\n"
                                  "Reasonable values for decay are close to 1.0, typically "
                                  "in the multiple-nines range: 0.999, 0.9999"))
        g_base.add_argument("--evaluation_iterations", type=int, default=None)


class AudioBase(Base):
    def build_evaluation_fetch_ops(self, do_eval):
        if do_eval:
            fetch_ops = {
                "labels_onehot": self.model.labels,
                "predictions_onehot": self.model.outputs,
                "total_loss": self.model.total_loss,
            }
            fetch_ops.update(self.metric_tf_op)
        else:
            fetch_ops = {
                "predictions_onehot": self.model.outputs,
            }

        return fetch_ops

    def build_basic_loss_ops(self):
        losses = {
            "total_loss": self.model.total_loss,
            "model_loss": self.model.model_loss,
        }
        losses.update(self.model.endpoints_loss)

        return losses
