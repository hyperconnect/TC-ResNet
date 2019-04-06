import subprocess
import shutil
from typing import Dict
from functools import reduce
from pathlib import Path
from operator import mul

import tensorflow as tf
import pandas as pd
import numpy as np
from termcolor import colored
from tensorflow.contrib.training import checkpoints_iterator
from common.utils import get_logger
from common.utils import wait

import const


def get_variables_to_train(trainable_scopes, logger):
    """Returns a list of variables to train.
    Returns:
    A list of variables to train by the optimizer.
    """
    if trainable_scopes is None or trainable_scopes == "":
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(",")]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    for var in variables_to_train:
        logger.info("vars to train > {}".format(var.name))

    return variables_to_train


def show_models(logger):
    trainable_variables = set(tf.contrib.framework.get_variables(collection=tf.GraphKeys.TRAINABLE_VARIABLES))
    all_variables = tf.contrib.framework.get_variables()
    trainable_vars = tf.trainable_variables()
    total_params = 0
    total_trainable_params = 0
    logger.info(colored(f">> Start of showing all variables", "cyan", attrs=["bold"]))
    for v in all_variables:
        is_trainable = v in trainable_variables
        count_params = reduce(mul, v.get_shape().as_list(), 1)
        total_params += count_params
        total_trainable_params += (count_params if is_trainable else 0)
        color = "cyan" if is_trainable else "green"
        logger.info(colored((
            f">>    {v.name} {v.dtype} : {v.get_shape().as_list()}, {count_params} ... {total_params} "
            f"(is_trainable: {is_trainable})"
        ), color))
    logger.info(colored(
        f">> End of showing all variables // Number of variables: {len(all_variables)}, "
        f"Number of trainable variables : {len(trainable_vars)}, "
        f"Total prod + sum of shape: {total_params} ({total_trainable_params} trainable)",
        "cyan", attrs=["bold"]))
    return total_params


def ckpt_iterator(checkpoint_dir, min_interval_secs=0, timeout=None, timeout_fn=None, logger=None):
    for ckpt_path in checkpoints_iterator(checkpoint_dir, min_interval_secs, timeout, timeout_fn):
        yield ckpt_path


class BestKeeper(object):
    def __init__(
        self,
        metric_with_modes,
        dataset_name,
        directory,
        logger=None,
        epsilon=0.00005,
        score_file="scores.tsv",
        metric_best: Dict={},
    ):
        """Keep best model's checkpoint by each datasets & metrics

        Args:
            metric_with_modes: Dict, metric_name: mode
                if mode is 'min', then it means that minimum value is best, for example loss(MSE, MAE)
                if mode is 'max', then it means that maximum value is best, for example Accuracy, Precision, Recall
            dataset_name: str, dataset name on which metric be will be calculated
            directory: directory path for saving best model
            epsilon: float, threshold for measuring the new optimum, to only focus on significant changes.
                Because sometimes early-stopping gives better generalization results
        """
        if logger is not None:
            self.log = logger
        else:
            self.log = get_logger("BestKeeper")

        self.score_file = score_file
        self.metric_best = metric_best

        self.log.info(colored(f"Initialize BestKeeper: Monitor {dataset_name} & Save to {directory}",
                              "yellow", attrs=["underline"]))
        self.log.info(f"{metric_with_modes}")

        self.x_better_than_y = {}
        self.directory = Path(directory)
        self.output_temp_dir = self.directory / f"{dataset_name}_best_keeper_temp"

        for metric_name, mode in metric_with_modes.items():
            if mode == "min":
                self.metric_best[metric_name] = self.load_metric_from_scores_tsv(
                    directory / dataset_name / metric_name / score_file,
                    metric_name,
                    np.inf,
                )
                self.x_better_than_y[metric_name] = lambda x, y: np.less(x, y - epsilon)
            elif mode == "max":
                self.metric_best[metric_name] = self.load_metric_from_scores_tsv(
                    directory / dataset_name / metric_name / score_file,
                    metric_name,
                    -np.inf,
                )
                self.x_better_than_y[metric_name] = lambda x, y: np.greater(x, y + epsilon)
            else:
                raise ValueError(f"Unsupported mode : {mode}")

    def load_metric_from_scores_tsv(
        self,
        full_path: Path,
        metric_name: str,
        default_value: float,
    ) -> float:
        def parse_scores(s: str):
            if len(s) > 0:
                return float(s)
            else:
                return default_value

        if full_path.exists():
            with open(full_path, "r") as f:
                header = f.readline().strip().split("\t")
                values = list(map(parse_scores, f.readline().strip().split("\t")))
                metric_index = header.index(metric_name)

            return values[metric_index]
        else:
            return default_value

    def monitor(self, dataset_name, eval_scores):
        metrics_keep = {}
        is_keep = False
        for metric_name, score in self.metric_best.items():
            score = eval_scores[metric_name]
            if self.x_better_than_y[metric_name](score, self.metric_best[metric_name]):
                old_score = self.metric_best[metric_name]
                self.metric_best[metric_name] = score
                metrics_keep[metric_name] = True
                is_keep = True
                self.log.info(colored("[KeepBest] {} {:.6f} -> {:.6f}, so keep it!".format(
                    metric_name, old_score, score), "blue", attrs=["underline"]))
            else:
                metrics_keep[metric_name] = False
        return is_keep, metrics_keep

    def save_best(self, dataset_name, metrics_keep, ckpt_glob):
        for metric_name, is_keep in metrics_keep.items():
            if is_keep:
                keep_path = self.directory / Path(dataset_name) / Path(metric_name)
                self.keep_checkpoint(keep_path, ckpt_glob)
                self.keep_converted_files(keep_path)

    def save_scores(self, dataset_name, metrics_keep, eval_scores, meta_info=None):
        eval_scores_with_meta = eval_scores.copy()
        if meta_info is not None:
            eval_scores_with_meta.update(meta_info)

        for metric_name, is_keep in metrics_keep.items():
            if is_keep:
                keep_path = self.directory / Path(dataset_name) / Path(metric_name)
                if not keep_path.exists():
                    keep_path.mkdir(parents=True)
                df = pd.DataFrame(pd.Series(eval_scores_with_meta)).sort_index().transpose()
                df.to_csv(keep_path / self.score_file, sep="\t", index=False, float_format="%.5f")

    def remove_old_best(self, dataset_name, metrics_keep):
        for metric_name, is_keep in metrics_keep.items():
            if is_keep:
                keep_path = self.directory / Path(dataset_name) / Path(metric_name)
                # Remove old directory to save space
                if keep_path.exists():
                    shutil.rmtree(str(keep_path))
                keep_path.mkdir(parents=True)

    def keep_checkpoint(self, keep_dir, ckpt_glob):
        if not isinstance(keep_dir, Path):
            keep_dir = Path(keep_dir)

        # .data-00000-of-00001, .meta, .index
        for ckpt_path in ckpt_glob.parent.glob(ckpt_glob.name):
            shutil.copy(str(ckpt_path), str(keep_dir))

        with open(keep_dir / "checkpoint", "w") as f:
            f.write(f'model_checkpoint_path: "{Path(ckpt_path.name).stem}"')  # noqa

    def keep_converted_files(self, keep_path):
        if not isinstance(keep_path, Path):
            keep_path = Path(keep_path)

        for path in self.output_temp_dir.glob("*"):
            if path.is_dir():
                shutil.copytree(str(path), str(keep_path / path.name))
            else:
                shutil.copy(str(path), str(keep_path / path.name))

    def remove_temp_dir(self):
        if self.output_temp_dir.exists():
            shutil.rmtree(str(self.output_temp_dir))


def resolve_checkpoint_path(checkpoint_path, log, is_training):
    if checkpoint_path is not None and Path(checkpoint_path).is_dir():
        old_ckpt_path = checkpoint_path
        checkpoint_path = tf.train.latest_checkpoint(old_ckpt_path)
        if not is_training:
            def stop_checker():
                return (tf.train.latest_checkpoint(old_ckpt_path) is not None)
            wait("There are no checkpoint file yet", stop_checker)  # wait until checkpoint occurs
        checkpoint_path = tf.train.latest_checkpoint(old_ckpt_path)
        log.info(colored(
            "self.args.checkpoint_path updated: {} -> {}".format(old_ckpt_path, checkpoint_path),
            "yellow", attrs=["bold"]))
    else:
        log.info(colored("checkpoint_path is {}".format(checkpoint_path), "yellow", attrs=["bold"]))

    return checkpoint_path


def get_global_step_from_checkpoint(checkpoint_path):
    """It is assumed that `checkpoint_path` is path to checkpoint file, not path to directory
    with checkpoint files.
    In case checkpoint path is not defined, 0 is returned."""
    if checkpoint_path is None or checkpoint_path == "":
        return 0
    else:
        if "-" in Path(checkpoint_path).stem:
            return int(Path(checkpoint_path).stem.split("-")[-1])
        else:
            return 0
