import argparse

import tensorflow as tf

from factory.base import TFModel
import factory.audio_nets as audio_nets
from helper.base import Base
from helper.evaluator import Evaluator
from helper.evaluator import SingleLabelAudioEvaluator
from datasets.audio_data_wrapper import AudioDataWrapper
from datasets.audio_data_wrapper import SingleLabelAudioDataWrapper
from datasets.audio_data_wrapper import DataWrapperBase
from metrics.base import MetricManagerBase
from common.tf_utils import ckpt_iterator
import common.utils as utils
import const


def main(args):
    is_training = False
    dataset_name = args.dataset_split_name[0]
    session = tf.Session(config=const.TF_SESSION_CONFIG)

    dataset = SingleLabelAudioDataWrapper(
        args,
        session,
        dataset_name,
        is_training,
    )
    wavs, labels = dataset.get_input_and_output_op()

    model = eval(f"audio_nets.{args.model}")(args, dataset)
    model.build(wavs=wavs, labels=labels, is_training=is_training)

    dataset_name = args.dataset_split_name[0]
    evaluator = SingleLabelAudioEvaluator(
        model,
        session,
        args,
        dataset,
        dataset_name,
    )
    log = utils.get_logger("EvaluateAudio")

    if args.valid_type == "once":
        evaluator.evaluate_once(args.checkpoint_path)
    elif args.valid_type == "loop":
        log.info(f"Start Loop: watching {evaluator.watch_path}")

        kwargs = {
            "min_interval_secs": 0,
            "timeout": None,
            "timeout_fn": None,
            "logger": log,
        }

        for ckpt_path in ckpt_iterator(evaluator.watch_path, **kwargs):
            log.info(f"[watch] {ckpt_path}")

            evaluator.evaluate_once(ckpt_path)
    else:
        raise ValueError(f"Undefined valid_type: {args.valid_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(title="Model", description="")

    Base.add_arguments(parser)
    Evaluator.add_arguments(parser)
    DataWrapperBase.add_arguments(parser)
    AudioDataWrapper.add_arguments(parser)
    TFModel.add_arguments(parser)
    audio_nets.AudioNetModel.add_arguments(parser)
    MetricManagerBase.add_arguments(parser)

    for class_name in audio_nets._available_nets:
        subparser = subparsers.add_parser(class_name)
        subparser.add_argument("--model", default=class_name, type=str, help="DO NOT FIX ME")
        add_audio_arguments = eval("audio_nets.{}.add_arguments".format(class_name))
        add_audio_arguments(subparser)

    args = parser.parse_args()

    log = utils.get_logger("AudioNetEvaluate")
    log.info(args)
    main(args)
