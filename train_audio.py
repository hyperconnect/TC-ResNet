import argparse
from typing import List

import tensorflow as tf

import const
import factory.audio_nets as audio_nets
import common.utils as utils
from datasets.data_wrapper_base import DataWrapperBase
from datasets.audio_data_wrapper import AudioDataWrapper
from datasets.audio_data_wrapper import SingleLabelAudioDataWrapper
from helper.base import Base
from helper.trainer import TrainerBase
from helper.trainer import SingleLabelAudioTrainer
from factory.base import TFModel
from metrics.base import MetricManagerBase


def train(args):
    is_training = True
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

    trainer = SingleLabelAudioTrainer(
        model,
        session,
        args,
        dataset,
        dataset_name,
    )

    trainer.train()


def parse_arguments(arguments: List[str]=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(title="Model", description="")

    TFModel.add_arguments(parser)
    audio_nets.AudioNetModel.add_arguments(parser)

    for class_name in audio_nets._available_nets:
        subparser = subparsers.add_parser(class_name)
        subparser.add_argument("--model", default=class_name, type=str, help="DO NOT FIX ME")
        add_audio_net_arguments = eval(f"audio_nets.{class_name}.add_arguments")
        add_audio_net_arguments(subparser)

    DataWrapperBase.add_arguments(parser)
    AudioDataWrapper.add_arguments(parser)
    Base.add_arguments(parser)
    TrainerBase.add_arguments(parser)
    SingleLabelAudioTrainer.add_arguments(parser)
    MetricManagerBase.add_arguments(parser)

    args = parser.parse_args(arguments)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    log = utils.get_logger("Trainer")

    utils.update_train_dir(args)

    log.info(args)
    train(args)
