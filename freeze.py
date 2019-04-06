from pathlib import Path
import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util

import common.model_loader as model_loader
from factory.base import TFModel
import factory.audio_nets as audio_nets
from factory.audio_nets import AudioNetModel
from factory.audio_nets import *
from helper.base import Base
import const


def freeze(args):
    graph = tf.Graph()
    with graph.as_default():
        session = tf.Session(config=const.TF_SESSION_CONFIG)

        model = eval(args.model)(args)
        input_tensors, output_tensor = model.build_deployable_model(include_preprocess=False)

        ckpt_loader = model_loader.Ckpt(
            session=session,
            include_scopes=args.checkpoint_include_scopes,
            exclude_scopes=args.checkpoint_exclude_scopes,
            ignore_missing_vars=args.ignore_missing_vars,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
        )
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        ckpt_loader.load(args.checkpoint_path)

        frozen_graph_def = graph_util.convert_variables_to_constants(
            session,
            session.graph_def,
            [output_tensor.op.name],
        )

        checkpoint_path = Path(args.checkpoint_path)
        output_raw_pb_path = checkpoint_path.parent / f"{checkpoint_path.name}.pb"
        tf.train.write_graph(frozen_graph_def,
                             str(output_raw_pb_path.parent),
                             output_raw_pb_path.name,
                             as_text=False)
        print(f"Save freezed pb : {output_raw_pb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(title="Model", description="")

    TFModel.add_arguments(parser)
    AudioNetModel.add_arguments(parser)

    for class_name in audio_nets._available_nets:
        subparser = subparsers.add_parser(class_name)
        subparser.add_argument("--model", default=class_name, type=str, help="DO NOT FIX ME")
        add_audio_net_arguments = eval(f"audio_nets.{class_name}.add_arguments")
        add_audio_net_arguments(subparser)

    Base.add_arguments(parser)

    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--channels", required=True, type=int)

    parser.add_argument("--sample_rate", type=int, default=16000, help="Expected sample rate of the wavs",)
    parser.add_argument("--clip_duration_ms", type=int)
    parser.add_argument("--window_size_ms", type=float, default=30.0, help="How long each spectrogram timeslice is.",)
    parser.add_argument("--window_stride_ms", type=float, default=30.0,
                        help="How far to move in time between spectogram timeslices.",)
    parser.add_argument("--num_mel_bins", type=int, default=64)
    parser.add_argument("--num_mfccs", type=int, default=64)
    parser.add_argument("--lower_edge_hertz", type=float, default=80.0)
    parser.add_argument("--upper_edge_hertz", type=float, default=7600.0)

    args = parser.parse_args()

    freeze(args)
