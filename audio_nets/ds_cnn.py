from collections import namedtuple
from functools import partial

import tensorflow as tf

slim = tf.contrib.slim


_DEFAULT = {
    "type": None,  # ["conv", "separable"]
    "kernel": [3, 3],
    "stride": [1, 1],
    "depth": None,
    "scope": None,
}

_Block = namedtuple("Block", _DEFAULT.keys())
Block = partial(_Block, **_DEFAULT)

S_NET_DEF = [
    Block(type="conv", depth=64, kernel=[10, 4], stride=[2, 2], scope="conv_1"),
    Block(type="separable", depth=64, kernel=[3, 3], stride=[1, 1], scope="conv_ds_1"),
    Block(type="separable", depth=64, kernel=[3, 3], stride=[1, 1], scope="conv_ds_2"),
    Block(type="separable", depth=64, kernel=[3, 3], stride=[1, 1], scope="conv_ds_3"),
    Block(type="separable", depth=64, kernel=[3, 3], stride=[1, 1], scope="conv_ds_4"),
]

M_NET_DEF = [
    Block(type="conv", depth=172, kernel=[10, 4], stride=[2, 1], scope="conv_1"),
    Block(type="separable", depth=172, kernel=[3, 3], stride=[2, 2], scope="conv_ds_1"),
    Block(type="separable", depth=172, kernel=[3, 3], stride=[1, 1], scope="conv_ds_2"),
    Block(type="separable", depth=172, kernel=[3, 3], stride=[1, 1], scope="conv_ds_3"),
    Block(type="separable", depth=172, kernel=[3, 3], stride=[1, 1], scope="conv_ds_4"),
]

L_NET_DEF = [
    Block(type="conv", depth=276, kernel=[10, 4], stride=[2, 1], scope="conv_1"),
    Block(type="separable", depth=276, kernel=[3, 3], stride=[2, 2], scope="conv_ds_1"),
    Block(type="separable", depth=276, kernel=[3, 3], stride=[1, 1], scope="conv_ds_2"),
    Block(type="separable", depth=276, kernel=[3, 3], stride=[1, 1], scope="conv_ds_3"),
    Block(type="separable", depth=276, kernel=[3, 3], stride=[1, 1], scope="conv_ds_4"),
    Block(type="separable", depth=276, kernel=[3, 3], stride=[1, 1], scope="conv_ds_5"),
]


def _depthwise_separable_conv(inputs, num_pwc_filters, kernel_size, stride):
    """ Helper function to build the depth-wise separable convolution layer."""
    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  depth_multiplier=1,
                                                  kernel_size=kernel_size,
                                                  scope="depthwise_conv")

    bn = slim.batch_norm(depthwise_conv, scope="dw_batch_norm")
    pointwise_conv = slim.conv2d(bn,
                                 num_pwc_filters,
                                 kernel_size=[1, 1],
                                 scope="pointwise_conv")
    bn = slim.batch_norm(pointwise_conv, scope="pw_batch_norm")
    return bn


def parse_block(input_net, block):
    if block.type == "conv":
        net = slim.conv2d(
            input_net,
            num_outputs=block.depth,
            kernel_size=block.kernel,
            stride=block.stride,
            scope=block.scope
        )
        net = slim.batch_norm(net, scope=f"{block.scope}/batch_norm")
    elif block.type == "separable":
        with tf.variable_scope(block.scope):
            net = _depthwise_separable_conv(
                input_net,
                block.depth,
                kernel_size=block.kernel,
                stride=block.stride
            )
    else:
        raise ValueError(f"Block type {block.type} is not supported!")

    return net


def DSCNN(inputs, num_classes, net_def, scope="DSCNN"):
    endpoints = dict()

    with tf.variable_scope(scope):
        net = inputs
        for block in net_def:
            net = parse_block(net, block)

        net = slim.avg_pool2d(net, kernel_size=net.shape[1:3], stride=1, scope="avg_pool")
        net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
        logits = slim.fully_connected(net, num_classes, activation_fn=None, scope="fc1")

    return logits, endpoints


def DSCNN_arg_scope(is_training):
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.96,
        "activation_fn": tf.nn.relu,
    }

    with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                        activation_fn=None,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        biases_initializer=slim.init_ops.zeros_initializer()):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.dropout],
                                is_training=is_training) as scope:
                return scope
