import tensorflow as tf

slim = tf.contrib.slim


def conv_relu_bn(inputs, num_outputs, kernel_size, stride, idx, use_dilation, bn=False):
    scope = f"conv{idx}"
    with tf.variable_scope(scope, values=[inputs]):
        if use_dilation:
            assert stride == 1
            rate = int(2**(idx // 3))
            net = slim.conv2d(inputs,
                              num_outputs=num_outputs,
                              kernel_size=kernel_size,
                              stride=stride,
                              rate=rate)
        else:
            net = slim.conv2d(inputs,
                              num_outputs=num_outputs,
                              kernel_size=kernel_size,
                              stride=stride)
        # conv + relu are done
    if bn:
        net = slim.batch_norm(net, scope=f"{scope}_bn")

    return net


def resnet(inputs, num_classes, num_layers, num_channels, pool_size, use_dilation, scope="Res"):
    """Re-implement https://github.com/castorini/honk/blob/master/utils/model.py"""
    endpoints = dict()

    with tf.variable_scope(scope):
        net = slim.conv2d(inputs, num_channels, kernel_size=3, stride=1, scope="f_conv")

        if pool_size:
            net = slim.avg_pool2d(net, kernel_size=pool_size, stride=1, scope="avg_pool0")

        # block
        num_blocks = num_layers // 2
        idx = 0
        for i in range(num_blocks):
            layer_in = net

            net = conv_relu_bn(net, num_outputs=num_channels, kernel_size=3, stride=1, idx=idx,
                               use_dilation=use_dilation, bn=True)
            idx += 1

            net = conv_relu_bn(net, num_outputs=num_channels, kernel_size=3, stride=1, idx=(2 * i + 1),
                               use_dilation=use_dilation, bn=False)
            idx += 1

            net += layer_in
            net = slim.batch_norm(net, scope=f"conv{2 * i + 1}_bn")

        if num_layers % 2 != 0:
            net = conv_relu_bn(net, num_outputs=num_channels, kernel_size=3, stride=1, idx=idx,
                               use_dilation=use_dilation, bn=True)

        # last
        net = slim.avg_pool2d(net, kernel_size=net.shape[1:3], stride=1, scope="avg_pool1")

        logits = slim.conv2d(net, num_classes, 1, activation_fn=None, scope="fc")
        logits = tf.reshape(logits, shape=(-1, logits.shape[3]), name="squeeze_logit")

        return logits, endpoints


def Res8(inputs, num_classes):
    return resnet(inputs,
                  num_classes,
                  num_layers=6,
                  num_channels=45,
                  pool_size=[4, 3],
                  use_dilation=False)


def Res8Narrow(inputs, num_classes):
    return resnet(inputs,
                  num_classes,
                  num_layers=6,
                  num_channels=19,
                  pool_size=[4, 3],
                  use_dilation=False)


def Res15(inputs, num_classes):
    return resnet(inputs,
                  num_classes,
                  num_layers=13,
                  num_channels=45,
                  pool_size=None,
                  use_dilation=True)


def Res15Narrow(inputs, num_classes):
    return resnet(inputs,
                  num_classes,
                  num_layers=13,
                  num_channels=19,
                  pool_size=None,
                  use_dilation=True)


def Res_arg_scope(is_training, weight_decay=0.00001):
    batch_norm_params = {
        "is_training": is_training,
        "center": False,
        "scale": False,
        "decay": 0.997,
        "fused": True,
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu,
                        biases_initializer=None,
                        normalizer_fn=None,
                        padding="SAME",
                        ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
            return scope
