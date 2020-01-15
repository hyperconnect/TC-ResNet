import tensorflow as tf

slim = tf.contrib.slim


def DpNet_arg_scope(is_training, weight_decay=0.0001, keep_prob=0.8):
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.99,
        "activation_fn": None,
    }

    with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                        activation_fn=None,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=None,
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.dropout],
                                keep_prob=keep_prob,
                                is_training=is_training) as scope:
                return scope


def dpnet(inputs, num_classes, n_channels, n_strides, n_ratios, n_layers, scope, last_channel=None):
    endpoints = dict()
    L = inputs.shape[1]
    C = inputs.shape[2]

    assert len(n_channels) == len(n_strides) + 1

    with tf.variable_scope(scope):
        inputs = tf.reshape(inputs, [-1, L, 1, C])  # [N, L, 1, C]
        first_conv_kernel = [3, 1]
        conv_kernel = [9, 1]

        net = slim.conv2d(
            inputs, num_outputs=n_channels[0], kernel_size=first_conv_kernel, stride=1, scope="conv0")

        n_channels = n_channels[1:]

        for i, n in enumerate(n_channels):
            with tf.variable_scope(f"block{i}"):
                expand_n = int(n * n_ratios[i])
                for j, channel in enumerate(range(n_layers[i])):
                    stride = n_strides[i] if j == 0 else 1
                    if stride != 1 or net.shape[-1] != n:
                        layer_in = slim.conv2d(
                            net, num_outputs=n, kernel_size=1, stride=stride, scope=f"down")
                    else:
                        layer_in = net

                    net = slim.conv2d(net,
                                      expand_n,
                                      kernel_size=[1, 1],
                                      scope=f"pointwise_conv{j}_0")
                    net = tf.nn.relu(net)
                    net = slim.separable_convolution2d(net,
                                                       num_outputs=None,
                                                       stride=stride,
                                                       depth_multiplier=1,
                                                       kernel_size=conv_kernel,
                                                       scope=f"depthwise_conv{j}")
                    net = tf.nn.relu(net)
                    net = slim.conv2d(net,
                                      n,
                                      kernel_size=[1, 1],
                                      scope=f"pointwise_conv{j}_1")

                    net += layer_in

        net = slim.avg_pool2d(
            net, kernel_size=net.shape[1:3], stride=1, scope="avg_pool")

        if last_channel is not None:
            net = slim.conv2d(net,
                              last_channel,
                              kernel_size=[1, 1],
                              scope=f"pointwise_conv")
            net = tf.nn.relu(net)

        net = slim.dropout(net)

        logits = slim.conv2d(
            net, num_classes, 1, activation_fn=None, normalizer_fn=None, scope="fc")
        logits = tf.reshape(
            logits, shape=(-1, logits.shape[3]), name="squeeze_logit")

        ranges = slim.conv2d(net, 2, 1, activation_fn=None,
                             normalizer_fn=None, scope="fc2")
        ranges = tf.reshape(
            ranges, shape=(-1, ranges.shape[3]), name="squeeze_logit2")
        endpoints["ranges"] = tf.sigmoid(ranges)

    return logits, endpoints


def DpNet1(inputs, num_classes, width_multiplier=1.0, scope="DpNet1"):
    n_channels = [24, 16, 24, 36, 48]
    n_strides = [2] * 4
    n_ratios = [3] * 4
    n_layers = [4] * 4
    n_channels = [int(x * width_multiplier) for x in n_channels]

    return dpnet(inputs, num_classes, n_channels, n_strides, n_ratios, n_layers, scope=scope)


def DpNet2(inputs, num_classes, width_multiplier=1.0, scope="DpNet2"):
    n_channels = [24, 16, 24, 36, 48]
    n_strides = [2] * 4
    n_ratios = [3] * 4
    n_layers = [4] * 4
    n_channels = [int(x * width_multiplier) for x in n_channels]
    last_channel = 128

    return dpnet(inputs, num_classes, n_channels, n_strides, n_ratios, n_layers, scope=scope, last_channel=last_channel)
