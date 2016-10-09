'''
This is the resnet structure. inference_small is the main body of the network.
test_graph is used to visualize the network on tensorboard. 
'''
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('weight_decay', 0.0001, '''scale for l2 regularization''')
tf.app.flags.DEFINE_float('fc_weight_decay', 0.0001, '''scale for fully connected layer's l2
regularization''')
BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.fc_weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def output_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True)
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer)

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def conv_bn_relu_layer(input_layer, filter_shape, stride, second_conv_residual=False,
                       relu=True):
    out_channel = filter_shape[-1]
    if second_conv_residual is False:
        filter = create_variables(name='conv', shape=filter_shape)
    else: filter = create_variables(name='conv2', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

    mean, variance = tf.nn.moments(conv_layer, axes=[0, 1, 2])

    if second_conv_residual is False:
        beta = create_variables('beta', [out_channel], initializer=tf.zeros_initializer)
        gamma = create_variables('gamma', [out_channel],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    else:
        beta = create_variables('beta2', [out_channel], initializer=tf.zeros_initializer)
        gamma = create_variables('gamma2', [out_channel],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

    bn_layer = tf.nn.batch_normalization(conv_layer, mean, variance, beta, gamma, BN_EPSILON)
    if relu:
        output = tf.nn.relu(bn_layer)
    else:
        output = bn_layer
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, second_conv_residual=False):
    in_channel = input_layer.get_shape().as_list()[-1]
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    if second_conv_residual is False:
        beta = create_variables('beta', [in_channel], initializer=tf.zeros_initializer)
        gamma = create_variables('gamma', [in_channel],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    else:
        beta = create_variables('beta2', [in_channel], initializer=tf.zeros_initializer)
        gamma = create_variables('gamma2', [in_channel],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    relu_layer = tf.nn.relu(bn_layer)

    if second_conv_residual is False:
        filter = create_variables(name='conv', shape=filter_shape)
    else: filter = create_variables(name='conv2', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel):
    input_channel = input_layer.get_shape().as_list()[-1]

    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    conv1 = conv_bn_relu_layer(input_layer, [3, 3, input_channel, output_channel], stride)
    conv2 = conv_bn_relu_layer(conv1, [3, 3, output_channel, output_channel], 1,
                               second_conv_residual=True, relu=False)

    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = tf.nn.relu(conv2 + padded_input)
    return output


def residual_block_new(input_layer, output_channel):
    input_channel = input_layer.get_shape().as_list()[-1]

    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)
    conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1,
                               second_conv_residual=True)

    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output



def inference_small(input_tensor_batch, n, reuse):
    '''
    total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    '''
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            conv1 = residual_block(layers[-1], 16)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        global_pool = tf.nn.avg_pool(layers[-1], ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1],
                                     padding='SAME')
        global_pool = tf.reshape(global_pool, [global_pool.get_shape().as_list()[0],
                                 global_pool.get_shape().as_list()[-1]])
        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10)
        layers.append(output)

    return layers[-1]

def test_graph(train_dir='logs'):
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference_small(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

