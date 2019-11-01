# coding:utf-8

import tensorflow as tf
from generateds import AGE
from tensorflow.contrib.layers import convolution2d, max_pool2d, fully_connected, batch_norm

'''
对于神经网络的参数按照功能分为了三种：
可训练参数：又称参数，是神经网络中用于从输入中提取特征的一系列参数，如w，b
结构化参数：用于构建网络结构的参数，如每一层的节点数，RNN中的步长等
超参数：是用于控制机器学习或说是调整（可训练）参数的方向的参数，如学习率等
'''
# 结构化参数
IMAGE_SIZE = 227
NUM_CHANNELS = 3
CONV1_SIZE = 3
if AGE:
    OUTPUT_NODE = 8
else:
    OUTPUT_NODE = 2


def get_weights(shape, regularizer):
    '''
    tf.add_to_collection：add_to_collection可以简单地认为Graph下维护了一个字典，key为name,value为list，而add_to_collection就是把变量添加到对应key下的list中。其第一个参数为name（即key），其第二个参数为要添加的元素
    tf.contrib.layers.l2_regularizer：L2正则化方法
    '''
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 网络结构
def forward(x, train, regularizer=0.0005):
    batch_norm_params = {
        "is_training": train,
        "trainable": True,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }

    weights_regularizer = tf.contrib.layers.l2_regularizer(regularizer)
    with tf.variable_scope("forward", "forward", [x]) as scope:
        with tf.contrib.slim.arg_scope(
                [convolution2d, fully_connected],
                weights_regularizer=weights_regularizer,
                biases_initializer=tf.constant_initializer(1.),
                weights_initializer=tf.random_normal_initializer(stddev=0.005),
                trainable=True):
            with tf.contrib.slim.arg_scope(
                    [convolution2d],
                    weights_initializer=tf.random_normal_initializer(stddev=0.01),
                    normalizer_fn=batch_norm,
                    normalizer_params=batch_norm_params):
                # layer1
                conv1 = convolution2d(x, 96, [7, 7], [4, 4], padding='VALID',
                                      biases_initializer=tf.constant_initializer(0.), scope='conv1')
                pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')

                # layer2
                conv2 = convolution2d(pool1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2')
                pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')

                # layer3
                conv3 = convolution2d(pool2, 384, [3, 3], [1, 1], biases_initializer=tf.constant_initializer(0.),
                                      padding='SAME', scope='conv3')
                pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')

                # layer4
                pool_shape = pool3.get_shape().as_list()
                nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
                flat = tf.reshape(pool3, [pool_shape[0], nodes], name='reshape')

                # layer5
                full1 = fully_connected(flat, 512, scope='full1')
                if train:
                    full1 = tf.nn.dropout(full1, 0.5)

                # layer6
                full2 = fully_connected(full1, 512, scope='full2')
                if train:
                    full2 = tf.nn.dropout(full2, 0.5)

    # layer output
    with tf.variable_scope('output') as scope:
        weights = tf.Variable(tf.random_normal([512, OUTPUT_NODE], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE], dtype=tf.float32), name='biases')
        y = tf.add(tf.matmul(full2, weights), biases, name=scope.name)
    return y
