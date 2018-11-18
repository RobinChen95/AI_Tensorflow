# coding:utf-8
import tensorflow as tf

# 784个输入结点，输入的图片像素值，28X28，每个点是0-1之间的浮点数
INPUT_NODE = 784
# 10个输出，0-9
OUTPUT_NODE = 10
# 500个隐藏层结点
LAYER1_NODE = 500


def get_weight(shape, regularizer):
    # 随机生成参数w
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    # 如果使用正则化，则把每个w的正则化计入总loss
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    # 参数w、b、y
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
