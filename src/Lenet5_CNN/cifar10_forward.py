# coding:utf-8
import tensorflow as tf

# 图片的分辨率为32X32
IMAGE_SIZE = 32
# 通道数为3
NUM_CHANNELS = 3
# 第一层卷积核边长为5X5
CONV1_SIZE = 5
# 第一层卷积核个数为32
CONV1_KERNEL_NUM = 32
# 第二层卷积核边长为5X5
CONV2_SIZE = 5
# 第二层卷积核个数为64
CONV2_KERNEL_NUM = 64
# 有512个神经元
FC_SIZE = 512
# 10个输出节点
OUTPUT_NODE = 10


# shape表示生成张量的维度
def get_weight(shape, regularizer):
    # 随机生成参数w，去掉偏离过大的随机数
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    # 如果使用正则化，则把每个w的正则化计入总loss
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, w):
    # 输入x，权重w，卷积核的滑动步长为1x1，使用零填充
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 输入x，池化核大小为2x2，行列滑动步长为2x2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建前向传播的神经网络
def forward(x, train, regularizer):
    # 初始化第一层的权重w
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    # 初始化第一层的偏置b
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # 执行卷积计算，输入是x，卷积核为conv1_w
    conv1 = conv2d(x, conv1_w)
    # 为结果添加偏置conv1_b并通过relu激活函数
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 最大池化
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # 第二层使用第一层的输出作为输入
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 将三维张量存为二维张量的list
    pool_shape = pool2.get_shape().as_list()
    # 提取特征的长度*宽度*深度，得到所有特征点的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    # 把上层的输出*本层的权重加上偏置并通过激活函数
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层的输出使用50%的dropout
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
