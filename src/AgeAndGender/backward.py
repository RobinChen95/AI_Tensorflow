# coding:utf-8

import os

import forward
import generateds
import numpy as np
import tensorflow as tf

# 超参数
AGE = generateds.AGE
BATCH_SIZE = 400  # 批训练的数量
LR_BASE = 0.001  # 学习率的基础值
LR_DECAY = 0.1  # 学习率衰减因子
REGULARIZER = 0.0005  # L2正则化的权重衰减因子(0.0001 - 0.0010)
STEPS = 50000  # 批训练最大轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减因子
LAMBDA = 0.01  # 总损失的滑动平均的衰减因子

if AGE:
    train_num_examples = 13717  # 样例总数
    MODEL_SAVE_PATH = "./ageModel/"  # model（主要是参数，以及训练状态）保存路径
    MODEL_NAME = "age_model"  # model名称前缀
else:
    train_num_examples = 14047  # 样例总数
    MODEL_SAVE_PATH = "./genderModel/"  # model（主要是参数，以及训练状态）保存路径
    MODEL_NAME = "gender_model"  # model名称前缀


def loss_fn(y, y_):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_mean + LAMBDA * sum(regularization_losses)
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def backward():
    # 为输入和期望输出设置占位符，以及设置预测值对应的网络
    x = tf.placeholder(tf.float32, [BATCH_SIZE, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, forward.OUTPUT_NODE])
    # y_ = tf.placeholder(tf.float32, [BATCH_SIZE,1])
    y = forward.forward(x, True, REGULARIZER)
    # 全局当前批训练的轮数
    global_step = tf.Variable(0, trainable=False)

    # tf.get_collection(name):与add_to_collection相当于是一套的，add_to_collection负责设置或是为列表添加新元素。get_collection就是从全局Collection中获取一个名为name参数的列表
    # tf.add_n([p1, p2, p3....])：实现一个列表的元素的相加。有点像tf.add(),tf.add_n([p1,p2])效果上等价于tf.add(p1,p2)

    # 设置loss函数
    loss = loss_fn(y, y_)

    # 设置学习率衰减
    lr = tf.train.exponential_decay(LR_BASE, global_step, 10000, LR_DECAY, staircase=True)

    # 设置训练计划
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    # 设置参数衰减
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 设置训练的保存器，用于保存模型
    saver = tf.train.Saver()
    img_batch, label_batch = generateds.get_tfRecord(BATCH_SIZE, isTrain=True)

    with tf.Session() as sess:
        # 全局变量初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 开始训练
        for i in range(STEPS):
            # 获取样例与标签
            xs, ys = sess.run([img_batch, label_batch])
            # 转换会图片
            xs = np.reshape(xs, [BATCH_SIZE, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNELS])
            # 一次训练
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            # 每隔千轮，输出一次loss，便于直接观察模型优化趋势，同时保存一次模型。
            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            if i % 100 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)


def main():
    # 执行训练函数（反向传播中包含前向传播，因此是完整的训练过程）
    backward()


# 当该模块为运行的主模块，执行下列语句
if __name__ == '__main__':
    main()
