# coding:utf-8
import tensorflow as tf
import cifar10_forward
import os
import cifar10_generateds

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
# 学习率衰减率
LEARNING_RATE_DECAY = 0.99
# 正则化系数
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径
MODEL_SAVE_PATH = "./model/"
# 模型保存名字
MODEL_NAME = "cifar10_model"
# 新加的
train_num_examples = 50000


def backward():
    x = tf.placeholder(tf.float32, [None, cifar10_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, cifar10_forward.OUTPUT_NODE])
    y = cifar10_forward.forward(x, REGULARIZER)
    # 轮数计数器赋初值，并标记为不可训练
    global_step = tf.Variable(0, trainable=False)

    # 使用交叉熵的协同使用实现正则化，并把参数w的正则化计入参数中
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    # 实例化Saver
    saver = tf.train.Saver()
    img_batch, label_batch = cifar10_generateds.get_tfRecord(BATCH_SIZE, isTrain=True)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 如果ckpt存在，则用restore方法把ckpt恢复到当前会话#
        # 给所有的w和b赋存在ckpt当中的值，继续上次训练
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])
            # sess.run()之后才会有结果
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)

def main():
    backward()


if __name__ == '__main__':
    main()
