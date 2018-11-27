# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data, mnist
import mnist_forward
import mnist_backward
import mnist_generateds
# 程序间隔时间5s
TEST_INTERVAL_SECS = 5
TEST_NUM = 10000

def test(mnist):
    # 复现计算图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])

        # 前向传播计算出y的值
        y = mnist_forward.forward(x, None)
        # 实例化带滑动平均的saver对象，赋值为各自的滑动平均值
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        img_batch, label_batch = mnist_generateds.get_tfrecord(TEST_NUM, isTrain=False)

        while True:
            with tf.Session() as sess:
                # 加载ckpt
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                # 先判断是否已经由模型，如果有，恢复模型到当前会话，恢复参数
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                    coord.request_stop()
                    coord.join(threads)
                else:
                    # 提示模型没有找到
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
