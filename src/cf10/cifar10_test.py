# coding:utf-8
import time

import tensorflow as tf

import cifar10_backward
import cifar10_forward
import cifar10_generateds

# 程序间隔时间10s
TEST_INTERVAL_SECS = 10
TEST_NUM = 10000


def test():
    # 复现计算图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, cifar10_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, cifar10_forward.OUTPUT_NODE])

        # 前向传播计算出y的值
        y = cifar10_forward.forward(x, None)
        # 实例化带滑动平均的saver对象，赋值为各自的滑动平均值
        ema = tf.train.ExponentialMovingAverage(cifar10_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        img_batch, label_batch = cifar10_generateds.get_tfRecord(TEST_NUM, isTrain=False)

        while True:
            with tf.Session() as sess:
                # 加载保存的模型
                ckpt = tf.train.get_checkpoint_state(cifar10_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 获取该模型的全局训练的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    # 计算精确度
                    xs, ys = sess.run([img_batch, label_batch])
                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    coord.request_stop()
                    coord.join(threads)
                else:
                    print("No CheckPoint File Found!")
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    test()


if __name__ == '__main__':
    main()
