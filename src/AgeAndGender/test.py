# coding: utf-8

import backward
import forward
import generateds
import numpy as np
import tensorflow as tf

# 每轮test之间的停顿间隙
# TEST_INTERVAL_SECS = 20
if backward.AGE:
    TEST_NUM = 3200
else:
    TEST_NUM = 3445
BATCH_SIZE = 640


def test():
    '''
    tf.Graph().as_default():一个将某图设置为默认图，并返回一个上下文的管理器。如果不显式添加一个默认图，系统会自动设置一个全局的默认图。所设置的默认图，在模块范围内所定义的节点都将默认加入默认图中。
    '''
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [BATCH_SIZE, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [BATCH_SIZE, forward.OUTPUT_NODE])
        y = forward.forward(x, False)

        # EMA：滑动平均变量
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        '''
        tf.euqal:相当于tensor变量比较的“==”符号。
        tf.cast(x,dtype,name=None):将x的数据格式转化成dtype.
        '''
        current_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(current_prediction, tf.float32))
        # num_right = tf.add_n(current_prediction)
        img_batch, label_batch = generateds.get_tfRecord(BATCH_SIZE, isTrain=False)

        pos_num = 0
        index = 0
        '''
        TEST
        '''
        with tf.Session() as sess:
            # 加载模型（参数）
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 获取该模型的全局批训练的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                while index < TEST_NUM:
                    index = index + BATCH_SIZE
                    # 计算精确度
                    xs, ys = sess.run([img_batch, label_batch])
                    xs = np.reshape(xs, [BATCH_SIZE, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNELS])
                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    # accuracy_number = sess.run(num_right, feed_dict={x:xs, y_:ys})
                    pos_num = pos_num + accuracy_score * BATCH_SIZE
                coord.request_stop()
                coord.join(threads)
            else:
                print("No CheckPoint File Found!")
                return
                # time.sleep(TEST_INTERVAL_SECS)
        print("After %s training step(s), test accuracy = %g" % (global_step, pos_num * 1.0 / (index + 1e-3)))


def main():
    test()


# 当该模块为运行的主模块，执行下列语句
if __name__ == '__main__':
    main()
