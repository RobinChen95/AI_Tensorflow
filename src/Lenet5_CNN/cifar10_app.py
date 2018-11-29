# coding:utf-8
import numpy as np
import tensorflow as tf
from PIL import Image

import cifar10_backward
import cifar10_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        # 为输入输出占位,因为输入改了，所以需要改为四位参数的输入
        x = tf.placeholder(tf.float32, [1, cifar10_forward.IMAGE_SIZE,cifar10_forward.IMAGE_SIZE,cifar10_forward.NUM_CHANNELS])
        y = cifar10_forward.forward(x, False, None)
        # 最大值为输出结果
        preValue = tf.argmax(y, 1)

        # 实例化saver
        variable_averages = tf.train.ExponentialMovingAverage(cifar10_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 用with结构实现断点续训
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(cifar10_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 执行预测操作
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                # 没有找到ckpt时给出提示
                print("No checkpoint is found")
                return -1


# 删除了之前的灰度转换
def pre_pic(picName):
    # 打开传入的图片
    img = Image.open(picName)
    # 将img重置为28*28像素，以达到输入标准
    reIm = img.resize((32,32), Image.ANTIALIAS)
    im_arr = np.array(reIm)
    # 将矩阵整理为1行784列
    nm_arr = im_arr.reshape([1, 32,32,3])
    # 将矩阵的值变为浮点数，因为该接口要求浮点数
    nm_arr = nm_arr.astype(np.float32)
    # 将1-255的浮点数变为0-1之间的浮点数
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)
    # 整理好的待识别图片
    return img_ready


def application():
    # 输入的图片
    testNum = input("input the number of the pictures:")
    # 验证testNum次
    for i in range(int(testNum)):
        # 在Python3当中应该将此改为input
        testPic = raw_input("the path of test picture:")
        # 喂入图片
        testPicArr = pre_pic(testPic)
        # 重现计算图
        preValue = restore_model(testPicArr)
        print("The prediction number is :", preValue)


def main():
    application()


if __name__ == '__main__'"":
    main()
