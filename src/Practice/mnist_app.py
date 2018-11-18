# coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        # 为输入输出占位
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        # 最大值为输出结果
        preValue = tf.argmax(y, 1)

        # 实例化saver
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 用with结构实现断点续训
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 执行预测操作
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                # 没有找到ckpt时给出提示
                print("No checkpoint is found")
                return -1


def pre_pic(picName):
    # 打开传入的图片
    img = Image.open(picName)
    # 将img重置为28*28像素，以达到输入标准
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # reIm.convert('L')表示将图片转为灰度图，np.array将其转换为矩阵的形式
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    # 将每个像素变反
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            # 将每个像素变为0与1两个值，以去除噪声
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    # 将矩阵整理为1行784列
    nm_arr = im_arr.reshape([1, 784])
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
        testPic = input("the path of test picture:")
        # 喂入图片
        testPicArr = pre_pic(testPic)
        # 重现计算图
        preValue = restore_model(testPicArr)
        print("The prediction number is :", preValue)


def main():
    application()


if __name__ == '__main__'"":
    main()
