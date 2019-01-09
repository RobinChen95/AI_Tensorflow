# coding: utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import forward
import backward

age_table=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
sex_table=['f','m']  # f:女; m:男

# 预测函数
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        # 导入网络架构
        x = tf.placeholder(tf.float32,[1,forward.IMAGE_SIZE,forward.IMAGE_SIZE, forward.NUM_CHANNELS])
        y = forward.forward(x, False)
        
        # 获取y列表中最大值的索引，即概率最大的数值为预测值。
        preValue = tf.argmax(y,1)
        
        # 权重滑动平均
        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        # 开启会话，运行网络
        with tf.Session() as sess:
            # 获取模型路径
            ckpt = tf.train.get_checkpoint_state("./model/")
            # 导入模型并预测
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return age_table[preValue[0]]
            else:
                print("No CheckPoint File Found!")
                return -1 

# 图片预处理
def pre_pic(picName):
    # 打开图片
    img = Image.open(picName)
    # 调整图片分辨率及抗锯齿
    img = img.resize((forward.IMAGE_SIZE, forward.IMAGE_SIZE),Image.ANTIALIAS)
    # 图片转化为灰度图，并将其数据类型转化为numpy.ARRAY
    im_arr = np.array(img)
    # 图片压扁成数列
    nm_arr = im_arr.reshape([1, forward.IMAGE_SIZE, forward.IMAGE_SIZE, 3])
    # 修改数据类型，为了归一化
    nm_arr = nm_arr.astype(np.float32)
    # 归一化
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready

# API接口
def application():
    # 测试次数
    testNum = input("Input the number of test: ")
    for i in range(int(testNum)):
        # 图片路径
        testPic = input("The path of test picture: ")
        #testPic = raw_input("The path of test picture: ")
        # 预处理后图片
        testPicArr = pre_pic(testPic)
        # 预测值
        preValue = restore_model(testPicArr)
        print ("The prediction number is: ",preValue)

# 主函数
def main():
    application()

# 当该模块为运行的主模块，执行下列语句
if __name__ == '__main__':
    main()
