# coding: utf-8

import os

import tensorflow as tf
from PIL import Image

# 60000的txt文件中只有50000个文件
image_train_path = './cifar10_data_jpg/cifar-10/train'
label_train_path = './cifar10_data_jpg/cifar10_train_jpg_60000.txt'
tfRecord_train = './data/cifar10_train.tfrecords'
image_test_path = './cifar10_data_jpg/cifar-10/test'
label_test_path = './cifar10_data_jpg/cifar10_test_jpg_10000.txt'
tfRecord_test = './data/cifar10_test.tfrecords'
data_path = './data'
resize_height = 32
resize_width = 32


def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        img_path = os.path.join(image_path, value[0])
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print("the number of picture:", num_pic)
    writer.close()
    print("Write tfrecord successful")


def generate_tfRecord():
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([10], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([3072])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tfRecord(num, isTrain=True):
    # 根据状态选择数据路径
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    # 读取数据
    img, label = read_tfRecord(tfRecord_path)
    '''
    name:tf.train.shuffle_batch
    return:  a list of tensors with the same number and types as tensor_list.     
    '''
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label],
        batch_size=num,  # 批处理规模
        num_threads=2,  # 线程数
        capacity=1000,  # 队列中元素的最大数目
        min_after_dequeue=700)  # 出队后队列元素的最小数目
    return img_batch, label_batch


def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()
