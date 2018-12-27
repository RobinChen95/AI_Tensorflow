# coding: utf-8

import os

import tensorflow as tf
from PIL import Image

AGE = True
if AGE:
    label_train_path = './Adience/age_train.txt'
    tfRecord_train = './data/age_train.tfrecords'
    label_test_path = './Adience/age_text.txt'
    tfRecord_test = './data/age_test.tfrecords'
else:
    label_train_path = './Adience/gender_train.txt'
    tfRecord_train = './data/gender_train.tfrecords'
    label_test_path = './Adience/gender_text.txt'
    tfRecord_test = './data/gender_test.tfrecords'
data_path = './data'
resize_height = 227
resize_width = 227


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfRecord(tfRecordName, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        img_path = value[0]
        img = Image.open(img_path)
        img = img.resize((resize_width, resize_height), Image.ANTIALIAS)
        img_raw = img.tobytes()
        if AGE:
            label = [0] * 8
        else:
            label = [0] * 2
        label[int(value[1])] = 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': _bytes_feature(img_raw),
            # 'gender' : _int64_feature(gender),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
        num_pic += 1
        if num_pic % 1000 == 0:
            print ("the number of picture:", num_pic)
    writer.close()
    print ("the number of picture:", num_pic, "\nDone!")


def generate_tfRecord():
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    write_tfRecord(tfRecord_train, label_train_path)
    write_tfRecord(tfRecord_test, label_test_path)


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': (tf.FixedLenFeature([8], tf.int64) if AGE else tf.FixedLenFeature([2], tf.int64)),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([resize_height * resize_height * 3])
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
