#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from PIL import Image

class Cnn:
    def __init__(self):
        b_conv1 = self.bias_variable([32])
        w_conv1 = self.weight_variable([5, 5, 3, 32])
        w_fc1 = self.weight_variable([112 * 112 * 32, 1024])
        b_fc1 = self.bias_variable([1024])

    def forward(self, x):
        x_image = tf.reshape(x, [-1, 224, 224, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(self.h_conv1)
        h_pool2_flat = tf.reshape(h_pool1, [-1, 112*112*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.w_fc1) + self.b_fc1)
        return h_fc1

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    #data_list = "/home/kantoku2/research/melanoma/224_224/ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
    data_list = "/home/yuma/programing/ikemen_check/target.csv"
    #data_dir = "/home/kantoku2/research/melanoma/224_224/"
    data_dir = "/home/yuma/programing/ikemen_check/image/divide_sex/man/"
    data = pd.read_csv(data_list)

    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, 3*224*224])
    y_ = tf.placeholder("float", shape=[None, 1])

    cnn = Cnn()

    fname_queue = tf.train.string_input_producer([data_list])
    reader = tf.TextLineReader()
    key, val = reader.read(fname_queue)
    fname, label = tf.decode_csv(val, [["aa"], [1]])
    jpeg_r = tf.read_file(fname)
    image = tf.image.decode_jpeg(jpeg_r, channels=0)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess)
    x = sess.run(image)
    print(x)
