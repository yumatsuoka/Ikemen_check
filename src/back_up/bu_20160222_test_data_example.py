#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import get_image_tensor as gi_tensor
from gi_tensor import Input_data

class Cnn:
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.w_fc1 = self.weight_variable([112 * 112 * 32, 1024])
        self.b_fc1 = self.bias_variable([1024])

    def forward(self, x):
        """NNのforword処理を行う関数"""
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_pool2_flat = tf.reshape(h_pool1, [-1, 112*112*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.w_fc1) + self.b_fc1)
        return h_fc1

    def weight_variable(self, shape):
        """重みに使う変数を初期化する関数"""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """バイアスを初期化する関数"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        """畳み込み計算を行う関数"""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """マックスプーリングを行う関数"""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    data_list = "/home/yuma/programing/ikemen_check/target.csv"
    data_dir = "/home/yuma/programing/ikemen_check/image/divide_sex/man/resize/"
    
    batch_size = 20
    epoch_size = 201
    image_size = 128

    x = tf.placeholder("float", shape=[None, 3 * image_size * image_size])
    y_ = tf.placeholder("float", shape=[None, 1])

    input_data = Input_data(data_list, data_dir, batch_size)
    
    cnn = Cnn()
    
    #cross_entropy
    loss = -tf.reduce_sum(y_ * tf.log(cnn.forward()))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(epoch_size):
        batch_x, batch_y = input_data.next_batch()
        train_step.run(feed_dict={x: batch_x, y_: batch_y })
        if i%100 == 0:
            print "step: %d loss: %f" %(i, loss)
    
   #train_step.run(feed_dict={x: test_batch_x, y_: test_batch_y })
   #print "test loss:%f" % loss
