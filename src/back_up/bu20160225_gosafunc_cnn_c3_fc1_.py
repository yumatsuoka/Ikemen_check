#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNNを作成した。get_image_tensorのオブジェクトを作成してデータセットを読み込む。
読み込んだデータセットをCNNに投げる。
２０１６年２月２３日時点で畳み込み層3，プーリング層3、全結合層１
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import get_image_tensor
import datetime

class Cnn:
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv2 = self.bias_variable([64])
        self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv3 = self.bias_variable([128])
        self.w_conv3 = self.weight_variable([5, 5, 64, 128])
        self.w_fc1 = self.weight_variable([16 * 16 * 128, 1])
        self.b_fc1 = self.bias_variable([1])

    def forward(self):
        """NNのforword処理を行う関数"""
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.w_conv3) + self.b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, self.w_fc1) + self.b_fc1)
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
    
    batch_size = 100
    test_data_num = 2000
    epoch_size = 10000
    image_size = 128

    #外部からデータを入れる変数を作成
    x = tf.placeholder("float", shape=[None, 3 * image_size * image_size])
    y_ = tf.placeholder("float", shape=[None, 1])

    print("creating input_data...")
    #データセットを取得する。教師データのcsvと画像フォルダの場所とバッチサイズと学習に使う画像数が引数
    input_data = get_image_tensor.Input_data(data_list, data_dir, batch_size, test_data_num)
    #############

    print("creating cnn...")
    #cnnのオブジェクトを作成
    cnn = Cnn()
    
    #cross_entropy
    loss = -tf.reduce_sum(y_ * tf.log(cnn.forward()))
    train_step = tf.train.AdagradOptimizer(1e-4).minimize(loss)
    
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    print(" training dataset...")
    for i in range(epoch_size):
        batch_x, batch_y = input_data.next_batch()
        train_step.run(feed_dict={x: batch_x, y_: batch_y })
        if i%100 == 0:
            #print("per100epoch loss:", sess.run(loss, feed_dict={x: batch_x, y_: batch_y}))
            print("epoch:%d loss:%e"%(i, sess.run(loss, feed_dict={x: batch_x, y_: batch_y})))
    #test 
    print("test loss:%e"%sess.run(loss, feed_dict={x: input_data.test_data, y_: input_data.test_target }))
    d = datetime.datetime.today()
    print("end:", d.strftime("%Y-%m-%d %H:%M:%S"))
