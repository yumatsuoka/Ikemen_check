#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNNを作成した。get_image_tensorのオブジェクトを作成してデータセットを読み込む。
読み込んだデータセットをCNNに投げる。
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import get_image_tensor
import datetime
import time
import logging
from os import path

class Cnn:
    def __init__(self):
        pass

    def forward(self):
        """NNのforward処理を行う関数"""
        pass
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

class C1fc1(Cnn):#cnnを継承
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.w_fc1 = self.weight_variable([64 * 64 * 32, 2])
        self.b_fc1 = self.bias_variable([2])

    def forward(self):#@override
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_pool1_flat = tf.reshape(h_pool1, [-1, 64 * 64 * 32])
        #classificationなのでsoftmax関数で出力
        h_fc1 = tf.nn.softmax(tf.matmul(h_pool1_flat, self.w_fc1) + self.b_fc1)
        return h_fc1

class C3fc1(Cnn):#Cnnを継承
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv2 = self.bias_variable([64])
        self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv3 = self.bias_variable([128])
        self.w_conv3 = self.weight_variable([5, 5, 64, 128])
        self.w_fc1 = self.weight_variable([16 * 16 * 128, 2])
        self.b_fc1 = self.bias_variable([2])

    def forward(self):#@override
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.w_conv3) + self.b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 128])
        h_fc1 = tf.nn.softmax(tf.matmul(h_pool3_flat, self.w_fc1) + self.b_fc1)
        return h_fc1

class C3fc2(Cnn):#Cnnを継承
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv2 = self.bias_variable([64])
        self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv3 = self.bias_variable([128])
        self.w_conv3 = self.weight_variable([5, 5, 64, 128])
        self.w_fc1 = self.weight_variable([16 * 16 * 128, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.w_fc2 = self.weight_variable([1024, 2])
        self.b_fc2 = self.bias_variable([2])

    def forward(self):#@override
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.w_conv3) + self.b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, self.w_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop, self.w_fc2) + self.b_fc2)
        return h_fc2

class C3fc3(Cnn):#Cnnを継承
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv2 = self.bias_variable([64])
        self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv3 = self.bias_variable([128])
        self.w_conv3 = self.weight_variable([5, 5, 64, 128])
        self.w_fc1 = self.weight_variable([16 * 16 * 128, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.w_fc2 = self.weight_variable([1024, 1024])
        self.b_fc2 = self.bias_variable([1024])
        self.w_fc3 = self.weight_variable([1024, 2])
        self.b_fc3 = self.bias_variable([2])

    def forward(self):#@override
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.w_conv3) + self.b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, self.w_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop, self.w_fc2) + self.b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        h_fc3 = tf.nn.softmax(tf.matmul(h_fc2_drop, self.w_fc3) + self.b_fc3)
        return h_fc3

class C4fc1(Cnn):#Cnnを継承
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv2 = self.bias_variable([64])
        self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv3 = self.bias_variable([128])
        self.w_conv3 = self.weight_variable([5, 5, 64, 128])
        self.b_conv4 = self.bias_variable([256])
        self.w_conv4 = self.weight_variable([5, 5, 128, 256])
        self.w_fc1 = self.weight_variable([8 * 8 * 256, 2])
        self.b_fc1 = self.bias_variable([2])

    def forward(self):#@override
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.w_conv3) + self.b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3, self.w_conv4) + self.b_conv4)
        h_pool4 = self.max_pool_2x2(h_conv4)
        h_pool4_flat = tf.reshape(h_pool4, [-1, 8 * 8 * 256])
        h_fc1 = tf.nn.softmax(tf.matmul(h_pool4_flat, self.w_fc1) + self.b_fc1)
        return h_fc1

class C4fc2(Cnn):#Cnnを継承
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv2 = self.bias_variable([64])
        self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv3 = self.bias_variable([128])
        self.w_conv3 = self.weight_variable([5, 5, 64, 128])
        self.b_conv4 = self.bias_variable([256])
        self.w_conv4 = self.weight_variable([5, 5, 128, 256])
        self.w_fc1 = self.weight_variable([8 * 8 * 256, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.w_fc2 = self.weight_variable([1024, 2])
        self.b_fc2 = self.bias_variable([2])

    def forward(self):#@override
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.w_conv3) + self.b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3, self.w_conv4) + self.b_conv4)
        h_pool4 = self.max_pool_2x2(h_conv4)
        h_pool4_flat = tf.reshape(h_pool4, [-1, 8 * 8 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, self.w_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop, self.w_fc2) + self.b_fc2)
        return h_fc2

class C4fc3(Cnn):#Cnnを継承
    def __init__(self):
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv2 = self.bias_variable([64])
        self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv3 = self.bias_variable([128])
        self.w_conv3 = self.weight_variable([5, 5, 64, 128])
        self.b_conv4 = self.bias_variable([256])
        self.w_conv4 = self.weight_variable([5, 5, 128, 256])
        self.w_fc1 = self.weight_variable([8 * 8 * 256, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.w_fc2 = self.weight_variable([1024, 1024])
        self.b_fc2 = self.bias_variable([1024])
        self.w_fc3 = self.weight_variable([1024, 2])
        self.b_fc3 = self.bias_variable([2])

    def forward(self):#@override
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.w_conv3) + self.b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3, self.w_conv4) + self.b_conv4)
        h_pool4 = self.max_pool_2x2(h_conv4)
        h_pool4_flat = tf.reshape(h_pool4, [-1, 8 * 8 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, self.w_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop, self.w_fc2) + self.b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        h_fc3 = tf.nn.softmax(tf.matmul(h_fc2_drop, self.w_fc3) + self.b_fc3)
        return h_fc3

if __name__=='__main__':
    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-7s %(message)s", filename='log_'+path.splitext(path.basename(__file__))[0]+'.txt')
    logging.info("---------------------------------------------------")

    #parameters
    data_list = "/home/yuma/programing/github/ikemen_check/target/bijo_target.csv"
    data_dir = "/home/yuma/programing/github/ikemen_check/image/woman/"

    batch_size = 20
    test_data_num = 3540
    epoch_size = 300
    image_size = 128
    alpha = 1e-4
    
    ########
    #コマンドライン引数でbatchサイズと学習係数を決める
    #param = sys.argv
    #batch_size = int(param[1])
    #alpha = float(param[2])
    ########

    #logging parameter
    logging.info("batch_sise:"+str(batch_size)+", epoch_size:"+str(epoch_size)+", alpha:"+str(alpha))

    #外部からデータを入れる変数を作成
    x = tf.placeholder("float", shape=[None, 3 * image_size * image_size])
    y_ = tf.placeholder("float", shape=[None, 1])
    keep_prob = tf.placeholder("float")
    
    print("creating input_data...")
    logging.info("creating input_data...")
    #データセットを取得する。教師データのcsvと画像フォルダの場所とバッチサイズと学習に使う画像数が引数
    input_data = get_image_tensor.Input_data(data_list, data_dir, batch_size, test_data_num)
    #############

    print("creating cnn...")
    logging.info("createing cnn...")
    #cnnのオブジェクトを作成###########
    cnn = C1fc1()
    #cnn = C3fc1()
    #cnn = C3fc2()
    #cnn = C3fc3()
    #cnn = C4fc1()
    #cnn = C4fc2()
    #cnn = C4fc3()
    ##################################
    
    #loss クラスタリングならクロスエントロピー使ったほうが学習が早い
    #loss = tf.reduce_sum(tf.pow(y_ - cnn.forward(), 2) * 0.5)
    loss = -tf.reduce_sum(y_ * tf.log(cnn.forward()))
    train_step = tf.train.AdagradOptimizer(alpha).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(cnn.forward(), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #initialize_valiables
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    print(" training dataset...")
    logging.info("training dataset...")
    for i in range(1, epoch_size+1):
        batch_x, batch_y = input_data.next_batch()
        if i%100 == 0:
            #train_output = sess.run(loss, feed_dict={x: batch_x, y_: batch_y})/batch_size
            #train_output = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})/batch_size
            #train_accuracy
            train_output = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            print("epoch:%d training accuracy:%g"%(i, train_output))
            logging.info("epoch,"+str(i)+", training_accuracy,"+str(train_output))
        #train_step.run(feed_dict={x: batch_x, y_: batch_y })
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    #test
    #test_output = sess.run(loss, feed_dict={x: input_data.test_data, y_: input_data.test_target })/batch_size 
    #test_output = sess.run(loss, feed_dict={x: input_data.test_data, y_: input_data.test_target, keep_prob: 1.0})/batch_size
    test_output = accuracy.eval(feed_dict={x: input_data.test_data, y_: input_data.test_target, keep_prob: 1.0}) 
    print("test accuracy:%g"%test_output)
    logging.info("test accuracy,"+str(test_output))

    d = datetime.datetime.today()
    print("end:", d.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("end:"+ str(d.strftime("%Y-%m-%d %H:%M:%S")))
    elapsed_time = time.time() - start
    print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]" )
    logging.info("elapsed_time"+str(elapsed_time)+"[sec]")

