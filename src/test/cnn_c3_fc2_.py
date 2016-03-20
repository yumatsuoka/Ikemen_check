#!/usr/bin/env python
# -*- coding: utf-8 -*-
#python2.7

"""
CNN$B$r:n@.$7$?!#(Bget_image_tensor$B$N%*%V%8%'%/%H$r:n@.$7$F%G!<%?%;%C%H$rFI$_9~$`!#(B
$BFI$_9~$s$@%G!<%?%;%C%H$r(BCNN$B$KEj$2$k!#(B
$B#2#0#1#6G/#27n#2#3F|;~E@$G>v$_9~$_AX(B3$B!$%W!<%j%s%0AX(B3$B!"A47k9gAX#1(B
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
        self.b_conv1 = self.bias_variable([32])
        self.w_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv2 = self.bias_variable([64])
        self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv3 = self.bias_variable([128])
        self.w_conv3 = self.weight_variable([5, 5, 64, 128])
        self.w_fc1 = self.weight_variable([16 * 16 * 128, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.w_fc2 = self.weight_variable([1024, 1])
        self.b_fc2 = self.bias_variable([1])

    def forward(self):
        """NN$B$N(Bforword$B=hM}$r9T$&4X?t(B"""
        x_image = tf.reshape(x, [-1, image_size, image_size, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, self.w_conv3) + self.b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, self.w_fc1) + self.b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.w_fc2) + self.b_fc2)
        return h_fc2

    def weight_variable(self, shape):
        """$B=E$_$K;H$&JQ?t$r=i4|2=$9$k4X?t(B"""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """$B%P%$%"%9$r=i4|2=$9$k4X?t(B"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        """$B>v$_9~$_7W;;$r9T$&4X?t(B"""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """$B%^%C%/%9%W!<%j%s%0$r9T$&4X?t(B"""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-7s %(message)s", filename='log_'+path.splitext(path.basename(__file__))[0]+'.txt')
    logging.info("------------------------------------------------")

    #parameters
    data_list = "/home/yuma/programing/ikemen_check/target/man_target.csv"
    data_dir = "/home/yuma/programing/ikemen_check/image/divide_sex/man/resize/"
    
    batch_size = 20
    test_data_num = 2000
    epoch_size = 3500
    image_size = 128
    alpha = 5e-5

    logging.info("batch_sise:"+str(batch_size)+", epoch_size:"+str(epoch_size)+", alpha:"+str(alpha))

    #$B30It$+$i%G!<%?$rF~$l$kJQ?t$r:n@.(B
    x = tf.placeholder("float", shape=[None, 3 * image_size * image_size])
    y_ = tf.placeholder("float", shape=[None, 1])

    print("creating input_data...")
    logging.info("creating input_data...")
    #$B%G!<%?%;%C%H$r<hF@$9$k!#65;U%G!<%?$N(Bcsv$B$H2hA|%U%)%k%@$N>l=j$H%P%C%A%5%$%:$H3X=,$K;H$&2hA|?t$,0z?t(B
    input_data = get_image_tensor.Input_data(data_list, data_dir, batch_size, test_data_num)
    #############

    print("creating cnn...")
    logging.info("creating cnn...")
    #cnn$B$N%*%V%8%'%/%H$r:n@.(B
    cnn = Cnn()
    
    #loss=cross_entropy
    #loss = -tf.reduce_sum(y_ * tf.log(cnn.forward()))
    #loss = (y_ - y) **2
    loss = tf.reduce_sum(tf.pow(y_ - cnn.forward(), 2) * 0.5)
    train_step = tf.train.AdagradOptimizer(alpha).minimize(loss)
    
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    print(" training dataset...")
    logging.info("training dataset...")
    for i in range(1, epoch_size+1):
        batch_x, batch_y = input_data.next_batch()
        train_step.run(feed_dict={x: batch_x, y_: batch_y })
        if i%100 == 0:
            #print("per100epoch loss:", sess.run(loss, feed_dict={x: batch_x, y_: batch_y}))
            train_output = sess.run(loss, feed_dict={x: batch_x, y_: batch_y})/batch_size
            print("epoch:%d average_loss:%f"%(i, train_output))
            logging.info("epoch,"+str(i)+", average_loss,"+str(train_output))
    #test
    test_output = sess.run(loss, feed_dict={x: input_data.test_data, y_: input_data.test_target })/batch_size
    print("test average_loss:%f"%test_output)
    logging.info("test average_loss,"+str(test_output))

    d = datetime.datetime.today()
    print("end:", d.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("end:"+ str(d.strftime("%Y-%m-%d %H:%M:%S")))
    elapsed_time = time.time() - start
    print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]" )
    logging.info("elapsed_time"+str(elapsed_time)+"[sec]")

