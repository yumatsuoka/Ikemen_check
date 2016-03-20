#!/usr/bin/env python
# -*- coding: utf-8 -*-

#今のプログラムは１枚の画像しか読み込めていない！参考サイトも１枚しか読み込んでいない！
#画像の順番をランダムにしたい
#バッチサイズごとに分けたテンソルを作りたい
#バッチサイズからはみ出た枚数＋αで検定用データセットを作りたい

import pandas as pd
from PIL import Image
import tensorflow as tf

class Input_data:
    def __init__(self, data_list, data_dir, batch_size):
        self.data_list = data_list
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.batch_itr = 0
        self.epoch_num = 0
        self.train_data = None
        self.train_target = None
        self.test_data = None
        self.test_target = None
        self.create_tensor()
        self.batch_data = self.make_batch()
        
    def create_tensor(self):
        """csvの中身を全部読み込む。読み込んだ画像名から対応する画像を読み込む"""
        csv_data = pd.read_csv(self.data_list, header=None)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        sess.run(self.get_image(csv_data))

        self.train_target = csv_data[:2000, 1]
        self.test_target = csv_data[2000:2101, 1]
        print "end"

    def get_image(self, csv_data):
        train_data = tf.Variable([2])
        for i in range(2000):
            image = tf.image.decode_jpeg(csv_data[0][i], channels=0)
            x_image = [tf.reshape(image, [-1])]
            #debug
            print i, train_data, x_image
            #
            if i != 0:
                train_data = tf.concat(0, [train_data, x_image])
            else:
                self.train_data = x_image
            #debug
            print train_data
            #
        for i in range(2000, 2100):
            image = tf.image.decode_jpeg(csv_data[0][i], channels=0)
            x_image = tf.reshape(image, [-1])
            if i != 0:
                self.test_data = tf.concat(0, [self.test_data, x_image])
            else:
                self.test_data = x_image

    def make_batch(self):
        #separate data by batch_size, and target too
        #self.batch_itr = hoge / batch_size
        return 0

    def next_batch(self):
        self.epoch_num += 1
        next_num = self.epoch_num%self.batch_itr
        return [batch_data[next_num], batch_target[next_num]] 

if __name__ == '__main__':
    data_list = "/home/yuma/programing/ikemen_check/target.csv"
    data_dir = "/home/yuma/programing/ikemen_check/image/divide_sex/man/resize/"
    batch = 20

    input_data = Input_data(data_list, data_dir, batch)

    print "test"




