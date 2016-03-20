#!/usr/bin/env python
# -*- coding: utf-8 -*-

#今のプログラムは１枚の画像しか読み込めていない！参考サイトも１枚しか読み込んでいない！
#画像の順番をランダムにしたい
#バッチサイズごとに分けたテンソルを作りたい
#バッチサイズからはみ出た枚数＋αで検定用データセットを作りたい

import pandas as pd
from PIL import Image
#import tensorflow as tf
import numpy as np

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
        self.data_batch, self.target_batch = self.make_batch()
        
    def create_tensor(self):
        """csvの中身を全部読み込む。読み込んだ画像名から対応する画像を読み込む"""
        csv_data = pd.read_csv(self.data_list, header=None)
        for i in range(2000):
            image = Image.open(csv_data[0][i], 'r')
            #image = tf.image.decode_jpeg(csv_data[0][i], channels=0)
            #リストの平坦化
            image = np.array(image)
            x_image = image.flatten()
            #x_image = [tf.reshape(image, [-1])]
            #debug
            #print i #, train_data, x_image
            #
            #if i != 0:
            #    train_data = tf.concat(0, [train_data, x_image])
            #else:
            #    self.train_data = x_image
            #debug
            #print train_data
            #
            if i == 0:
                self.train_data = x_image
            else:
                self.train_data = np.vstack([self.train_data, x_image/255.])
#        for i in range(2000, 2100):
        for i in range(2000, 2020):
            image = Image.open(csv_data[0][i], 'r')
            #image = tf.image.decode_jpeg(csv_data[0][i], channels=0)
            #x_image = tf.reshape(image, [-1])
            #if i != 0:
            #    self.test_data = tf.concat(0, [self.test_data, x_image])
            #else:
            #    self.test_data = x_image
            #x_image = sum(image, [])
            image = np.array(image)
            x_iamge = image.flatten()
            #print i
            if i == 2000:
                self.test_data = x_image
            else:
                self.test_data = np.vstack([self.test_data, x_image/255.])
            #self.test_data.append(x_image/255.) 
        
        csv_data = np.array(csv_data)
        self.train_target = csv_data[:2000, 1]
 #       self.test_target = csv_data[2000:2101, 1]
        self.test_target = csv_data[2000:2021, 1]

        #print "train_data, test_data", self.train_data.shape, self.test_data.shape
        #print "train_target, test_target", self.train_target.shape, self.test_target.shape

    def make_batch(self):
        #separate data by batch_size, and target too
        self.batch_itr =  2000 / self.batch_size
        #print type(self.train_data)
        data_sep = np.vsplit(self.train_data, self.batch_itr)
        target_sep = np.hsplit(self.train_target, self.batch_itr)
        return data_sep, target_sep

    def next_batch(self):
        self.epoch_num += 1
        next_num = self.epoch_num%self.batch_itr
        return data_batch[next_num], target_batch[next_num] 

if __name__ == '__main__':
    data_list = "/home/yuma/programing/ikemen_check/target.csv"
    data_dir = "/home/yuma/programing/ikemen_check/image/divide_sex/man/resize/"
    batch = 20

    input_data = Input_data(data_list, data_dir, batch)
    

    print "test"




