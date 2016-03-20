#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像の順番をランダムにしたい
バッチサイズごとに分けたテンソルを作りたい
tensorflowの組み込みの関数を用いてtensor型のデータセットを作りたかったが、テンソルフロー自体が理解不足であること、ネットの記事を見る感じでは、numpy型のtensorを用意して、tensorflowを実行時に、placeholderにこのtensorを入れればなんとかなる感じがするので、pillowとpandasとnumpyを使って画像を読み込んでデータセットを作成した。
"""

import pandas as pd
from PIL import Image
#import tensorflow as tf
import numpy as np

class Input_data:
    def __init__(self, data_list, data_dir, batch_size, train_data_num):
        self.data_list = data_list
        self.data_dir = data_dir
        self.batch_size = batch_size
        #学習に用いるデータの数を指定
        self.train_data_num = train_data_num
        self.batch_itr = 0
        self.epoch_num = 0
        self.train_data = None
        self.train_target = None
        self.test_data = None
        self.test_target = None
        self.create_tensor()
        self.data_batch, self.target_batch = self.make_batch()
        
    def create_tensor(self):
        """csvから画像名と教師データを読み込む。画像名から画像をnumpy型のリストとして読み込む
        引数なし
        返り値もなし：メンバ変数を直接触るため
        """
        #csvの中身を全部読み込む。
        csv_data = pd.read_csv(self.data_list, header=None)
        #2000枚を学習データとして用いて、残りの中からバッチサイズ分をテストデータとして用いる
        #画像を読み込んで、１枚の画像を１次元に変換して、それをまとめて２次元のtensorにする
        for i in range(self.train_data_num):
            image = Image.open(csv_data[0][i], 'r')
            #リストの平坦化
            image = np.array(image)
            x_image = image.flatten()
            if i == 0:
                self.train_data = x_image
                self.train_target = np.array([csv_data[1][i]])
            else:
                self.train_data = np.vstack([self.train_data, x_image/255.])
                self.train_target = np.vstack([self.train_target, np.array([csv_data[1][i]])])
        for i in range(self.train_data_num, self.train_data_num+self.batch_size):
            image = Image.open(csv_data[0][i], 'r')
            image = np.array(image)
            x_iamge = image.flatten()
            if i == self.train_data_num:
                self.test_data = x_image
                self.test_target = np.array([csv_data[1][i]])
            else:
                self.test_data = np.vstack([self.test_data, x_image/255.])
                self.test_target = np.vstack([self.test_target, np.array([csv_data[1][i]])])
        #debug
        print "train_data, test_data", self.train_data.shape, self.test_data.shape
        print "train_target, test_target", self.train_target.shape, self.test_target.shape
        #

    def make_batch(self):
        #バッチサイズに従って学習用データを分ける。
        self.batch_itr =  self.train_data_num / self.batch_size
        data_sep = np.vsplit(self.train_data, self.batch_itr)
        target_sep = np.vsplit(self.train_target, self.batch_itr)
        return data_sep, target_sep

    def next_batch(self):
        #実行するとバッチ１つ分の学習用データを返す。
        self.epoch_num += 1
        next_num = self.epoch_num%self.batch_itr
        return self.data_batch[next_num], self.target_batch[next_num] 

if __name__ == '__main__':
    data_list = "/home/yuma/programing/ikemen_check/target.csv"
    data_dir = "/home/yuma/programing/ikemen_check/image/divide_sex/man/resize/"
    batch = 20
    train_data_num = 2000

    input_data = Input_data(data_list, data_dir, batch, train_data_num)
    
