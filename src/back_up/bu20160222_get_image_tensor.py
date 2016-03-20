#!/usr/bin/env python
# -*- coding: utf-8 -*-

#今のプログラムは１枚の画像しか読み込めていない！参考サイトも１枚しか読み込んでいない！
#画像の順番をランダムにしたい
#バッチサイズごとに分けたテンソルを作りたい
#バッチサイズからはみ出た枚数＋αで検定用データセットを作りたい

import pandas as pd
from PIL import Image

class Input_data:
    def __init__(self, data_list, data_dir, batch_size):
        self.data_list = data_list
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.batch_itr = 0
        self.epoch_num = 0
        self.data, self.target = self.create_tensor()
        self.batch_data = self.make_batch()
        
    def create_tensor(self):
        data = pd.read_csv(self.data_list)
        sess = tf.InteractiveSession()
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

        #wanna return a pair of x and y_
        #y_ = sess.run(label)
        #return [x, y_ ]
        return x

    def make_batch(self):
        #separate data by batch_size, and target too
        self.batch_itr = hoge / batch_size
        pass

    def next_batch(self):
        self.epoch_num += 1
        next_num = self.epoch_num%self.batch_itr
        return [batch_data[next_num], batch_target[next_num]] 

if __name__ == '__main__':
    data_list = "/home/yuma/programing/ikemen_check/target.csv"
    data_dir = "/home/yuma/programing/ikemen_check/image/divide_sex/man/"
    batch = 20

    input_data = Input_data(data_list, data_dir, batch)

    print "test"




