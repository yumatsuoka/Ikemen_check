# -*-coding:utf-8-*-
#Multilayer Convolutional Network in tutorials of tensorflow 
#2015/12/29

#like python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob

#get file_names
list = glob.glob('./*.jpg')

print("test")
for file in list:
    tf.image.decode_jpeg(file, 0)
    print(file)

print("test_end")
