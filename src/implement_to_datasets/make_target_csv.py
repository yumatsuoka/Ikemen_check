#!/usr/bin/env python
# -*- coding: utf-8 -*-
#python2.7

import pandas as pd
import os

#valiables
data_dir = "/home/yuma/programing/ikemen_check/image/woman_kawai_or_not/"
one_object = "bad/"
another_object = "good/"

#placeholder
target_flag = [0, 1]
output_list = []

#implement
for i in target_flag:
    if i == 0:
        dir_name = data_dir + one_object
        files = os.listdir(dir_name)
    else: 
        dir_name = data_dir + another_object
        files = os.listdir(dir_name)
    
    for file in files:
        temp = [file, i]
        output_list.append(temp)

csv_data = pd.DataFrame(output_list)
#header削除を保存時にオプション追加
csv_data.to_csv("bijo_target.csv", header=False, index=False)
    
