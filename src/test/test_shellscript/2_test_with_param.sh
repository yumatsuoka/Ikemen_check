#!/bin/sh

for i in "cnn_c3_fc2_.py 20 5e-5" "cnn_c3_fc2_.py 20 1e-5" "cnn_c3_fc2_.py 20 5e-6" "cnn_c3_fc2_.py 20 1e-6" "cnn_c3_fc2_.py 40 5e-5" "cnn_c3_fc2_.py 40 1e-5" "cnn_c3_fc2_.py 40 5e-6" "cnn_c3_fc2_.py 40 1e-6"  "cnn_c4_fc1_.py 20 5e-5" "cnn_c4_fc1_.py 20 1e-5" "cnn_c4_fc1_.py 20 5e-6" "cnn_c4_fc1_.py 20 1e-6" "cnn_c4_fc1_.py 40 5e-5" "cnn_c4_fc1_.py 40 1e-5" "cnn_c4_fc1_.py 40 5e-6" "cnn_c4_fc1_.py 40 1e-6" 
do
	 python $i
	 echo $i
done

