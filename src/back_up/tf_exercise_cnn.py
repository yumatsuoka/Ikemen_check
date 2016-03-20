# -*-coding:utf-8-*-
#Multilayer Convolutional Network in tutorials of tensorflow 
#2015/12/29

#like python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#get mnist data
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#InteractiveSession$B$O%0%i%U$r:n$k$H$-$K=@Fp$KLrN)$D$i$7$$!*(B
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#weight_initialization
#$B>.$5$JBg$-$5$N%N%$%:$r=E$_$H%P%$%"%9$K2C$($k!#8{G[$,#0$K$J$k$NK8$2$k$?$a!#(B
#ReLU$B$r;H$&$+$i!"8m:9$,=P$J$$;`$s$@%K%e!<%m%s$r:n$i$J$$$?$a$K$3$N4X?t$r:n$k!#(B
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#$B%9%H%i%$%I$O#1$@$+$i!"F~NO$HF1$8%5%$%:$N=PNO$,=P$F$/$k(B
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#2x2$B$N%5%$%:$N%W!<%j%s%0$r9T$&!#(B
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#$B:G=i$N#2$D$O%Q%C%A$N%5%$%:!#$D$.$OF~NO$N%A%c%s%M%k?t!":G8e$O%"%&%H%W%C%H$N%A%c%s%M%k$N?t(B
w_conv1 = weight_variable([5, 5, 1, 32])
#$B%P%$%"%9$O%"%&%H%W%C%H%A%c%s%M%k$HF1$8Bg$-$5(B
b_conv1 = bias_variable([32])
#$B%l%$%d!<$KE,MQ$9$k$?$a$K!"%j%5%$%:$9$k!#:G=i$N#1$D$O(B4d tensor$B$K$9$k$?$a$NCM(B
#$B#2$D$a$H#3$D$a$O2hA|$N2#$H=D$KBP1~$7$F$$$k!#:G8e$N$b$N$O?'$N<!85?t$K0lCW$9$k!#(B
#$B:G=i$N(B-1$B$O#1<!85$K$9$k$?$a$N$b$N!)(B
x_image = tf.reshape(x, [-1, 28, 28, 1])

#$BF~NO2hA|$H=E$_$r(Bcovolve$B$7$F%P%$%"%9$r2C$($F!"(BReLU$B4X?t$K$V$A9~$`!#(B
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
#$B3h@-2=4X?t$+$i=P$F$-$?$b$N$r(Bmax_pooling$B$9$k!#(B
h_pool1 = max_pool_2x2(h_conv1)

#$B#2$DL\$NAX$G$O#5!_#5%Q%C%A$G!"#6#4$NFCD'$r:n$k$h$&$K@_Dj(B
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely connected layer($BA47k9gAX(B)
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#2x2$B$N%^%C%/%9%W!<%j%s%0$r#22s9T$C$?$+$i!"2hA|%5%$%:$O(B7x7$B$K$J$C$F$$$k(B
#1024$B<!85$NA47k9g%K%e!<%m%s$r:n$k!#%j%5%$%:$7$F%Y%/%H%k$KD>$9!#(B
#$B%Y%/%H%k$H=E$_$NFb@Q$r<h$C$F!"%P%$%"%9$r2C$($F(BReLU$B4X?t$KFM$C9~$`!#(B
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#overfitting($B2a3X=,(B)$B$rKI$0$?$a$K%I%m%C%W%"%&%H$r!"%j!<%I%"%&%H%l%$%d!<$NA0$KF~$l$k!#(B
#$B3X=,$7$F$$$k$H$-$K$O3NN(E*$K%I%m%C%W%"%&%H$5$l$k$h$&$K!"%F%9%H$7$F$$$k$H$-$O$5$l$J$$$h$&$K(B
#placeholder$B$r:n$k!#(Btf.nn.dropout$B$r;H$($P<+F0E*$K%^%9%/$r$+$V$;$k$h$&$K$d$C$F$/$l$k$h!<(B
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#convolute$B$7$F$-$?%Y%/%H%k$r2s5"$5$;$k%l%$%d!<(B(Readout Layer)$B$KF~$l$k!#(B
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#softmax$B4X?t$r$+$^$7$F!"3NN($N7A$G2s5"$r9T$&!#(B
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

#$B3X=,$r%F%9%H(B
#$B8m:9$r;;=P$9$k!#%/%m%9%(%s%H%m%T!<$r;H$&!#%P%C%AC10L$G8m:9$r7W;;$7$?$[$&$,A4BNE*$KNI$$(B
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#ADAM_optimizer$B$O3NN(E*8{G[K!$h$j$b$h$j@vN}$5$l$F$$$k!#(B
#keep_prob$B$O%I%m%C%W%"%&%H$5$;$k3NN($G!"(Bfeed_dict$B$K2C$($F$$$k!#(B
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#tf.argmax$B$O#1$D$N<4$NCf$G$b$C$H$b9b$$%$%s%G%C%/%9$rJV$9!#$=$NCM$,Ey$7$$$+$I$&$+(Btf.equal$B$G3N$+$a$k(B
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#boolean$B$N%j%9%H$rJV$7$F!"MWAG$N3d9g$r(Bfloat$B7?$NCM$GJV$9!#(B
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#$BHy>.$JCM$GJQ?t(B(Variable)$B$N=i4|CM$rKd$a$F$/$l$k!)$N$+$J(B
sess.run(tf.initialize_all_variables())
#$B3X=,%9%?!<%H(B
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" %(i, train_accuracy))
    #placeholder$B$K(Bfeed_dict$B$r;H$C$FCM$rF~$l$F$$$/(B
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

