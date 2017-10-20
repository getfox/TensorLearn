from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.train.next_batch(800)

xtr = tf.placeholder(tf.float32,[None,784])
xte = tf.placeholder(tf.float32,[784])

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

pred = tf.arg_min(distance, 0)

accuracy = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        nn_index = sess.run(pred, feed_dict={xtr:Xtr, xte:Xte[i, :]})

        if i%100 == 0:
            print("With the", i, "'th step,we predict the number will be :", np.argmax(Ytr[nn_index]),
                            ", and the truth is:", np.argmax(Yte[i]))
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Yte)
    print("It's all done~ \nThe accuracy is:", accuracy, ", Hooray!")