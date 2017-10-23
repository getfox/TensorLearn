from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=False)

num_steps = 500
mini_batch = 2000
num_trees = 10
num_pixels = 784
num_classes = 10
max_nodes = 1000

X = tf.placeholder(tf.float32, shape=[None, num_pixels])
Y = tf.placeholder(tf.int32, shape=[None])

Hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_pixels,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
forest_graph = tensor_forest.RandomForestGraphs(Hparams)

train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

infer_op = forest_graph.inference_graph(X)

target_num = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(target_num, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(num_steps):
        xbatch, ybatch = mnist.train.next_batch(mini_batch)
        _, l = sess.run([train_op, loss_op], feed_dict={X:xbatch, Y:ybatch})
        if i%50 == 0 or i ==1:
            a = sess.run(accuracy, feed_dict={X:xbatch, Y:ybatch})
            print("In the %i'th step, the accuracy is :%f, and the loss is:%f" %(i, a, l))

    text_x, text_y = mnist.test.images, mnist.test.labels
    print("The argorithem in test data sets is:%f" %(sess.run(accuracy, feed_dict={X:text_x, Y:text_y})))
