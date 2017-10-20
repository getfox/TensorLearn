from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib as plt

rng = numpy.random

learning_rate = 0.05
training_epochs = 1000
display_range = 50

train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="Weights")
b = tf.Variable(rng.randn(), name="Bias")

pridiction = tf.add(tf.multiply(X, W), b)

Cost = tf.reduce_sum(tf.pow(pridiction-Y, 2))/(2*n)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(Cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epochs in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer,feed_dict={X:x, Y:y})
        if (epochs+1) % display_range == 0:
            c = sess.run(Cost, feed_dict={X:train_X, Y:train_Y})

            print("Epoch:", '%04d' % (epochs + 1), "Cost=", '{:.9f}'.format(c), \
                  "W=", sess.run(W), "b=", sess.run(b))

    print("ALL EPOCHS FINISHED!LET'S CHECK IT OUT~")

    training_cost = sess.run(Cost,feed_dict={X:train_X, Y:train_Y})
    print("Training cost:", training_cost, "W=", sess.run(W), "b=", sess.run(b))
