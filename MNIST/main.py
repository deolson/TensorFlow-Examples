#import data set from Yann LeCun's website
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf
if __name__ == '__main__':
    # get from feed_dict, x = MNIST image that is flattened into 784
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    # softmax gives a prob distribution over defined probabilities
    # normilizes
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    #get from feed_dict, y_ = correct answer
    y_ = tf.placeholder(tf.float32, [None, 10])
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #get avg by casting to float and taking mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # about 90-93% accuracy -- not very good
