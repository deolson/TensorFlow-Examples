from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf

##WARNING takes a long time to run, step 0 = 10% accuracy
##step 100 = 86%
##step 200 = 80%
##step 300+ 96+%




#functions weight_variable and bias_variable let us define these variables
#without having to init repeatadly we use these functions'
#init weights with a small amount of noise for symmetry breaking and prevent vanishing gradient
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#zero padded so output is same as input, 2x2max pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

if __name__ == '__main__':
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    #weights and bias for the various layers
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    #2nd and 3rd diminesions = image w and h, final color channels
    x_image = tf.reshape(x, [-1,28,28,1])
    #applying the ReLU function, and max pooling
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+ b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #image size is now 7x7 adding a fully connected layer with 1024 nuerons

    # internally applies softmax sums accross all classes, tf reduce takes the avg of the sums
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    #applies backprop and AdamOptimizer
    for i in range(400):
        batch = mnist.train.next_batch(50)
        if i%100 ==0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

    #get avg by casting to float and taking mean
    n_batches = mnist.test.images.shape[0] // 50
    cumulative_accuracy = 0.0
    for index in range(n_batches):
        batch = mnist.test.next_batch(50)
        cumulative_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print("test accuracy {}".format(cumulative_accuracy / n_batches))
