import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features, model_dir="./network")

    x = np.array([1,2,3,4])
    y = np.array([0,-1,-2,-3])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)
