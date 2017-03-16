# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import Image, display

def DisplayFractal(a, fmt='jpeg'):
    """Display an array of iteration counts as a
       colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic), 30+50*np.sin(a_cyclic), 155-80*np.cos(a_cyclic)],2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a,0,255))
    with open("./images/image.jpg", "w") as f:
        PIL.Image.fromarray(a).save(f, "jpeg")

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    file_writer = tf.summary.FileWriter('./graph', sess.graph)
    #using numpy to create a 2d array of complex numbers
    Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
    Z = X+1j*Y

    #define and init tenserflow tensors
    xs = tf.constant(Z.astype(np.complex64))
    zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))

    #explicitly init variables before use
    tf.global_variables_initializer().run()

    #defining and running the computation
    #compute the new values of z: z^2 + x
    zs_ = zs*zs + xs

    #have we diverged with this new value
    not_diverged = tf.abs(zs_) < 4

    #operation to update the zs and the iteration count
    #still computing zs after they diverge - wasteful

    step = tf.group(
        zs.assign(zs_),
        ns.assign_add(tf.cast(not_diverged, tf.float32))
    )

    #running for a couple 100 steps
    for i in range(10): step.run()

    #displaying output
    DisplayFractal(ns.eval())
