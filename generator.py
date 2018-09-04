import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from ops import *
from tensorflow.layers import batch_normalization
from tensorflow.keras.layers import UpSampling2D

class Generator:
    def __init__(self, img_shape, batch_size):
        self.img_rows, self.img_cols, self.channels = img_shape
        self.batch_size = batch_size
        with tf.variable_scope('g'):
            print("Initializing generator weights")
            self.W1 = init_weights([100, 7*7*512])
            self.W2 = init_weights([3, 3, 512, 256])
            self.W3 = init_weights([3, 3, 256, 128])
            self.W4 = init_weights([3, 3, 128, 1])
            

    def forward(self, X, momentum=0.5):
        z = tf.matmul(X, self.W1)
        z = tf.nn.relu(z)
        z = tf.reshape(z, [-1, 7, 7, 512])

        z = UpSampling2D()(z)
        z = conv2d(z, self.W2, [1, 1, 1, 1], padding="SAME")
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = UpSampling2D()(z)
        z = conv2d(z, self.W3, [1, 1, 1, 1], padding="SAME") 
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = conv2d(z, self.W4, [1, 1, 1, 1], padding="SAME") 

        return tf.nn.tanh(z)




        
