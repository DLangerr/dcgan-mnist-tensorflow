import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from ops import *
from tensorflow.layers import batch_normalization


class Discriminator:
    def __init__(self, img_shape):

        self.img_rows, self.img_cols, self.channels = img_shape
        with tf.variable_scope('d'):
            print("Initializing discriminator weights")
            self.W1 = init_weights([5, 5, self.channels, 64])
            self.b1 = init_bias([64])
            self.W2 = init_weights([3, 3, 64, 64])
            self.b2 = init_bias([64])
            self.W3 = init_weights([3, 3, 64, 128])
            self.b3 = init_bias([128])
            self.W4 = init_weights([2, 2, 128, 256])
            self.b4 = init_bias([256])
            self.W5 = init_weights([7*7*256, 1])
            self.b5 = init_bias([1])
            
    
    def forward(self, X, momentum=0.5):
        X = tf.reshape(X, [-1, self.img_rows, self.img_cols, self.channels])
        z = conv2d(X, self.W1, [1, 2, 2, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.b1)
        z = tf.nn.leaky_relu(z)
        
        z = conv2d(z, self.W2, [1, 1, 1, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.b2)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = conv2d(z, self.W3, [1, 2, 2, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.b3)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = conv2d(z, self.W4, [1, 1, 1, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.b4)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)
        
        z = tf.reshape(z, [-1, 7*7*256])
        logits = tf.matmul(z, self.W5)
        logits = tf.nn.bias_add(logits, self.b5)
        return logits


        
