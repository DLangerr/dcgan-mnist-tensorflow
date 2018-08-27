import numpy as np
import tensorflow as tf

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.02))

def init_bias(shape):
    return tf.Variable(tf.zeros(shape))

def conv2d(x, filter, strides, padding):
    return tf.nn.conv2d(x, filter, strides=strides, padding=padding)

def cost(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
