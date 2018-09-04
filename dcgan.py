import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from generator import Generator 
from discriminator import Discriminator
from ops import *
import os

class DCGAN:
    def __init__(self, img_shape, epochs=50000, lr_gen=0.0001, lr_disc=0.0001, z_shape=100, batch_size=64, beta1=0.5, epochs_for_sample=500):
        
       
        self.rows, self.cols, self.channels = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.z_shape = z_shape
        self.epochs_for_sample = epochs_for_sample
        self.generator = Generator(img_shape, self.batch_size)
        self.discriminator = Discriminator(img_shape)

        mnist = tf.keras.datasets.mnist 
        (x_train, _), (x_test, _) = mnist.load_data()

        X = np.concatenate([x_train, x_test])
        self.X = X / 127.5 - 1 # Scale between -1 and 1
        self.phX = tf.placeholder(tf.float32, [None, self.rows, self.cols])
        self.phZ = tf.placeholder(tf.float32, [None, self.z_shape])
    
        self.gen_out = self.generator.forward(self.phZ)

        disc_logits_fake = self.discriminator.forward(self.gen_out)
        disc_logits_real = self.discriminator.forward(self.phX)

        disc_fake_loss = cost(tf.zeros_like(disc_logits_fake), disc_logits_fake)
        disc_real_loss = cost(tf.ones_like(disc_logits_real), disc_logits_real)

        self.disc_loss = tf.add(disc_fake_loss, disc_real_loss)
        self.gen_loss = cost(tf.ones_like(disc_logits_fake), disc_logits_fake)

        train_vars = tf.trainable_variables()

        disc_vars = [var for var in train_vars if 'd' in var.name]
        gen_vars = [var for var in train_vars if 'g' in var.name]

        self.disc_train = tf.train.AdamOptimizer(lr_disc,beta1=beta1).minimize(self.disc_loss, var_list=disc_vars)
        self.gen_train = tf.train.AdamOptimizer(lr_gen, beta1=beta1).minimize(self.gen_loss, var_list=gen_vars)
        


    def train(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        for i in range(self.epochs):
            idx = np.random.randint(0, len(self.X), self.batch_size)
            batch_X = self.X[idx]
            batch_Z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))


            _, d_loss = self.sess.run([self.disc_train, self.disc_loss], feed_dict={self.phX:batch_X, self.phZ:batch_Z})
            batch_Z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
            _, g_loss = self.sess.run([self.gen_train, self.gen_loss], feed_dict={self.phZ: batch_Z})
            if i % self.epochs_for_sample == 0:
                self.generate_sample(i)
                print(f"Epoch: {i}. Discriminator loss: {d_loss}. Generator loss: {g_loss}")


    def generate_sample(self, epoch):
        c = 7
        r = 7
        z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
        imgs = self.sess.run(self.gen_out, feed_dict={self.phZ:z})
        imgs = imgs*0.5 + 0.5
        # scale between 0, 1
        fig, axs = plt.subplots(c, r)
        cnt = 0
        for i in range(c):
            for j in range(r):
                axs[i, j].imshow(imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("samples/%d.png" % epoch)
        plt.close()



if __name__ == '__main__':
    img_shape = (28, 28, 1)
    epochs = 50000
    dcgan = DCGAN(img_shape, epochs)

    if not os.path.exists('samples/'):
        os.makedirs('samples/')
    
    dcgan.train()
    
    
