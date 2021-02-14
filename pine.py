from __future__ import division
import os
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import get_custom_objects

import pandas as pd

from ops import *
from utils import *
import main_models
import interpreters

class PINE(object):
    model_name = "PINE"     # name for checkpoint

    def __init__(self, sess, main_model, interpreter, epoch, batch_size, dataset_name, checkpoint_dir, result_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.result_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28
      
            self.y_dim = 10        
            self.c_dim = 1

            self.c1 = 10000
            self.c2 = 10000
            self.c3 = 1000000
            self.c4 = 0
            # train
            self.learning_rate = 0.0001
            self.beta1 = 0.5

            # test
            self.sample_num = 64  

            # load mnist
            self.data_X, self.data_X_test, self.data_y, self.data_y_test = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
            self.kcc = tf.keras.losses.CategoricalCrossentropy()

        elif dataset_name == 'cifar10':
            # parameters
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32
      
            self.y_dim = 10        
            self.c_dim = 3


            self.c1 = 10000
            self.c2 = 10000
            self.c3 = 1000000
            self.c4 = 10000
            # train
            self.learning_rate = 0.0001
            self.beta1 = 0.5

            # test
            self.sample_num = 64  

            # load mnist
            self.data_X, self.data_X_test, self.data_y, self.data_y_test = load_cifar10(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
            self.kcc = tf.keras.losses.CategoricalCrossentropy()
        else:
            raise NotImplementedError

        self.main_model = getattr(main_models,main_model)
        self.interpreter = getattr(interpreters,interpreter)
        
    def build_pine(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        ### Graph Input ###
        # images
        self.inputs = tf.compat.v1.placeholder(tf.float32, [bs] + image_dims, name='real_images')
	# test images
        self.samples_tests = tf.compat.v1.placeholder(tf.float32, [bs] + image_dims, name='test_images')
        # labels
        self.y = tf.compat.v1.placeholder(tf.float32, [bs, self.y_dim], name='y')
        # labels
        self.y_test = tf.compat.v1.placeholder(tf.float32, [bs, self.y_dim], name='y_test')
        ### Loss Function ###


        tafsir, tafsir_err, mask = self.interpreter(self.inputs, self.batch_size, is_training=True)
        out_mask, out_logit_mask = self.main_model(mask, self.batch_size, is_training=False)
        out_tafsir, out_logit_tafsir = self.main_model(tafsir, self.batch_size, is_training=False, reuse= True)
        out_real, out_logit_real = self.main_model(self.inputs, self.batch_size, is_training=True, reuse= True)        



        self.mm_loss = self.kcc(out_real,self.y)
        out_sqrt = tf.keras.backend.sqrt(mask)
        sumi = tf.keras.backend.sum(out_sqrt)**2
        self.int_loss = self.c1*tafsir_err + self.c2*self.kcc(out_tafsir, self.y) + sumi / self.c3 + self.c4*self.kcc(out_mask, self.y)


        ### Training ###

        t_vars = tf.compat.v1.trainable_variables()
        int_vars = [var for var in t_vars if 'int_' in var.name]
        mm_vars = [var for var in t_vars if 'mm_' in var.name]
  

        # optimizers
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):

            self.int_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
            .minimize(self.int_loss, var_list=int_vars)
            self.mm_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
            .minimize(self.mm_loss, var_list=mm_vars)


        ### Testing ###
        # for test
        self.test_images = self.interpreter(self.samples_tests, self.batch_size, is_training=False, reuse=True)
        ### Summary ###
        int_loss_sum = tf.compat.v1.summary.scalar("int_loss", self.int_loss)
        mm_loss_sum = tf.compat.v1.summary.scalar("mm_loss", self.mm_loss)

        self.int_sum = tf.compat.v1.summary.merge([int_loss_sum])
        self.mm_sum = tf.compat.v1.summary.merge([mm_loss_sum])


        #################################################### 
        #                                ________________  #
        #    ___________                \               /  #
        #   /           \    Parallel    \             /   #
        #  / MAIN  MODEL \      ||        \           /    #
        # /_______________\  Training      INTERPRETER     #
        #                                 /           \    #
        #                                /             \   #
        #                               /_______________\  #
        ####################################################           
    def train(self):

        # initialize all variables
        tf.compat.v1.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.test_codes = self.data_y[0:self.batch_size]
        self.sample_input = self.data_X[0:self.batch_size]

        # saver to save model
        self.saver = tf.compat.v1.train.Saver()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [i] OK. I've found it.")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [i] NOTHING FOUND TO LOAD!")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_codes = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]


                #update Interpreter
                _, summary_str, int_loss = self.sess.run([self.int_optim, self.int_sum, self.int_loss],
                                                       feed_dict={self.inputs: batch_images, self.y: batch_codes})

                # update Main Model
                _, summary_str_mm, mm_loss = self.sess.run(
                    [self.mm_optim, self.mm_sum, self.mm_loss],
                    feed_dict={self.y: batch_codes, self.inputs: batch_images})


                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, int_loss: %.8f,mm_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, int_loss, mm_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples, non, non2 = self.sess.run(self.test_images, feed_dict={self.samples_tests: self.data_X_test[0:self.batch_size]})
                    print(samples.shape)
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(self.data_X_test[:manifold_h * manifold_w], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}_original.png'.format(
                                    epoch, idx))
                    save_images(samples[:manifold_h * manifold_w], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}_interpretations.png'.format(
                                    epoch, idx))
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)
        # show temporal results
        self.save_results(epoch)

    def save_results(self, epoch):

        tot_num_samples = min(self.sample_num, self.batch_size)

        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        samples = self.sess.run(self.test_images, feed_dict={self.samples_tests: self.data_X_test[0:self.batch_size]})
        samples = samples.numpy()
        save_images(self.data_X_test[:image_frame_dim * image_frame_dim], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_original.png')
        save_images(samples[:image_frame_dim * image_frame_dim], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_interpretations.png')
    @property
    def model_dir(self):
        return "{}_{}".format(
            self.model_name, self.dataset_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [i] Wait a sec...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [i] OK. Reading Completed! {}".format(ckpt_name))
            return True, counter
        else:
            print(" [i] NO CHECKPOINTS FOUND!")
            return False, 0
    def just_load(self, tobe_tafsired):
        
        # initialize all variables
        tf.compat.v1.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.test_codes = self.data_y[0:self.batch_size]
        self.sample_input = self.data_X[0:self.batch_size]


        # saver to save model
        self.saver = tf.compat.v1.train.Saver()


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [i] Loading done!")
        samples, recon_error, code = self.sess.run(self.tafsir_images, feed_dict={self.inputs: tobe_tafsired})
        return samples

