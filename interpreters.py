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

from ops import *
from utils import *
#          _________________
#          \               /
#           \             /
#            \           /
#             INTERPRETER
#            /           \
#           /             \
#          /_______________\
                

def mnist_interpreter_no1(x, batch_size, is_training=True, reuse=False):
        
    with tf.compat.v1.variable_scope("interpreter", reuse=reuse):

        net = tf.nn.relu(coinv2d(x, 64, 4, 4, 2, 2, name='int_conv1'))
        net = tf.reshape(net, [batch_size, -1])
        code = (linear(net, 32, scope='int_fc6')) # bn and relu are excluded since code is used in pullaway_loss
        net = tf.nn.relu(bn(linear(code, 64 * 14 * 14, scope='int_fc3'), is_training=is_training, scope='int_bn3'))
        net = tf.reshape(net, [batch_size, 14, 14, 64])
        out = tf.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='int_dc5'))

        # recon loss
        recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x)) / batch_size
        return out, recon_error, code
