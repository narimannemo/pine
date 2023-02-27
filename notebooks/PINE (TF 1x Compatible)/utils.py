from __future__ import division
import math
import random
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import matplotlib.pyplot as plt
import os, gzip
import platform
from subprocess import check_output
from tensorflow.keras.datasets import cifar10

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import tensorflow as tf


def load_cifar10(dataset_name):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    x_train = X_train.astype('float32') / 255.
    x_test = X_test.astype('float32') / 255.
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).astype(np.int64)
    np.random.seed(547)
    np.random.shuffle(X)
    np.random.seed(547)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float32)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0
    return X, y_vec


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000,))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000,))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int32)

    np.random.seed(547)
    np.random.shuffle(X)
    np.random.seed(547)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float32)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X, y_vec


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.compat.v1.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale=False):
    if grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float32)
    else:
        return scipy.misc.imread(path).astype(np.float32)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3,4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c), dtype=np.float32)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]), dtype=np.float32)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('In merge(images, size), images parameter must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w]).astype(np.float32)

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width]).astype(np.float32)
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.) / 2.

def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    """
    Save a scatter plot of the latent variables with color-coded labels.

    Args:
    z (numpy array): a 2D array of size (num_samples, 2) containing the latent variables.
    id (numpy array): a 2D array of size (num_samples, num_classes) containing the labels.
    z_range_x (float): the range of values for the x-axis.
    z_range_y (float): the range of values for the y-axis.
    name (str): the filename to save the plot.

    Returns:
    None
    """
    num_classes = id.shape[1]
    colors = plt.cm.get_cmap('jet', num_classes)
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, axis=1), marker='o', edgecolor='none', cmap=colors)
    plt.colorbar(ticks=range(num_classes))
    plt.xlim([-z_range_x, z_range_x])
    plt.ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map.

    Args:
    N (int): the number of colors in the colormap.
    base_cmap (str or colormap, optional): the base colormap to use. Default is 'jet'.

    Returns:
    colormap: a matplotlib colormap instance.
    """
    if base_cmap is None:
        base_cmap = 'jet'
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


