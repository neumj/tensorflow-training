#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from pylib import mnist_dataset

def _get_mnist(set_type):
    data = getattr(mnist_dataset, set_type)('/tmp/data{}'.format(set_type))
    x = []
    y = []
    iterator_full = data.batch(1000).make_one_shot_iterator()
    next_ = iterator_full.get_next()
    with tf.Session() as sess:
        try:
            while True:
                x_t, y_t = sess.run(next_)
                x.append(x_t)
                y.append(y_t)
        except tf.errors.OutOfRangeError:
            pass
    return np.vstack(x), np.hstack(y)

def mnist_train():
    return _get_mnist('train')

def mnist_test():
    return _get_mnist('test')
