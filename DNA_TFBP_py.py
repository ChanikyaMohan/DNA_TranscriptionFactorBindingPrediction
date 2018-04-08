from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
#dir_path = "C:/Users/konya/Desktop/DNA TFBP/DNA_TranscriptionFactorBindingPrediction"
filename = dir_path + "data/train.csv"

label = tf.placeholder(tf.int32, name='label')
sequence = tf.placeholder(tf.string, name='sequence')

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            _, sequence, label = line.strip().split(",")
            print(sequence, label)
