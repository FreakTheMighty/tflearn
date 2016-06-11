# -*- coding: utf-8 -*-

""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.

VGG 16-layers convolutional with semantic segmentation

References:
    Fully Convolutional Networks for Semantic Segmentation
    Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015.

Links:
    https://arxiv.org/abs/1605.06211

"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upscore_layer, score_layer
from tflearn.layers.estimator import regression

# Data loading and preprocessing
# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True)
train = False
num_classes = 17

# Building 'VGG Network'
network = input_data(shape=[None, 224, 224, 3])

# Conv 1
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

# Conv 2
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

# Conv 3
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
pool3 = max_pool_2d(network, 2, strides=2)

# Conv 4
network = conv_2d(pool3, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
pool4 = max_pool_2d(network, 2, strides=2)

# Conv 5
network = conv_2d(pool4, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

# FC 6
network = conv_2d(network, 4096, 7, activation='relu')
network = dropout(network, 0.5)

# FC 7
network = conv_2d(network, 1000, 1, activation='relu')
network = dropout(network, 0.5)

pred = tf.argmax(network, dimension=3)

upscore2 = upscore_layer(network,
                         shape=tf.shape(pool4),
                         num_classes=num_classes,
                         kernel_size=4, strides=2, name='upscore2')

score_pool4 = score_layer(pool4, num_classes=num_classes, name='score_pool4')
fuse_pool4 = tf.add(upscore2, score_pool4)

upscore4 = upscore_layer(fuse_pool4, num_classes=num_classes,
                         kernel_size=4, strides=2, name='upscore4')

score_pool3 = score_layer(pool3, num_classes=num_classes, name='score_pool3')
fuse_pool3 = tf.add(upscore4, score_pool3)

upscore32 = upscore_layer(fuse_pool3, num_classes=num_classes,
                          kernel_size=16, strides=8, name='upscore32')


network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)

if train:
    model.fit(X, Y, n_epoch=500, shuffle=True,
              show_metric=True, batch_size=32, snapshot_step=500,
              snapshot_epoch=False, run_id='vgg_oxflowers17')