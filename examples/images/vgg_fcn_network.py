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

import numpy as np
import tensorflow as tf
import tflearn
from scipy import misc
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upscore_layer, score_layer
from tflearn.layers.estimator import regression

weights =  np.load('vgg16.npy').item()

# Data loading and preprocessing
# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True)
train = False
num_classes = 17

def get_conv_filter(name, shape=None):
    data = weights[name][0]
    if shape:
      data = data.reshape(shape)
    init = tf.constant(value=data, dtype=tf.float32)
    return init


# Building 'VGG Network'
network = input_data(shape=[None, 224, 224, 3])

# Conv 1
network = conv_2d(network, 64, 3, activation='relu', weights_init=get_conv_filter('conv1_1'))
network = conv_2d(network, 64, 3, activation='relu', weights_init=get_conv_filter('conv1_2'))
network = max_pool_2d(network, 2, strides=2)

# Conv 2
network = conv_2d(network, 128, 3, activation='relu', weights_init=get_conv_filter('conv2_1'))
network = conv_2d(network, 128, 3, activation='relu', weights_init=get_conv_filter('conv2_2'))
network = max_pool_2d(network, 2, strides=2)

# Conv 3
network = conv_2d(network, 256, 3, activation='relu', weights_init=get_conv_filter('conv3_1'))
network = conv_2d(network, 256, 3, activation='relu', weights_init=get_conv_filter('conv3_2'))
network = conv_2d(network, 256, 3, activation='relu', weights_init=get_conv_filter('conv3_3'))
pool3 = max_pool_2d(network, 2, strides=2)

# Conv 4
network = conv_2d(pool3, 512, 3, activation='relu', weights_init=get_conv_filter('conv4_1'))
network = conv_2d(network, 512, 3, activation='relu', weights_init=get_conv_filter('conv4_2'))
network = conv_2d(network, 512, 3, activation='relu', weights_init=get_conv_filter('conv4_3'))
pool4 = max_pool_2d(network, 2, strides=2)

# Conv 5
network = conv_2d(pool4, 512, 3, activation='relu', weights_init=get_conv_filter('conv5_1'))
network = conv_2d(network, 512, 3, activation='relu', weights_init=get_conv_filter('conv5_2'))
network = conv_2d(network, 512, 3, activation='relu', weights_init=get_conv_filter('conv5_3'))
network = max_pool_2d(network, 2, strides=2)

# FC 6
network = conv_2d(network, 4096, 7, activation='relu', weights_init=get_conv_filter('fc6', shape=[7, 7, 512, 4096]))
network = dropout(network, 0.5)

# FC 7
network = conv_2d(network, 4096, 1, activation='relu', weights_init=get_conv_filter('fc7', shape=[1, 1, 4096, 4096]))
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

pred_up = tf.argmax(upscore32, dimension=3)

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
#model.save('model_vgg')

if train:
    model.fit(X, Y, n_epoch=500, shuffle=True,
              show_metric=True, batch_size=32, snapshot_step=500,
              snapshot_epoch=False, run_id='vgg_oxflowers17')
else:
    import pdb; pdb.set_trace()
    img = misc.imread("./tabby_cat.png")/255.0
    resized = misc.imresize(img, (224, 224))

    images = tf.placeholder("float")
    feed_dict = {images: img}
    batch_images = tf.expand_dims(images, 0)

    print('Finished building Network.')

    init = tf.initialize_all_variables()
    model.session.run(tf.initialize_all_variables())
    output = model.predict([resized])

    print('Running the Network')
    tensors = [pred, pred_up]
    output = model.predict([resized])
    down, up = model.session.run(tensors, feed_dict=feed_dict)

    # down_color = utils.color_image(down[0])
    # up_color = utils.color_image(up[0])

    # scp.misc.imsave('fcn32_downsampled.png', down_color)
    # scp.misc.imsave('fcn32_upsampled.png', up_color)

    #output = model.predict([resized])
    print ('woot')