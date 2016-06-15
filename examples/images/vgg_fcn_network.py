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
from tflearn import utils
from scipy import misc
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upscore_layer, score_layer
from tflearn.layers.estimator import regression

weights =  np.load('vgg16.npy').item()

# Data loading and preprocessing
# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True)
train = False
num_classes = 20
VGG_MEAN = [103.939, 116.779, 123.68]

def get_conv_filter(name, shape=None):
    data = weights[name][0]
    if shape:
      data = data.reshape(shape)
    init = tf.constant(value=data, dtype=tf.float32)
    return init

def color_image(image, num_classes=20):
    import matplotlib as mpl
    from matplotlib import cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = cm.get_cmap('Set1')
    return mycm(norm(image))


# Building 'VGG Network'
images_placeholder = input_data(shape=[None, 224, 224, 3])

red, green, blue = tf.split(3, 3, images_placeholder)
# assert red.get_shape().as_list()[1:] == [224, 224, 1]
# assert green.get_shape().as_list()[1:] == [224, 224, 1]
# assert blue.get_shape().as_list()[1:] == [224, 224, 1]
bgr = tf.concat(3, [
    blue - VGG_MEAN[0],
    green - VGG_MEAN[1],
    red - VGG_MEAN[2],
])

# Conv 1
network = conv_2d(bgr, 64, 3, activation='relu', weights_init=get_conv_filter('conv1_1'))
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
if train:
  network = dropout(network, 0.5)

# FC 7
network = conv_2d(network, 4096, 1, activation='relu', weights_init=get_conv_filter('fc7', shape=[1, 1, 4096, 4096]))
if train:
  network = dropout(network, 0.5)

score_fr = score_layer(pool4, num_classes=num_classes, name='score_fr')
pred = tf.argmax(score_fr, dimension=3)

upscore = upscore_layer(score_fr, num_classes=num_classes,
                                   name='up', kernel_size=64, strides=32)
pred_up = tf.argmax(upscore, dimension=3)


# pred = tf.argmax(network, dimension=3)

# upscore2 = upscore_layer(network,
#                          shape=tf.shape(pool4),
#                          num_classes=num_classes,
#                          kernel_size=4, strides=2, name='upscore2')

# score_pool4 = score_layer(pool4, num_classes=num_classes, name='score_pool4')
# fuse_pool4 = tf.add(upscore2, score_pool4)

# upscore4 = upscore_layer(fuse_pool4, num_classes=num_classes,
#                          kernel_size=4, strides=2, name='upscore4')

# score_pool3 = score_layer(pool3, num_classes=num_classes, name='score_pool3')
# fuse_pool3 = tf.add(upscore4, score_pool3)

# upscore32 = upscore_layer(fuse_pool3, num_classes=num_classes,
#                           kernel_size=16, strides=8, name='upscore32')

# pred_up = tf.argmax(upscore32, dimension=3)

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
#model.save('model_vgg')

with model.session.graph.as_default():
  if train:
    model.fit(X, Y, n_epoch=500, shuffle=True,
              show_metric=True, batch_size=32, snapshot_step=500,
              snapshot_epoch=False, run_id='vgg_oxflowers17')
  else:
    img = misc.imread("./tabby_cat.png")
    resized = misc.imresize(img, (224, 224))
    feed_dict = {images_placeholder: [resized]}

    print('Finished building Network.')

    #init = tf.initialize_all_variables()
    #model.session.run(tf.initialize_all_variables())

    print('Running the Network')
    tensors = [pred, pred_up]
    down, up = model.session.run(tensors, feed_dict=feed_dict)

    down_color = color_image(down[0])
    up_color = color_image(up[0])

    misc.imsave('fcn32_downsampled.png', down_color)
    misc.imsave('fcn32_upsampled.png', up_color)

    # output = model.predict([resized])
    # down, up = model.session.run(tensors, feed_dict=feed_dict)

    # down_color = utils.color_image(down[0])
    # up_color = utils.color_image(up[0])

    # scp.misc.imsave('fcn32_downsampled.png', down_color)
    # scp.misc.imsave('fcn32_upsampled.png', up_color)

    #output = model.predict([resized])
    print ('woot')