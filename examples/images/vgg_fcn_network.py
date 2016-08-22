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

def bias_reshape(bweight, num_orig, num_new):
    n_averaged_elements = num_orig//num_new
    avg_bweight = np.zeros(num_new)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
    return avg_bweight

def summary_reshape(fweight, shape, num_new):
    num_orig = shape[3]
    shape[3] = num_new
    n_averaged_elements = num_orig//num_new
    avg_fweight = np.zeros(shape)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        avg_fweight[:, :, :, avg_idx] = np.mean(
            fweight[:, :, :, start_idx:end_idx], axis=3)
    return avg_fweight

def get_fc_weight_reshape(name, shape, num_classes=None):
    print('Layer name: %s' % name)
    print('Layer shape: %s' % shape)
    w = weights[name][0]
    w = w.reshape(shape)
    if num_classes is not None:
        w = summary_reshape(w, shape, num_new=num_classes)
    init = tf.constant_initializer(value=w,
                                   dtype=tf.float32)
    return init

def get_bias(name, num_classes=None):
    bias_wights = weights[name][1]
    shape = weights[name][1].shape
    if name == 'fc8':
        bias_wights = bias_reshape(bias_wights, shape[0],
                                         num_classes)
        shape = [num_classes]
    init = tf.constant_initializer(value=bias_wights,
                                   dtype=tf.float32)
    return init

def color_image(image, num_classes=20):
    import matplotlib as mpl
    from matplotlib import cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = cm.get_cmap('Set1')
    return mycm(norm(image))

def conv_layer(network, nb_filter, filter_size, layer, conv_shape=None):
    bias = get_bias(layer)
    weights = get_conv_filter(layer, conv_shape)
    return conv_2d(network, nb_filter, filter_size, activation='relu', 
      weights_init=weights, bias_init=bias)

def fc_layer(network, nb_filter, filter_size, layer, num_classes=None):
    shape = None
    if layer is 'fc6':
        shape = [7, 7, 512, 4096]
    elif layer is 'fc8':
        shape = [1, 1, 4096, 1000]
    weights = get_fc_weight_reshape(layer, shape, num_classes=num_classes)
    bias = get_bias(layer, num_classes=num_classes)
    fc = conv_2d(network, nb_filter, filter_size, weights_init=weights, bias_init=bias)
    return fc

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
network = conv_layer(bgr, 64, 3, 'conv1_1')
network = conv_layer(network, 64, 3, 'conv1_2')
network = max_pool_2d(network, 2, strides=2)

# Conv 2
network = conv_layer(network, 128, 3, 'conv2_1')
network = conv_layer(network, 128, 3, 'conv2_2')
network = max_pool_2d(network, 2, strides=2)

# Conv 3
network = conv_layer(network, 256, 3, 'conv3_1')
network = conv_layer(network, 256, 3, 'conv3_2')
network = conv_layer(network, 256, 3, 'conv3_3')
pool3 = max_pool_2d(network, 2, strides=2)

# Conv 4
network = conv_layer(pool3, 512, 3, 'conv4_1')
network = conv_layer(network, 512, 3, 'conv4_2')
network = conv_layer(network, 512, 3, 'conv4_3')
pool4 = max_pool_2d(network, 2, strides=2)

# Conv 5
network = conv_layer(pool4, 512, 3, 'conv5_1')
network = conv_layer(network, 512, 3, 'conv5_2')
network = conv_layer(network, 512, 3, 'conv5_3')
network = max_pool_2d(network, 2, strides=2)

# FC 6
network = fc_layer(network, 4096, 7, 'fc6')
if train:
  network = dropout(network, 0.5)

# FC 7
network = fc_layer(network, 4096, 1, 'fc7')
if train:
  network = dropout(network, 0.5)

# score_fr
score_fr = score_layer(network, num_classes=num_classes, name='score_fr')

pred = tf.argmax(score_fr, dimension=3)

# 1,27,27,20 (wronge feature dim)
upscore2 = upscore_layer(score_fr, num_classes, shape=tf.shape(pool4))

score_pool4 = score_layer(pool4, num_classes, name='score_pool4')

fuse_pool4 = tf.add(upscore2, score_pool4)
upscore4 = upscore_layer(fuse_pool4, num_classes, shape=tf.shape(pool3))

score_pool3 = score_layer(pool3, num_classes, name='score_pool3')
fuse_pool3 = tf.add(upscore4, score_pool3)

upscore32 = upscore_layer(fuse_pool3, num_classes)
pred_up = tf.argmax(fuse_pool3, dimension=3)

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

    misc.imsave('fcn8_downsampled.png', down_color)
    misc.imsave('fcn8_upsampled.png', up_color)

    # output = model.predict([resized])
    # down, up = model.session.run(tensors, feed_dict=feed_dict)

    # down_color = utils.color_image(down[0])
    # up_color = utils.color_image(up[0])

    # scp.misc.imsave('fcn32_downsampled.png', down_color)
    # scp.misc.imsave('fcn32_upsampled.png', up_color)

    #output = model.predict([resized])
    #writer = tf.python.training.summary_io.SummaryWriter("./log", model)
    print ('woot')
