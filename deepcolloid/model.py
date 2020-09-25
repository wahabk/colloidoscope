from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import tensorflow as tf
import numpy as np
from scipy import rand

MOVING_AVERAGE_DECAY = 0.999


def _variable_on_cpu(name, shape, initializer, trainable=True):
    #with tf.device('/cpu:1'):
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var
def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    var = _variable_on_cpu(name, shape, initializer, trainable)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
Winit = tf.contrib.layers.xavier_initializer_conv2d
def nonLinearity(conv, biases):
    return tf.nn.softplus(tf.nn.bias_add(conv, biases))
def makeKernel(shape, name='weights', wd=None, dprob=1., trainable=True):
    kernel = _variable_with_weight_decay(name, shape=shape, wd=wd,
                                         initializer = Winit(),
                                         trainable=trainable)
    return kernel if dprob==1 else tf.nn.dropout(kernel, dprob)
def makeBiases(nChannels, name='biases', bias_init=0.001, trainable=True):
    return _variable_on_cpu(name, [nChannels],
                            tf.constant_initializer(bias_init),
                            trainable=trainable)
def network(currentFrame, previousFrame, first=False, batch_size=1, wd=None, dprob=1., tflag=True):
    imagesShape = tf.shape(currentFrame)
    imgSize = tf.slice(imagesShape, [1], [2])
    ## conv1
    with tf.variable_scope('conv1') as scope:
        kernel1 = makeKernel(shape=[9, 9, 1, 3], name='weights1', wd=wd, dprob=dprob, trainable=tflag)
        kernel2 = makeKernel(shape=[5, 5, 1, 3], name='weights2', wd=wd, dprob=dprob, trainable=tflag)
        kernel3 = makeKernel(shape=[3, 3, 1, 3], name='weights3', wd=wd, dprob=dprob, trainable=tflag)
        biases = makeBiases(9, trainable=tflag)
        conv1 = tf.nn.conv2d(currentFrame, kernel1, [1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.conv2d(currentFrame, kernel2, [1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.conv2d(currentFrame, kernel3, [1, 1, 1, 1], padding='SAME')
        pool1 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.concat([conv1, conv2, pool1], 3)
        toConv2 = nonLinearity(conv, biases)
    ################################################################
    ## conv2
    with tf.variable_scope('conv2') as scope:
        kernel2 = makeKernel(shape=[7, 7, 6, 6], name='weights2', wd=wd, dprob=dprob, trainable=tflag)
        kernel4 = makeKernel(shape=[7, 7, 9, 6], name='weights4', wd=wd, dprob=dprob, trainable=tflag)
        kernel5 = makeKernel(shape=[3, 3, 9, 6], name='weights5', wd=wd, dprob=dprob, trainable=tflag)
        biases = makeBiases(18, trainable=tflag)
        biasesRNN = makeBiases(6, name='biasesRNN', trainable=tflag)

        conv4 = tf.nn.atrous_conv2d(toConv2, kernel4, rate=1, padding='SAME')
        # conv1 = tf.nn.atrous_conv2d(toConv2, kernel1, rate=4, padding='SAME')
        preConv = tf.nn.atrous_conv2d(nonLinearity(conv4, biasesRNN), kernel2, rate=2, padding='SAME')
        toNextFrame = tf.nn.atrous_conv2d(preConv, kernel2, rate=3, padding='SAME')

        conv5 = tf.nn.conv2d(toConv2, kernel5, [1, 1, 1, 1], padding='SAME')
        if first:
            conv = tf.concat([toNextFrame, conv4, conv5], 3)
        else:
            conv = tf.concat([previousFrame, conv4, conv5], 3)
        toConv3 = nonLinearity(conv, biases)
    ################################################################
    ## conv3
    with tf.variable_scope('conv3') as scope:
        kernel = makeKernel([5, 5, 18, 2], wd=wd, dprob=dprob, trainable=tflag)
        biases = makeBiases(2, trainable=tflag)
        conv = tf.nn.conv2d(toConv3, kernel, [1, 1, 1, 1], padding='SAME')
        toIntrp1 = nonLinearity(conv, biases)
    ################################################################
    ## interpolate 1
    # with tf.variable_scope('intrp1') as scope:
        # Ikernel1 = makeKernelinterp([3, 3, 5, 5], name='Iweights1', dprob=1, trainable=tflag)
        # finalOut = convolutionTranspose(toIntrp1, Ikernel1, [1, 2, 2, 1], img_size, channel_size = 5)
    finalOut = tf.image.resize_images(toIntrp1, imgSize)
    return finalOut, toNextFrame
################################################################
################################################################
################################################################

