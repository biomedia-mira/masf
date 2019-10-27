""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

import itertools

FLAGS = flags.FLAGS

## Image reader
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network blocks
def conv_block(inp, cweight, bweight, stride_y=2, stride_x=2, groups=1, reuse=False, scope=''):
    stride = [1, stride_y, stride_x, 1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=stride, padding='SAME')

    if groups==1:
        conv_output = tf.nn.bias_add(convolve(inp, cweight), bweight)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=inp)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=cweight)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        conv = tf.concat(axis=3, values=output_groups)
        conv_output = tf.nn.bias_add(conv, bweight)

    relu = tf.nn.relu(conv_output)

    return relu

def normalize(inp, activation, reuse, scope):
    return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)

def max_pool(x, filter_height, filter_width, stride_y, stride_x, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding)

def lrn(x, radius, alpha, beta, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

def fc(x, wweight, bweight, activation=None):
    """Create a fully connected layer."""
    
    act = tf.nn.xw_plus_b(x, wweight, bweight)

    if activation is 'relu':
        return tf.nn.relu(act)
    elif activation is 'leaky_relu':
        return tf.nn.leaky_relu(act)
    elif activation is None:
        return act
    else:
        raise NotImplementedError

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)

def kd(data1, label1, data2, label2, bool_indicator, n_class=7, temperature=2.0):

    kd_loss = 0.0
    eps = 1e-16

    prob1s = []
    prob2s = []

    for cls in range(n_class):
        mask1 = tf.tile(tf.expand_dims(label1[:, cls], -1), [1, n_class])
        logits_sum1 = tf.reduce_sum(tf.multiply(data1, mask1), axis=0)
        num1 = tf.reduce_sum(label1[:, cls])
        activations1 = logits_sum1 * 1.0 / (num1 + eps) # add eps for prevent un-sampled class resulting in NAN
        prob1 = tf.nn.softmax(activations1 / temperature)
        prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN

        mask2 = tf.tile(tf.expand_dims(label2[:, cls], -1), [1, n_class])
        logits_sum2 = tf.reduce_sum(tf.multiply(data2, mask2), axis=0)
        num2 = tf.reduce_sum(label2[:, cls])
        activations2 = logits_sum2 * 1.0 / (num2 + eps)
        prob2 = tf.nn.softmax(activations2 / temperature)
        prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)

        KL_div = (tf.reduce_sum(prob1 * tf.log(prob1 / prob2)) + tf.reduce_sum(prob2 * tf.log(prob2 / prob1))) / 2.0
        kd_loss += KL_div * bool_indicator[cls]

        prob1s.append(prob1)
        prob2s.append(prob2)

    kd_loss = kd_loss / n_class

    return kd_loss, prob1s, prob2s

def JS(data1, label1, data2, label2, bool_indicator, n_class=7, temperature=2.0):

    kd_loss = 0.0
    eps = 1e-16

    prob1s = []
    prob2s = []

    for cls in range(n_class):
        mask1 = tf.tile(tf.expand_dims(label1[:, cls], -1), [1, n_class])
        logits_sum1 = tf.reduce_sum(tf.multiply(data1, mask1), axis=0)
        num1 = tf.reduce_sum(label1[:, cls])
        activations1 = logits_sum1 * 1.0 / (num1 + eps) # add eps for prevent un-sampled class resulting in NAN
        prob1 = tf.nn.softmax(activations1 / temperature)
        prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN

        mask2 = tf.tile(tf.expand_dims(label2[:, cls], -1), [1, n_class])
        logits_sum2 = tf.reduce_sum(tf.multiply(data2, mask2), axis=0)
        num2 = tf.reduce_sum(label2[:, cls])
        activations2 = logits_sum2 * 1.0 / (num2 + eps)
        prob2 = tf.nn.softmax(activations2 / temperature)
        prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)

        mean_prob = (prob1 + prob2) / 2

        JS_div = (tf.reduce_sum(prob1 * tf.log(prob1 / mean_prob)) + tf.reduce_sum(prob2 * tf.log(prob2 / mean_prob))) / 2.0
        kd_loss += JS_div * bool_indicator[cls]

        prob1s.append(prob1)
        prob2s.append(prob2)

    kd_loss = kd_loss / n_class

    return kd_loss, prob1s, prob2s

def contrastive(feature1, label1, feature2, label2, bool_indicator=None, margin=50):

    l1 = tf.argmax(label1, axis=1)
    l2 = tf.argmax(label2, axis=1)
    pair = tf.to_float(tf.equal(l1,l2))

    delta = tf.reduce_sum(tf.square(feature1-feature2), 1) + 1e-10
    match_loss = delta

    delta_sqrt = tf.sqrt(delta + 1e-10)
    mismatch_loss = tf.square(tf.nn.relu(margin - delta_sqrt))

    if bool_indicator is None:
        loss = tf.reduce_mean(0.5 * (pair * match_loss + (1-pair) * mismatch_loss))
    else:
        loss = 0.5 * tf.reduce_sum(match_loss*pair)/tf.reduce_sum(pair)

    debug_dist_positive = tf.reduce_sum(delta_sqrt * pair)/tf.reduce_sum(pair)
    debug_dist_negative = tf.reduce_sum(delta_sqrt * (1-pair))/tf.reduce_sum(1-pair)

    return loss, pair, delta, debug_dist_positive, debug_dist_negative

def compute_distance(feature1, label1, feature2, label2):
    l1 = tf.argmax(label1, axis=1)
    l2 = tf.argmax(label2, axis=1)
    pair = tf.to_float(tf.equal(l1,l2))

    delta = tf.reduce_sum(tf.square(feature1-feature2), 1)
    delta_sqrt = tf.sqrt(delta + 1e-16)

    dist_positive_pair = tf.reduce_sum(delta_sqrt * pair)/tf.reduce_sum(pair)
    dist_negative_pair = tf.reduce_sum(delta_sqrt * (1-pair))/tf.reduce_sum(1-pair)

    return dist_positive_pair, dist_negative_pair
