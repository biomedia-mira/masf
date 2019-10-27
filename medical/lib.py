'''
U-Net shaped architecture for medical image segmentation, we reduced the model scale for computation efficiency
'''

import tensorflow as tf

def construct_unet_weights(self):

    weights = {}
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)

    with tf.variable_scope('conv1') as scope:
        weights['conv1_weights'] = tf.get_variable('weights', shape=[5, 5, 3, 16], initializer=conv_initializer)
        weights['conv1_biases'] = tf.get_variable('biases', [16])

    with tf.variable_scope('conv2') as scope:
        weights['conv2_weights'] = tf.get_variable('weights', shape=[5, 5, 16, 32], initializer=conv_initializer)
        weights['conv2_biases'] = tf.get_variable('biases', [32])

    ## Network has downsample here

    with tf.variable_scope('conv3') as scope:
        weights['conv3_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=conv_initializer)
        weights['conv3_biases'] = tf.get_variable('biases', [64])

    with tf.variable_scope('conv4') as scope:
        weights['conv4_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
        weights['conv4_biases'] = tf.get_variable('biases', [64])

    ## Network has downsample here

    with tf.variable_scope('conv5') as scope:
        weights['conv5_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=conv_initializer)
        weights['conv5_biases'] = tf.get_variable('biases', [128])

    # with tf.variable_scope('conv6') as scope:
    #     weights['conv6_weights'] = tf.get_variable('weights', shape=[3, 3, 128, 128], initializer=conv_initializer)
    #     weights['conv6_biases'] = tf.get_variable('biases', [128])

    with tf.variable_scope('deconv1') as scope:
        weights['deconv1_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=conv_initializer)

    with tf.variable_scope('conv7') as scope:
        weights['conv7_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
        weights['conv7_biases'] = tf.get_variable('biases', [64])

    with tf.variable_scope('conv8') as scope:
        weights['conv8_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
        weights['conv8_biases'] = tf.get_variable('biases', [64])

    with tf.variable_scope('deconv2') as scope:
        weights['deconv2_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=conv_initializer)

    # with tf.variable_scope('conv9') as scope:
    #     weights['conv9_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 32], initializer=conv_initializer)
    #     weights['conv9_biases'] = tf.get_variable('biases', [32])

    with tf.variable_scope('conv10') as scope:
        weights['conv10_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 32], initializer=conv_initializer)
        weights['conv10_biases'] = tf.get_variable('biases', [32])

    with tf.variable_scope('output') as scope:
        weights['output_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 4], initializer=conv_initializer)

    return weights


def forward_unet(self, inp, weights):

    self.conv1 = conv_block(inp, weights['conv1_weights'], weights['conv1_biases'])
    self.conv2 = conv_block(self.conv1, weights['conv2_weights'], weights['conv2_biases'])
    self.pool2 = max_pool(self.conv2, 2, 2, 2, 2, padding='VALID')

    self.conv3 = conv_block(self.pool2, weights['conv3_weights'], weights['conv3_biases'])
    self.conv4 = conv_block(self.conv3, weights['conv4_weights'], weights['conv4_biases'])
    self.pool4 = max_pool(self.conv4, 2, 2, 2, 2, padding='VALID')

    self.conv5 = conv_block(self.pool4, weights['conv5_weights'], weights['conv5_biases'])
    # self.conv6 = conv_block(self.conv5, weights['conv6_weights'], weights['conv6_biases'])

    ## add upsampling, meanwhile, channel number is reduced to half
    # self.up1 = deconv_block(self.conv6, weights['deconv1_weights'])
    self.up1 = deconv_block(self.conv5, weights['deconv1_weights'])

    self.conv7 = conv_block(self.up1, weights['conv7_weights'], weights['conv7_biases'])
    self.conv8 = conv_block(self.conv7, weights['conv8_weights'], weights['conv8_biases'])

    ## add upsampling, meanwhile, channel number is reduced to half
    self.up2 = deconv_block(self.conv8, weights['deconv2_weights'])

    # self.conv9 = conv_block(self.up2, weights['conv9_weights'], weights['conv9_biases'])
    # self.conv10 = conv_block(self.conv9, weights['conv10_weights'], weights['conv10_biases'])
    self.conv10 = conv_block(self.up2, weights['conv10_weights'], weights['conv10_biases'])

    self.logits = tf.nn.conv2d(self.conv10, weights['output_weights'], strides=[1, 1, 1, 1], padding='SAME')

    self.pred_prob = tf.nn.softmax(self.logits) # shape [batch, w, h, num_classes]
    self.pred_compact = tf.argmax(self.pred_prob, axis=-1) # shape [batch, w, h]

    return self.conv10, self.logits, self.pred_prob, self.pred_compact


# # Network blocks
def conv_block(inp, cweight, bweight):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    conv = tf.nn.conv2d(inp, cweight, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, bweight)
    relu = tf.nn.leaky_relu(conv)
    return relu


def deconv_block(inp, cweight):
    # x_shape = tf.shape(inp)
    x_shape = inp.get_shape().as_list()
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    deconv = tf.nn.conv2d_transpose(inp, cweight, output_shape, strides=[1,2,2,1], padding='SAME')
    return deconv
