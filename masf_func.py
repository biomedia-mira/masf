from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e, file=sys.stderr)

from tensorflow.python.platform import flags
from utils import conv_block, fc, max_pool, lrn, dropout
from utils import xent, kd

FLAGS = flags.FLAGS

class MASF:
    def __init__(self):
        """ Call construct_model_*() after initializing MASF"""
        self.inner_lr = FLAGS.inner_lr
        self.outer_lr = FLAGS.outer_lr
        self.metric_lr = FLAGS.metric_lr
        self.SKIP_LAYER = ['fc8']
        self.forward = self.forward_alexnet
        self.forward_metric_net = self.forward_metric_net
        self.construct_weights = self.construct_alexnet_weights
        self.loss_func = xent
        self.global_loss_func = kd
        self.WEIGHTS_PATH = '/path/to/pretrained_weights/bvlc_alexnet.npy'

    def construct_model_train(self, prefix='metatrain_'):
        # a: meta-train for inner update, b: meta-test for meta loss
        self.inputa = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.inputa1= tf.placeholder(tf.float32)
        self.labela1= tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)
        self.bool_indicator_b_a = tf.placeholder(tf.float32, shape=(7,))
        self.bool_indicator_b_a1 = tf.placeholder(tf.float32, shape=(7,))

        meta_sample_num = (FLAGS.meta_batch_size /3) * 3
        self.input_group = tf.placeholder(tf.float32)
        self.label_group = tf.placeholder(tf.int32, shape=(meta_sample_num,))

        self.clip_value = FLAGS.gradients_clip_value
        self.margin = FLAGS.margin
        self.KEEP_PROB = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                print('weights already defined')
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            def task_metalearn(inp, global_bool_indicator, reuse=True):
                # Function to perform meta learning update """
                inputa, inputa1, inputb, input_group, labela, labela1, labelb, label_group = inp
                global_bool_indicator_b_a, global_bool_indicator_b_a1 = global_bool_indicator

                # Obtaining the conventional task loss on meta-train
                _, task_outputa = self.forward(inputa, weights, reuse=reuse)
                task_lossa = self.loss_func(task_outputa, labela)
                _, task_outputa1 = self.forward(inputa1, weights, reuse=reuse)
                task_lossa1 = self.loss_func(task_outputa1, labela1)

                ## perform inner update with plain gradient descent on meta-train
                grads = tf.gradients((task_lossa + task_lossa1)/2.0, list(weights.values()))
                grads = [tf.stop_gradient(grad) for grad in grads] # first-order gradients approximation
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.inner_lr * tf.clip_by_norm(gradients[key], clip_norm=self.clip_value) for key in weights.keys()]))

                ## compute global loss
                _, new_task_outputa =  self.forward(inputa, fast_weights, reuse=reuse)
                _, new_task_outputa1 = self.forward(inputa1, fast_weights, reuse=reuse)
                _, task_outputb = self.forward(inputb, fast_weights, reuse=reuse)
                global_loss_b_a, _, _ = self.global_loss_func(task_outputb, labelb, new_task_outputa, labela, global_bool_indicator_b_a)
                global_loss_b_a1,_, _ = self.global_loss_func(task_outputb, labelb, new_task_outputa1,labela1,global_bool_indicator_b_a1)
                global_loss = (global_loss_b_a + global_loss_b_a1) / 2.0

                ## compute local loss
                embeddings, _ = self.forward(input_group, fast_weights, reuse=True)
                embeddings = self.forward_metric_net(embeddings)
                metric_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=label_group, embeddings=embeddings, margin=self.margin)

                task_output = [global_loss, task_lossa, task_lossa1, metric_loss]
                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1)) #this accuracy already gathers batch size
                task_accuracya1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa1), 1), tf.argmax(labela1, 1))
                task_output.extend([task_accuracya, task_accuracya1])

                return task_output

            self.global_step = tf.Variable(0, trainable=False)
            # self.inner_lr = tf.train.exponential_decay(learning_rate=FLAGS.inner_lr, global_step=self.global_step, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate)

            input_tensors = (self.inputa, self.inputa1, self.inputb, self.input_group, self.labela, self.labela1, self.labelb, self.label_group)
            global_bool_indicator = (self.bool_indicator_b_a, self.bool_indicator_b_a1)
            result = task_metalearn(inp=input_tensors, global_bool_indicator=global_bool_indicator)
            global_loss, self.lossa_raw, self.lossa1_raw, metric_loss, accuracya, accuracya1 = result
            self.global_loss = global_loss * 1.0
            self.metric_loss = metric_loss * 0.005

        ## Performance & Optimization
        if 'train' in prefix:
            self.lossa = avg_lossa = tf.reduce_mean(self.lossa_raw)
            self.lossa1 = avg_lossa1 = tf.reduce_mean(self.lossa1_raw)
            self.source_loss = (avg_lossa + avg_lossa1) / 2.0
            self.task_train_op = tf.train.AdamOptimizer(learning_rate=self.outer_lr).minimize(self.source_loss, global_step=self.global_step)

            self.accuracya = accuracya * 100.
            self.accuracya1 = accuracya1 * 100.
            self.source_accuracy = (self.accuracya + self.accuracya1) / 2.0

            var_list_feature_extractor = [v for v in tf.trainable_variables() if (v.name.split('/')[1] not in self.SKIP_LAYER and 'meta' not in v.name.split('/'))]
            var_list_classifier = [v for v in tf.trainable_variables() if v.name.split('/')[1] in self.SKIP_LAYER]
            var_list_metric = [v for v in tf.trainable_variables() if 'metric' in v.name.split('/')]

            optimizer = tf.train.AdamOptimizer(self.outer_lr)
            gvs = optimizer.compute_gradients(self.global_loss+self.metric_loss, var_list=var_list_feature_extractor+var_list_classifier)

            # observe stability of gradients for meta loss
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for grad, var in gvs:
                tf.summary.histogram("gradients_norm/" + var.name, l2_norm(grad))
                tf.summary.histogram("feature_extractor_var_norm/" + var.name, l2_norm(var))
                tf.summary.histogram('gradients/' + var.name, var)
                tf.summary.histogram("feature_extractor_var/" + var.name, var)

            gvs = [(tf.clip_by_norm(grad, clip_norm=self.clip_value), var) for grad, var in gvs]

            for grad, var in gvs:
                tf.summary.histogram("gradients_norm_clipped/" + var.name, l2_norm(grad))
                tf.summary.histogram('gradients_clipped/' + var.name, var)

            self.meta_train_op = optimizer.apply_gradients(gvs)
            self.metric_train_op = tf.train.AdamOptimizer(self.metric_lr).minimize(self.metric_loss, var_list=var_list_metric)

        ## Summaries
        tf.summary.scalar(prefix+'source_1 loss', self.lossa)
        tf.summary.scalar(prefix+'source_2 loss', self.lossa1)
        tf.summary.scalar(prefix+'source_1 accuracy', self.accuracya)
        tf.summary.scalar(prefix+'source_2 accuracy', self.accuracya1)
        tf.summary.scalar(prefix+'global loss', self.global_loss)
        tf.summary.scalar(prefix+'metric loss', self.metric_loss)

    def construct_model_test(self, prefix='test'):

        self.test_input = tf.placeholder(tf.float32)
        self.test_label = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as testing_scope:
            if 'weights' in dir(self):
                testing_scope.reuse_variables()
                weights = self.weights
            else:
                raise ValueError('Weights not initilized. Create training model before testing model')

            self.semantic_feature, outputs = self.forward(self.test_input, weights)
            self.metric_embedding = self.forward_metric_net(self.semantic_feature)
            losses = self.loss_func(outputs, self.test_label)
            accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), 1), tf.argmax(self.test_label, 1))
            self.pred_prob = tf.nn.softmax(outputs)
            self.outputs = outputs

        self.test_loss = losses
        self.test_acc = accuracies

    def load_initial_weights(self, session):
        """Load weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        The weights come as a dict of lists (e.g. weights['conv1'] is a list)
        Load the weights into the model
        """
        weights_dict = np.load(self.WEIGHTS_PATH, allow_pickle= True, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope('model', reuse=True):
                    with tf.variable_scope(op_name, reuse=True):

                        for data in weights_dict[op_name]:
                            # Biases
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases', trainable=True)
                                session.run(var.assign(data))
                            # Weights
                            else:
                                var = tf.get_variable('weights', trainable=True)
                                session.run(var.assign(data))

    def forward_metric_net(self, x):

        with tf.variable_scope('metric', reuse=tf.AUTO_REUSE) as scope:

            w1 = tf.get_variable('w1', shape=[4096,1024])
            b1 = tf.get_variable('b1', shape=[1024])
            out = fc(x, w1, b1, activation='leaky_relu')
            w2 = tf.get_variable('w2', shape=[1024,256])
            b2 = tf.get_variable('b2', shape=[256])
            out = fc(out, w2, b2, activation='leaky_relu')

        return out

    def construct_alexnet_weights(self):

        weights = {}
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

        with tf.variable_scope('conv1') as scope:
            weights['conv1_weights'] = tf.get_variable('weights', shape=[11, 11, 3, 96], initializer=conv_initializer)
            weights['conv1_biases'] = tf.get_variable('biases', [96])

        with tf.variable_scope('conv2') as scope:
            weights['conv2_weights'] = tf.get_variable('weights', shape=[5, 5, 48, 256], initializer=conv_initializer)
            weights['conv2_biases'] = tf.get_variable('biases', [256])

        with tf.variable_scope('conv3') as scope:
            weights['conv3_weights'] = tf.get_variable('weights', shape=[3, 3, 256, 384], initializer=conv_initializer)
            weights['conv3_biases'] = tf.get_variable('biases', [384])

        with tf.variable_scope('conv4') as scope:
            weights['conv4_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 384], initializer=conv_initializer)
            weights['conv4_biases'] = tf.get_variable('biases', [384])

        with tf.variable_scope('conv5') as scope:
            weights['conv5_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 256], initializer=conv_initializer)
            weights['conv5_biases'] = tf.get_variable('biases', [256])

        with tf.variable_scope('fc6') as scope:
            weights['fc6_weights'] = tf.get_variable('weights', shape=[9216, 4096], initializer=conv_initializer)
            weights['fc6_biases'] = tf.get_variable('biases', [4096])

        with tf.variable_scope('fc7') as scope:
            weights['fc7_weights'] = tf.get_variable('weights', shape=[4096, 4096], initializer=conv_initializer)
            weights['fc7_biases'] = tf.get_variable('biases', [4096])

        with tf.variable_scope('fc8') as scope:
            weights['fc8_weights'] = tf.get_variable('weights', shape=[4096, 7], initializer=fc_initializer)
            weights['fc8_biases'] = tf.get_variable('biases', [7])

        return weights

    def forward_alexnet(self, inp, weights, reuse=False):
        # reuse is for the normalization parameters.

        conv1 = conv_block(inp, weights['conv1_weights'], weights['conv1_biases'], stride_y=4, stride_x=4, groups=1, reuse=reuse, scope='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75)
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv_block(pool1, weights['conv2_weights'], weights['conv2_biases'], stride_y=1, stride_x=1, groups=2, reuse=reuse, scope='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75)
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv_block(pool2, weights['conv3_weights'], weights['conv3_biases'], stride_y=1, stride_x=1, groups=1, reuse=reuse, scope='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv_block(conv3, weights['conv4_weights'], weights['conv4_biases'], stride_y=1, stride_x=1, groups=2, reuse=reuse, scope='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv_block(conv4, weights['conv5_weights'], weights['conv5_biases'], stride_y=1, stride_x=1, groups=2, reuse=reuse, scope='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, weights['fc6_weights'], weights['fc6_biases'], activation='relu')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, weights['fc7_weights'], weights['fc7_biases'], activation='relu')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, weights['fc8_weights'], weights['fc8_biases'])

        return fc7, fc8
