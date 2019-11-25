import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from data_generator import ImageDataGenerator
from masf_func import MASF

FLAGS = flags.FLAGS

if not str(sys.argv[1]):
    print('No GPU given... setting to 0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])

## Dataset PACS
flags.DEFINE_string('dataset', 'pacs', 'set the dataset of PACS')
flags.DEFINE_string('target_domain', 'art_painting', 'set the target domain')
flags.DEFINE_string('dataroot', '/path/to/PACS_dataset/kfold/', 'Root folder where PACS dataset is stored')
flags.DEFINE_integer('num_classes', 7, 'number of classes used in classification.')

## Training options
flags.DEFINE_integer('train_iterations', 10000, 'number of training iterations.')
flags.DEFINE_integer('meta_batch_size', 128, 'number of images sampled per source domain')
flags.DEFINE_float('inner_lr', 1e-5, 'step size alpha for inner gradient update on meta-train')
flags.DEFINE_float('outer_lr', 1e-5, 'learning rate for outer updates with (task-loss + meta-loss)')
flags.DEFINE_float('metric_lr', 1e-5, 'learning rate for the metric embedding nn with AdamOptimizer')
flags.DEFINE_float('margin', 10, 'distance margin in metric loss')
flags.DEFINE_bool('clipNorm', True, 'if True, gradients clip by Norm, otherwise, gradients clip by value')
flags.DEFINE_float('gradients_clip_value', 2.0, 'clip_by_value for SGD computing new theta at meta loss')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/log/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_integer('summary_interval', 100, 'frequency for logging training summaries')
flags.DEFINE_integer('save_interval', 20000, 'intervals to save model')
flags.DEFINE_integer('print_interval', 100, 'intervals to print out training info')
flags.DEFINE_integer('test_print_interval', 100, 'intervals to test the model')

class_list = {'0': 'dog',
              '1': 'elephant',
              '2': 'giraffe',
              '3': 'guitar',
              '4': 'horse',
              '5': 'house',
              '6': 'person'}

def train(model, saver, sess, exp_string, train_file_list, test_file, resume_itr=0):

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    source_losses, target_losses, source_accuracies, target_accuracies = [], [], [], []

    # Data loaders
    with tf.device('/cpu:0'):
        tr_data_list, train_iterator_list, train_next_list = [],[],[]
        for i in range(len(train_file_list)):
            tr_data = ImageDataGenerator(train_file_list[i], dataroot=FLAGS.dataroot, mode='training', \
                                         batch_size=FLAGS.meta_batch_size, num_classes=FLAGS.num_classes, shuffle=True)
            tr_data_list.append(tr_data)
            train_iterator_list.append(tf.data.Iterator.from_structure(tr_data.data.output_types,tr_data.data.output_shapes))
            train_next_list.append(train_iterator_list[i].get_next())

        test_data = ImageDataGenerator(test_file, dataroot=FLAGS.dataroot, mode='inference', \
                                       batch_size=1, num_classes=FLAGS.num_classes, shuffle=False)
        test_iterator = tf.data.Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
        test_next_batch = test_iterator.get_next()

    # Ops for initializing different iterators
    training_init_op = []
    train_batches_per_epoch = []
    for i in range(len(train_file_list)):
        training_init_op.append(train_iterator_list[i].make_initializer(tr_data_list[i].data))
        train_batches_per_epoch.append(int(np.floor(tr_data_list[i].data_size/FLAGS.meta_batch_size)))

    test_init_op = test_iterator.make_initializer(test_data.data)
    test_batches_per_epoch = int(np.floor(test_data.data_size / 1))

    # Training begins
    best_test_acc = 0
    for itr in range(resume_itr, FLAGS.train_iterations):

        # Sampling training and test tasks
        num_training_tasks = len(train_file_list)
        num_meta_train = num_training_tasks-1
        num_meta_test = num_training_tasks-num_meta_train  # as setting num_meta_test = 1
        
        # Randomly choosing meta train and meta test domains
        task_list = np.random.permutation(num_training_tasks)
        meta_train_index_list = task_list[:num_meta_train]
        meta_test_index_list = task_list[num_meta_train:]
        
        for i in range(len(train_file_list)):
            if itr%train_batches_per_epoch[i] == 0:
                sess.run(training_init_op[i])  # initialize training sample generator at itr=0

        # Sampling meta-train, meta-test data
        for i in range(num_meta_train):
            task_ind = meta_train_index_list[i]
            if i == 0:
                inputa, labela = sess.run(train_next_list[task_ind])
            elif i == 1:
                inputa1, labela1 = sess.run(train_next_list[task_ind])
            else:
                raise RuntimeError('check number of meta-train domains.')

        for i in range(num_meta_test):
            task_ind = meta_test_index_list[i]
            if i == 0:
                inputb, labelb = sess.run(train_next_list[task_ind])
            else:
                raise RuntimeError('check number of meta-test domains.')

        # to avoid a certain un-sampled class affect stability of of global class alignment
        # i.e., mask-out the un-sampled class from computing kd-loss
        sampledb = np.unique(np.argmax(labelb, axis=1))
        sampleda = np.unique(np.argmax(labela, axis=1))
        bool_indicator_b_a = [0.0] * FLAGS.num_classes
        for i in range(FLAGS.num_classes):
            # only count class that are sampled in both source domains
            if (i in sampledb) and (i in sampleda):
                bool_indicator_b_a[i] = 1.0

        sampledb = np.unique(np.argmax(labelb, axis=1))
        sampleda1 = np.unique(np.argmax(labela1, axis=1))
        bool_indicator_b_a1 = [0.0] * FLAGS.num_classes
        for i in range(FLAGS.num_classes):
            if (i in sampledb) and (i in sampleda1):
                bool_indicator_b_a1[i] = 1.0

        part = FLAGS.meta_batch_size / 3
        input_group = np.concatenate((inputa[:part],inputa1[:part],inputb[:part]), axis=0)
        label_group = np.concatenate((labela[:part],labela1[:part],labelb[:part]), axis=0)
        group_list = np.sum(label_group, axis=0)
        label_group = np.argmax(label_group, axis=1)  # transform one-hot labels into class-wise integer

        feed_dict = {model.inputa: inputa, model.labela: labela, \
                     model.inputa1: inputa1, model.labela1: labela1, \
                     model.inputb: inputb, model.labelb: labelb, \
                     model.input_group: input_group, model.label_group: label_group,\
                     model.bool_indicator_b_a: bool_indicator_b_a, model.bool_indicator_b_a1: bool_indicator_b_a1,
                     model.KEEP_PROB: 0.5}

        output_tensors = [model.task_train_op, model.meta_train_op, model.metric_train_op]
        output_tensors.extend([model.summ_op, model.global_loss, model.source_loss, model.source_accuracy, model.metric_loss])
        _, _, _, summ_writer, global_loss, source_loss, source_accuracy, metric_loss = sess.run(output_tensors, feed_dict)

        source_losses.append(source_loss)
        source_accuracies.append(source_accuracy)

        if itr % FLAGS.print_interval == 0:
            print('---'*10 + '\n%s' % exp_string)
            print('number of samples per category:', group_list)
            print('global loss: %.7f' % global_loss)
            print('metric_loss: %.7f ' % metric_loss)
            print('Iteration %d' % itr + ': Loss ' + 'training domains ' + str(np.mean(source_losses)))
            print('Iteration %d' % itr + ': Accuracy ' + 'training domains ' + str(np.mean(source_accuracies)))
            source_losses, target_losses = [], []

        if itr % FLAGS.summary_interval == 0 and FLAGS.log:
            train_writer.add_summary(summ_writer, itr)

        if (itr!=0) and itr % FLAGS.save_interval == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # Testing periodically
        class_accs = [0.0] * FLAGS.num_classes
        class_samples = [0.0] * FLAGS.num_classes
        if itr % FLAGS.test_print_interval == 0:
            test_acc, test_loss, test_count = 0.0, 0.0, 0.0
            sess.run(test_init_op) # initialize testing data generator
            for it in range(test_batches_per_epoch):
                test_input, test_label = sess.run(test_next_batch)
                feed_dict = {model.test_input: test_input, model.test_label: test_label, model.KEEP_PROB: 1.}
                output_tensors = [model.test_loss, model.test_acc]
                result = sess.run(output_tensors, feed_dict)
                test_loss += result[0]
                test_acc += result[1]
                test_count += 1
                this_class = np.argmax(test_label, axis=1)[0]
                class_accs[this_class] += result[1] # added for debug
                class_samples[this_class] += 1
            test_acc = test_acc/test_count
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                saver.save(sess, FLAGS.logdir + '/' + exp_string + '/itr' + str(itr) + '_model_acc' + str(best_test_acc))

            print('Unseen Target Validation results: Iteration %d, Loss: %f, Accuracy: %f' %(itr, test_loss/test_count, test_acc))
            print('Current best accuracy {}'.format(best_test_acc))

            with open((os.path.join(FLAGS.logdir,exp_string,'eva.txt')), 'a') as fle:
                fle.write('Unseen Target Validation results: Iteration %d, Loss: %f, Accuracy: %f \n' %(itr, test_loss/test_count, test_acc))

def main():

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    filelist_root = '/path/to/image/filelist' # path to .txt files (e.g., art_painting.txt, cartoon.txt) where images are listed line by line
    source_list = ['art_painting', 'cartoon', 'photo', 'sketch']
    source_list.remove(FLAGS.target_domain)

    exp_string = 'masf_' + FLAGS.target_domain + '.mbs_'+str(FLAGS.meta_batch_size) + \
                 '.inner' + str(FLAGS.inner_lr) + '.outer' + str(FLAGS.outer_lr) + '.clipNorm' + str(FLAGS.gradients_clip_value) + \
                 '.metric' + str(FLAGS.metric_lr) + '.margin' + str(FLAGS.margin)

    # Constructing model
    model = MASF()
    model.construct_model_train()
    model.construct_model_test()
    
    model.summ_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    
    print('Loading pretrained weights')
    model.load_initial_weights(sess)

    resume_itr = 0
    model_file = None
    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    train_file_list = [os.path.join(filelist_root, source_domain+'.txt') for source_domain in source_list]
    test_file_list = [os.path.join(filelist_root, FLAGS.target_domain+'.txt')]
    train(model, saver, sess, exp_string, train_file_list, test_file_list[0], resume_itr)

if __name__ == "__main__":
    main()
