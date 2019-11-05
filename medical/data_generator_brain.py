import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import scipy.io as sio

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=300):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and separated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensorFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value, different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the initial file list.
            buffer_size: Number of images used as buffer for TensorFlows shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()
        self.img_paths = self.img_paths[:-1]
        self.data_size = self.data_size - 1

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        # self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices(self.img_paths)

        # patch_size = [128,128]
        self._parse_function_train = lambda filename: tf.py_func(self._extract_patch, [filename], [tf.float32, tf.float32])
        self._parse_function_inference = lambda filename: tf.py_func(self._extract_patch, [filename], [tf.float32, tf.float32])
        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=8)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.data_size = 0
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.img_paths.append(line[:-1])
                self.data_size += 1

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        for i in permutation:
            self.img_paths.append(path[i])

    def _extract_patch(self, filename):

        mat_content = sio.loadmat(filename)
        image = mat_content['image']
        image = image[34:-35,16:-17,:] # [18:-19,1:,:] is 192*192
        label = mat_content['label']
        label = label[34:-35,16:-17,0]
        mask = self._label_decomp(label)
        return image.astype(np.float32), mask.astype(np.float32)

    def _label_decomp(self, label_vol):
        """
        decompose label for softmax classifier
        original labels are batchsize * W * H * 1, with label values 0,1,2,3...
        this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
        numpy version of tf.one_hot
        """
        one_hot = []
        for i in xrange(self.num_classes):
            _vol = np.zeros(label_vol.shape)
            _vol[label_vol == i] = 1
            one_hot.append(_vol)

        return np.stack(one_hot, axis=-1)