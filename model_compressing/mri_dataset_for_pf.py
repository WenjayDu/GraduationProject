import os
import gzip
import numpy as np
import tensorflow as tf
from project_config import GlobalVar, cal_np_unique_num

from module_pocketflow.datasets.abstract_dataset import AbstractDataset

DATASET_PATH = GlobalVar.DATASET_PATH

DATA_DIR = DATASET_PATH + "/mri_pad_4_results/data"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('nb_classes',
                            cal_np_unique_num(DATA_DIR + "/validate_y.npy"),
                            '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train',
                            len(np.load(DATA_DIR + "/train_x.npy")),
                            '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val',
                            len(np.load(DATA_DIR + "/validate_x.npy")),
                            '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval',
                            len(np.load(DATA_DIR + "/test_x.npy")),
                            '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 64, 'batch size for evaluation')

# Fashion-MNIST specifications
IMAGE_HEI = 144
IMAGE_WID = 122
IMAGE_CHN = 1


def load_mnist(image_file, label_file):
    """
    Load images and labels for training

    :param image_file: path of npy file containing images for training
    :param label_file: path of npy file containing labels for training
    :return:
    """
    images = np.load(image_file)
    labels = np.load(label_file)
    return images, labels


def parse_fn(image, label, is_train):
    """Parse an (image, label) pair and apply data augmentation if needed.

    Args:
    * image: image tensor
    * label: label tensor
    * is_train: whether data augmentation should be applied

    Returns:
    * image: image tensor
    * label: one-hot label tensor
    """

    # data parsing
    label = tf.one_hot(tf.reshape(label, []), FLAGS.nb_classes)
    image = tf.cast(tf.reshape(image, [IMAGE_HEI, IMAGE_WID, IMAGE_CHN]), tf.float32)
    image = tf.image.per_image_standardization(image)

    # data augmentation
    if is_train:
        image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEI + 8, IMAGE_WID + 8)
        image = tf.random_crop(image, [IMAGE_HEI, IMAGE_WID, IMAGE_CHN])
        image = tf.image.random_flip_left_right(image)

    return image, label


class MriDataset(AbstractDataset):
    '''Fashion-MNIST dataset.'''

    def __init__(self, is_train):
        """Constructor function.

        Args:
        * is_train: whether to construct the training subset
        """

        # initialize the base class
        super(MriDataset, self).__init__(is_train)

        # setup paths to image & label files, and read in images & labels
        if is_train:
            self.batch_size = FLAGS.batch_size
            image_file = DATA_DIR + '/train_x.npy'
            label_file = DATA_DIR + '/train_y.npy'
        else:
            self.batch_size = FLAGS.batch_size_eval
            image_file = DATA_DIR + '/validate_x.npy'
            label_file = DATA_DIR + '/validate_y.npy'
        self.images, self.labels = load_mnist(image_file, label_file)
        self.parse_fn = lambda x, y: parse_fn(x, y, is_train)

    def build(self, enbl_trn_val_split=False):
        """Build iterator(s) for tf.data.Dataset() object.

        Args:
        * enbl_trn_val_split: whether to split into training & validation subsets

        Returns:
        * iterator_trn: iterator for the training subset
        * iterator_val: iterator for the validation subset
          OR
        * iterator: iterator for the chosen subset (training OR testing)
        """

        # create a tf.data.Dataset() object from NumPy arrays
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        dataset = dataset.map(self.parse_fn, num_parallel_calls=FLAGS.nb_threads)

        # create iterators for training & validation subsets separately
        if self.is_train and enbl_trn_val_split:
            iterator_val = self.__make_iterator(dataset.take(FLAGS.nb_smpls_val))
            iterator_trn = self.__make_iterator(dataset.skip(FLAGS.nb_smpls_val))
            return iterator_trn, iterator_val

        return self.__make_iterator(dataset)

    def __make_iterator(self, dataset):
        """Make an iterator from tf.data.Dataset.

        Args:
        * dataset: tf.data.Dataset object

        Returns:
        * iterator: iterator for the dataset
        """

        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.buffer_size))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(FLAGS.prefetch_size)
        iterator = dataset.make_one_shot_iterator()

        return iterator
