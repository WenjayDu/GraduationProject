import numpy as np
import tensorflow as tf

from compressing_with_PF.config import GlobalPath, cal_np_unique_num
from datasets.abstract_dataset import AbstractDataset

DATASET_PATH = GlobalPath.DATASET_PATH

DEFAULT_DATA_DIR = DATASET_PATH + "/mri_pad_4/data"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_dir', DEFAULT_DATA_DIR, 'dir of data used to train')
tf.flags.DEFINE_string('input_shape', "[144, 112, 1]", 'shape of input data')
tf.flags.DEFINE_string('structure', 'improved', 'structure of the unet to use, like original/improved')
tf.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.flags.DEFINE_integer('epoch_num', 10, 'num of epoch')
tf.flags.DEFINE_integer('divisor', 1, 'divisor of the number of filters, must be a factor of 64')
tf.flags.DEFINE_string('exec_mode', 'train', 'execution mode: train / eval')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size per GPU for training')
tf.flags.DEFINE_integer('batch_size_eval', 100, 'batch size for evaluation')
tf.flags.DEFINE_boolean('data_augmentation', True, 'whether to use data augmentation')
tf.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.flags.DEFINE_integer('nb_classes', cal_np_unique_num(FLAGS.data_dir + "/validate_y.npy"), '# of classes')
tf.flags.DEFINE_integer('nb_smpls_train', len(np.load(FLAGS.data_dir + "/train_x.npy")),
                        '# of samples for training')
tf.flags.DEFINE_integer('nb_smpls_val', len(np.load(FLAGS.data_dir + "/validate_x.npy")),
                        '# of samples for validation')
tf.flags.DEFINE_integer('nb_smpls_eval', len(np.load(FLAGS.data_dir + "/test_x.npy")),
                        '# of samples for evaluation')

INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL = eval(FLAGS.input_shape)


class BrainImgDataset(AbstractDataset):
    """
    MRI dataset
    """

    def __init__(self, is_train):
        """
        Constructor function.
        :param is_train: whether to construct the training subset
        """

        # initialize the base class
        super(BrainImgDataset, self).__init__(is_train)

        # setup paths to image & label files, and read in images & labels
        if is_train:
            self.batch_size = FLAGS.batch_size
            image_file = FLAGS.data_dir + '/train_x.npy'
            label_file = FLAGS.data_dir + '/train_y.npy'
        else:
            self.batch_size = FLAGS.batch_size_eval
            image_file = FLAGS.data_dir + '/validate_x.npy'
            label_file = FLAGS.data_dir + '/validate_y.npy'
        self.images, self.labels = load_npy(image_file, label_file)
        self.parse_fn = lambda x, y: parse_fn(x, y, is_train)

    def build(self, enbl_trn_val_split=False):
        """
        Build iterator(s) for tf.data.Dataset() object.

        :param enbl_trn_val_split: whether to split into training & validation subsets
        :return: iterator_trn: iterator for the training subset
                 iterator_val: iterator for the validation subset
                 OR
                 iterator: iterator for the chosen subset (training OR testing)
        """

        # create a tf.data.Dataset() object from NumPy arrays
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        dataset = dataset.map(self.parse_fn, num_parallel_calls=FLAGS.nb_threads)
        # create iterators for training & validation subsets separately
        if self.is_train and enbl_trn_val_split:
            iterator_val = self.__make_iterator(dataset.take(FLAGS.nb_smpls_val))
            iterator_trn = self.__make_iterator(dataset.skip(FLAGS.nb_smpls_train))
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


def load_npy(image_file, label_file):
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
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    # data augmentation
    # if FLAGS.data_augmentation and is_train:
    #     image = tf.image.resize_image_with_crop_or_pad(image, INPUT_HEIGHT + 8, INPUT_WIDTH + 8)
    #     image = tf.random_crop(image, [INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])
    #     image = tf.image.random_flip_left_right(image)

    return image, label
