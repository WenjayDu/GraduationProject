import tensorflow as tf
from model_compressing.mri_dataset_for_pf import MriDataset
from config import GlobalVar, cal_np_unique_num
from module_pocketflow.nets.abstract_model_helper import AbstractModelHelper
from module_pocketflow.utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from module_pocketflow.utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\'s ratio')
tf.app.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 3e-4, 'weight decaying loss\'s coefficient')

DATASET_PATH = GlobalVar.DATASET_PATH
DATA_DIR = DATASET_PATH + "/mri_pad_4_results/data"

INPUT_HEIGHT = 144
INPUT_WIDTH = 122
INPUT_CHANNEL = 1
CLASS_NUM = cal_np_unique_num(DATA_DIR + "/validate_y.npy")


def forward_fn(inputs, data_format):
    """Forward pass function.

    Args:
    * inputs: inputs to the network's forward pass
    * data_format: data format ('channels_last' OR 'channels_first')

    Returns:
    * inputs: outputs from the network's forward pass
    """

    # tranpose the image tensor if needed
    if data_format == 'channel_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # construct with tf, but temporarily stopped
    # batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
    # # conv1
    # conv1 = tf.layers.conv2d(input, 32, 3, 3, activation=tf.nn.relu, name='conv1_1')
    # conv1 = tf.layers.conv2d(conv1, 32, 3, 3, activation=tf.nn.relu, name='conv1_2')
    # pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=2, name='pool1')
    # # conv2
    # conv2 = tf.layers.conv2d(pool1, 64, 3, 3, activation=tf.nn.relu, name='conv2_1')
    # conv2 = tf.layers.conv2d(conv2, 64, 3, 3, activation=tf.nn.relu, name='conv3_2')
    # pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2, name='pool2')
    # # conv3
    # conv3 = tf.layers.conv2d(pool2, 128, 3, 3, activation=tf.nn.relu, name='conv3_1')
    # conv3 = tf.layers.conv2d(conv3, 128, 3, 3, activation=tf.nn.relu, name='conv3_2')
    # pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=2, name='pool3')
    # # conv4
    # conv4 = tf.layers.conv2d(pool3, 256, 3, 3, activation=tf.nn.relu, name='conv4_1')
    # conv4 = tf.layers.conv2d(conv4, 256, [3, 3], activation=tf.nn.relu, name='conv4_2')
    # pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=2, name='pool4')
    # # conv5
    # conv5 = tf.layers.conv2d(pool4, 512, 3, 3, activation=tf.nn.relu, name='conv5_1')
    # conv5 = tf.layers.conv2d(conv5, 512, [3, 3], activation=tf.nn.relu, name='conv5_2')
    #
    # up5 = tf.nn.conv2d_transpose(
    #     value=conv5, filter=[2, 2, ],
    #     output_shape=[batch_size, int(relu_2_result.shape[1]) * 2, int(relu_2_result.shape[2]) * 2, 512],
    #     strides=[1, 2, 2, 1], padding='VALID', name='Up_Sampling')
    #
    # conc5 = tf.concat(values=[conv5, up5])

    # construct with tf.keras
    IN = tf.keras.layers.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL))(inputs)
    BN1 = tf.keras.layers.BatchNormalization(IN)

    conv1 = tf.keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(BN1)
    conv1 = tf.keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    # up6 = tf.keras.layers.Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc5 = tf.keras.layers.Concatenate(axis=3)([up5, conv4])
    conv6 = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conc5)
    conv6 = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    # up6 = tf.keras.layers.Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc6 = tf.keras.layers.Concatenate(axis=3)([up6, conv3])
    conv7 = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conc6)
    conv7 = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    # up7 = tf.keras.layers.Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv7)
    conc7 = tf.keras.layers.Concatenate(axis=3)([up7, conv2])
    conv8 = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conc7)  # (up8)
    conv8 = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    # up8 = tf.keras.layers.Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv8)
    conc8 = tf.keras.layers.Concatenate(axis=3)([up8, conv1])
    conv9 = tf.keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conc8)
    conv9 = tf.keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = tf.keras.layers.Convolution2D(CLASS_NUM, 1, 1, activation='softmax')(conv9)

    return conv10


class ModelHelper(AbstractModelHelper):
    """Model helper for creating a U-Net model for the MRI dataset."""

    def __init__(self, data_format='channels_last'):
        """Constructor function."""

        # class-independent initialization
        super(ModelHelper, self).__init__(data_format)

        # initialize training & evaluation subsets
        self.dataset_train = MriDataset(is_train=True)
        self.dataset_eval = MriDataset(is_train=False)

    def build_dataset_train(self, enbl_trn_val_split=False):
        """Build the data subset for training, usually with data augmentation."""

        return self.dataset_train.build(enbl_trn_val_split)

    def build_dataset_eval(self):
        """Build the data subset for evaluation, usually without data augmentation."""

        return self.dataset_eval.build()

    def forward_train(self, inputs):
        """Forward computation at training."""

        return forward_fn(inputs, self.data_format)

    def forward_eval(self, inputs):
        """Forward computation at evaluation."""

        return forward_fn(inputs, self.data_format)

    def calc_loss(self, labels, outputs, trainable_vars):
        """Calculate loss (and some extra evaluation metrics)."""

        loss = tf.losses.softmax_cross_entropy(labels, outputs)
        loss += FLAGS.loss_w_dcy * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(outputs, axis=1)), tf.float32))
        metrics = {'accuracy': accuracy}

        return loss, metrics

    def setup_lrn_rate(self, global_step):
        """Setup the learning rate (and number of training iterations)."""

        nb_epochs = 160
        idxs_epoch = [40, 80, 120]
        decay_rates = [1.0, 0.1, 0.01, 0.001]
        batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
        lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
        nb_iters = int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)

        return lrn_rate, nb_iters

    @property
    def model_name(self):
        """Model's name."""

        return 'convnet'

    @property
    def dataset_name(self):
        """Dataset's name."""

        return 'MRI'
