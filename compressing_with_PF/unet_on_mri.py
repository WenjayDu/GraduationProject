import tensorflow as tf
from nets.abstract_model_helper import AbstractModelHelper
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant
from compressing_with_PF.mri_dataset import MriDataset
from compressing_with_PF.config import GlobalPath, cal_np_unique_num

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\'s ratio')
tf.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.flags.DEFINE_float('batch_size_norm', 1, 'normalization factor of batch size')
tf.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.flags.DEFINE_float('loss_w_dcy', 3e-4, 'weight decaying loss\'s coefficient')
tf.flags.DEFINE_integer('epoch_num', 10, 'num of epoch')

DATASET_PATH = GlobalPath.DATASET_PATH

INPUT_HEIGHT = 144
INPUT_WIDTH = 112
INPUT_CHANNEL = 1
CLASS_NUM = cal_np_unique_num(FLAGS.data_dir + "/validate_y.npy")


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
        print("🚩️Initializing training dataset")
        return self.dataset_train.build(enbl_trn_val_split)

    def build_dataset_eval(self):
        """Build the data subset for evaluation, usually without data augmentation."""
        print("🚩️Initializing evaluation dataset")
        return self.dataset_eval.build()

    def forward_train(self, inputs):
        """Forward computation at training."""

        return forward_fn(inputs, self.data_format)

    def forward_eval(self, inputs):
        """Forward computation at evaluation."""

        return forward_fn(inputs, self.data_format)

    def calc_loss(self, labels, outputs, trainable_vars):
        """Calculate loss (and some extra evaluation metrics)."""
        labels = tf.reshape(labels, [INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH])
        outputs = tf.reshape(outputs, [INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH, 3])
        labels = tf.cast(labels, tf.int32)
        outputs = tf.cast(outputs, tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs, name="loss")
        # loss += FLAGS.loss_w_dcy * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])
        loss = tf.reduce_mean(loss)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(input=outputs, axis=3, output_type=tf.int32), labels), tf.float32))
        metrics = {'accuracy': accuracy}

        return loss, metrics

    def setup_lrn_rate(self, global_step):
        """Setup the learning rate (and number of training iterations)."""

        nb_epochs = FLAGS.epoch_num
        idxs_epoch = [40, 80, 120]
        decay_rates = [1.0, 0.1, 0.01, 0.001]
        batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
        lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
        nb_iters = int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)

        return lrn_rate, nb_iters

    @property
    def model_name(self):
        """Model's name."""

        return 'U-Net'

    @property
    def dataset_name(self):
        """Dataset's name."""

        return 'MRI'


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

    print("❗now is at forward function")
    # construct with tf
    # batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
    # conv1
    with tf.name_scope('layer_1'):
        conv1 = tf.layers.conv2d(inputs, filters=32, kernel_size=3, strides=1, activation=tf.nn.relu, name='conv1_1',
                                 padding="same", use_bias=False)
        conv1 = tf.layers.conv2d(conv1, 32, 3, 1, activation=tf.nn.relu, name='conv1_2', padding="same", use_bias=False)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name='pool1', padding="valid")

    # conv2
    with tf.name_scope('layer_2'):
        conv2 = tf.layers.conv2d(pool1, 64, 3, 1, activation=tf.nn.relu, name='conv2_1', padding="same", use_bias=False)
        conv2 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu, name='conv2_2', padding="same", use_bias=False)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='pool2', padding="valid")
    # conv3
    with tf.name_scope('layer_3'):
        conv3 = tf.layers.conv2d(pool2, 128, 3, 1, activation=tf.nn.relu, name='conv3_1', padding="same",
                                 use_bias=False)
        conv3 = tf.layers.conv2d(conv3, 128, 3, 1, activation=tf.nn.relu, name='conv3_2', padding="same",
                                 use_bias=False)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), name='pool3', padding="valid")
    # conv4
    with tf.name_scope('layer_4'):
        conv4 = tf.layers.conv2d(pool3, 256, 3, 1, activation=tf.nn.relu, name='conv4_1', padding="same",
                                 use_bias=False)
        conv4 = tf.layers.conv2d(conv4, 256, 3, 1, activation=tf.nn.relu, name='conv4_2', padding="same",
                                 use_bias=False)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='pool4', padding="valid")
    # conv5
    with tf.name_scope('layer_5'):
        conv5 = tf.layers.conv2d(pool4, 512, 3, 1, activation=tf.nn.relu, name='conv5_1', padding="same",
                                 use_bias=False)
        conv5 = tf.layers.conv2d(conv5, 512, 3, 1, activation=tf.nn.relu, name='conv5_2', padding="same",
                                 use_bias=False)

        up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        conc5 = tf.keras.layers.Concatenate(axis=3)([up5, conv4])

    # conv6
    with tf.name_scope('layer_6'):
        conv6 = tf.layers.conv2d(conc5, 256, 3, 1, activation=tf.nn.relu, name='conv6_1', padding="same",
                                 use_bias=False)
        conv6 = tf.layers.conv2d(conv6, 256, 3, 1, activation=tf.nn.relu, name='conv6_2', padding="same",
                                 use_bias=False)

        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        conc6 = tf.keras.layers.Concatenate(axis=3)([up6, conv3])

    # conv7
    with tf.name_scope('layer_7'):
        conv7 = tf.layers.conv2d(conc6, 128, 3, 1, activation=tf.nn.relu, name='conv7_1', padding="same",
                                 use_bias=False)
        conv7 = tf.layers.conv2d(conv7, 128, 3, 1, activation=tf.nn.relu, name='conv7_2', padding="same",
                                 use_bias=False)

        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
        conc7 = tf.keras.layers.Concatenate(axis=3)([up7, conv2])

    # conv8
    with tf.name_scope('layer_8'):
        conv8 = tf.layers.conv2d(conc7, 64, 3, 1, activation=tf.nn.relu, name='conv8_1', padding="same", use_bias=False)
        conv8 = tf.layers.conv2d(conv8, 64, 3, 1, activation=tf.nn.relu, name='conv8_2', padding="same", use_bias=False)

        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
        conc8 = tf.keras.layers.Concatenate(axis=3)([up8, conv1])

    # conv9
    with tf.name_scope('layer_9'):
        conv9 = tf.layers.conv2d(conc8, 32, 3, 1, activation=tf.nn.relu, name='conv9_1', padding="same", use_bias=False)
        conv9 = tf.layers.conv2d(conv9, 32, 3, 1, activation=tf.nn.relu, name='conv9_2', padding="same", use_bias=False)
        conv9 = tf.layers.conv2d(conv9, 3, 3, 1, activation=tf.nn.relu, name='conv9_3', padding="same", use_bias=False)
    # with tf.name_scope('softmax_loss'):
    #     conv10 = tf.keras.layers.Convolution2D(CLASS_NUM, 1, 1, activation='softmax', name="loss")(conv9)

    return conv9

    # # construct with tf.keras
    # IN = tf.keras.layers.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL))
    # BN1 = tf.keras.layers.BatchNormalization(IN)
    #
    # conv1 = tf.keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(BN1)
    # conv1 = tf.keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    # pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    # conv2 = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    # conv2 = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    # pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    # conv3 = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    # conv3 = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    # pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    # conv4 = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    # conv4 = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    # pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    # conv5 = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    # conv5 = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    # up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    # # up6 = tf.keras.layers.Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    # conc5 = tf.keras.layers.Concatenate(axis=3)([up5, conv4])
    # conv6 = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conc5)
    # conv6 = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    #
    # up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    # # up6 = tf.keras.layers.Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    # conc6 = tf.keras.layers.Concatenate(axis=3)([up6, conv3])
    # conv7 = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conc6)
    # conv7 = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
    #
    # up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    # # up7 = tf.keras.layers.Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv7)
    # conc7 = tf.keras.layers.Concatenate(axis=3)([up7, conv2])
    # conv8 = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conc7)  # (up8)
    # conv8 = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    #
    # up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    # # up8 = tf.keras.layers.Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv8)
    # conc8 = tf.keras.layers.Concatenate(axis=3)([up8, conv1])
    # conv9 = tf.keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conc8)
    # conv9 = tf.keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    #
    # conv10 = tf.keras.layers.Convolution2D(CLASS_NUM, 1, 1, activation='softmax')(conv9)
    # print("❗️out of forward function")
    # return conv10
