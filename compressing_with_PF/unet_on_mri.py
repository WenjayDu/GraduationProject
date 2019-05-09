import tensorflow as tf
from nets.abstract_model_helper import AbstractModelHelper
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant
from compressing_with_PF.mri_dataset import MriDataset
from compressing_with_PF.config import GlobalPath, cal_np_unique_num

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\'s ratio')
tf.app.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 1, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 3e-4, 'weight decaying loss\'s coefficient')

DATASET_PATH = GlobalPath.DATASET_PATH
DATA_DIR = DATASET_PATH + "/mri_pad_4_results/data"

INPUT_HEIGHT = 144
INPUT_WIDTH = 112
INPUT_CHANNEL = 1
CLASS_NUM = cal_np_unique_num(DATA_DIR + "/validate_y.npy")


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
        print("labels.shape", labels.shape, "outputs.shape", outputs.shape)
        labels = tf.reshape(labels, [INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH])
        outputs = tf.reshape(outputs, [INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH, 3])
        labels = tf.cast(labels, tf.int32)
        outputs = tf.cast(outputs, tf.float32)
        print("labels.shape", labels.shape, "outputs.shape", outputs.shape)
        print("labels.dtype", labels.dtype, "outputs.dtype", outputs.dtype)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs, name="loss")
        # loss += FLAGS.loss_w_dcy * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])
        loss = tf.reduce_mean(loss)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(input=outputs, axis=3, output_type=tf.int32), labels), tf.float32))
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
    print("❗️inputs shape", inputs.shape)
    conv1 = tf.layers.conv2d(inputs, filters=32, kernel_size=3, strides=1, activation=tf.nn.relu, name='conv1_1',
                             padding="same")
    print("❗️conv1 shape", conv1.shape)
    conv1 = tf.layers.conv2d(conv1, 32, 3, 1, activation=tf.nn.relu, name='conv1_2', padding="same")
    print("❗️conv1 shape", conv1.shape)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name='pool1', padding="valid")
    print("❗pool1 shape", pool1.shape)
    # conv2
    conv2 = tf.layers.conv2d(pool1, 64, 3, 1, activation=tf.nn.relu, name='conv2_1', padding="same")
    print("❗️conv2 shape", conv2.shape)
    conv2 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu, name='conv2_2', padding="same")
    print("❗️conv2 shape", conv2.shape)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='pool2', padding="valid")
    print("❗pool2 shape", pool2.shape)
    # conv3
    conv3 = tf.layers.conv2d(pool2, 128, 3, 1, activation=tf.nn.relu, name='conv3_1', padding="same")
    print("❗️conv3 shape", conv3.shape)
    conv3 = tf.layers.conv2d(conv3, 128, 3, 1, activation=tf.nn.relu, name='conv3_2', padding="same")
    print("❗️conv3 shape", conv3.shape)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), name='pool3', padding="valid")
    print("❗pool3 shape", pool3.shape)
    # conv4
    conv4 = tf.layers.conv2d(pool3, 256, 3, 1, activation=tf.nn.relu, name='conv4_1', padding="same")
    print("❗️conv4 shape", conv4.shape)
    conv4 = tf.layers.conv2d(conv4, 256, 3, 1, activation=tf.nn.relu, name='conv4_2', padding="same")
    print("❗️conv4 shape", conv4.shape)
    pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='pool4', padding="valid")
    print("❗pool4 shape", pool4.shape)
    # conv5
    conv5 = tf.layers.conv2d(pool4, 512, 3, 1, activation=tf.nn.relu, name='conv5_1', padding="same")
    print("❗️conv5 shape", conv5.shape)
    conv5 = tf.layers.conv2d(conv5, 512, 3, 1, activation=tf.nn.relu, name='conv5_2', padding="same")
    print("❗️conv5 shape", conv5.shape)

    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    print("❗up5 shape", up5.shape)
    conc5 = tf.keras.layers.Concatenate(axis=3)([up5, conv4])
    print("❗conc5 shape", conc5.shape)

    # conv6
    conv6 = tf.layers.conv2d(conc5, 256, 3, 1, activation=tf.nn.relu, name='conv6_1', padding="same")
    print("❗️conv6 shape", conv6.shape)
    conv6 = tf.layers.conv2d(conv6, 256, 3, 1, activation=tf.nn.relu, name='conv6_2', padding="same")
    print("❗️conv6 shape", conv6.shape)

    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    print("❗up6 shape", up6.shape)
    conc6 = tf.keras.layers.Concatenate(axis=3)([up6, conv3])
    print("❗conc6 shape", conc6.shape)

    # conv7
    conv7 = tf.layers.conv2d(conc6, 128, 3, 1, activation=tf.nn.relu, name='conv7_1', padding="same")
    print("❗️conv7 shape", conv7.shape)
    conv7 = tf.layers.conv2d(conv7, 128, 3, 1, activation=tf.nn.relu, name='conv7_2', padding="same")
    print("❗️conv7 shape", conv7.shape)

    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    print("❗up7 shape", up7.shape)
    conc7 = tf.keras.layers.Concatenate(axis=3)([up7, conv2])
    print("❗️conc7 shape", conc7.shape)

    # conv8
    conv8 = tf.layers.conv2d(conc7, 128, 3, 1, activation=tf.nn.relu, name='conv8_1', padding="same")
    print("❗️conv8 shape", conv8.shape)
    conv8 = tf.layers.conv2d(conv8, 128, 3, 1, activation=tf.nn.relu, name='conv8_2', padding="same")
    print("❗️conv8 shape", conv8.shape)

    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    print("❗up8 shape", up8.shape)
    conc8 = tf.keras.layers.Concatenate(axis=3)([up8, conv1])
    print("❗️conc8 shape", conc8.shape)

    # conv9
    conv9 = tf.layers.conv2d(conc8, 128, 3, 1, activation=tf.nn.relu, name='conv9_1', padding="same")
    print("❗️conv9 shape", conv9.shape)
    conv9 = tf.layers.conv2d(conv9, 128, 3, 1, activation=tf.nn.relu, name='conv9_2', padding="same")
    print("❗️conv9 shape", conv9.shape)

    conv10 = tf.keras.layers.Convolution2D(CLASS_NUM, 1, 1, activation='softmax', name="conv10")(conv9)
    print("❗️conv10 shape", conv10.shape)

    print("❗️out of forward function")
    return conv10

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
