import tensorflow as tf

from compressing_with_PF.config import GlobalPath, cal_np_unique_num
from compressing_with_PF.mri_dataset import MriDataset
from compressing_with_PF.various_unets import (original_with_BN, original, smaller_with_BN, smaller)
from nets.abstract_model_helper import AbstractModelHelper
from utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\'s ratio')
tf.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.flags.DEFINE_float('batch_size_norm', 1, 'normalization factor of batch size')
tf.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.flags.DEFINE_float('loss_w_dcy', 3e-4, 'weight decaying loss\'s coefficient')

INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL = eval(FLAGS.input_shape)
CLASS_NUM = cal_np_unique_num(FLAGS.data_dir + "/validate_y.npy")
DATASET_PATH = GlobalPath.DATASET_PATH


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

    unet = choose_unet(FLAGS.structure)
    return unet.forward_rn(inputs)


def choose_unet(structure_name):
    switcher = {
        "original": original,
        "original_with_BN": original_with_BN,
        "smaller": smaller,
        "smaller_with_BN": smaller_with_BN
    }
    return switcher.get(structure_name)
