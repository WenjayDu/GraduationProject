import tensorflow as tf
import project_config as config
from module_minc_keras.minc_keras import *
from data_processing import convert_npy_to_tfrecords

PROJECT_DIR = config.get_project_path()
DATASET_DIR = PROJECT_DIR + "/datasets"
OUTPUT_DIR = PROJECT_DIR + "/output/tf_implementation"

TRAIN_SET_NAME = "/tfrecords/train.tfrecords"
VALIDATE_SET_NAME = "/tfrecords/validate.tfrecords"
TEST_SET_NAME = "/tfrecords/test.tfrecords"

LOGS_DIR = OUTPUT_DIR + "/logs"
SAVED_MODELS = OUTPUT_DIR + "/saved_models"

TRAIN_BATCH_SIZE = 1
VALIDATION_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1

TEST_SET_SIZE = 30

# the path of dir stroing imgs for predicting operation
ORIGIN_PREDICT_DIRECTORY = DATASET_DIR + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_anat_rsl"

# the path of dir for saving imgs output from predicting operation
PREDICT_SAVED_DIRECTORY = OUTPUT_DIR + "/prediction_saved"

INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, INPUT_IMG_CHANNEL = 144, 112, 1
INPUT_SHAPE = (INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, INPUT_IMG_CHANNEL)

OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH, OUTPUT_IMG_CHANNEL = 144, 112, 3
OUTPUT_SHAPE = (OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH, OUTPUT_IMG_CHANNEL)

EPOCH_NUM = 3

# EPS below is the value added to denominator in BN operation,
# to prevent the 0 operation when dividing by the variance,
# may be different in different frameworks.
EPS = 10e-5


def read_image(file_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    img = tf.decode_raw(features['data'], np.float16)
    print("img type", type(img), "img shape", img.get_shape())
    print("img reshaping")
    img = tf.reshape(img, [FLAGS.input_shape[0], FLAGS.input_shape[1], FLAGS.input_shape[2]])
    label = tf.decode_raw(features['label'], np.float16)
    print("label type", type(label), "label shape", label.get_shape())
    print("label reshaping")
    label = tf.reshape(label, [FLAGS.input_shape[0], FLAGS.input_shape[1], FLAGS.input_shape[2]])
    return img, label


def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue)
    min_after_dequeue = 500
    capacity = 510
    # image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
    image_batch, label_batch = tf.train.shuffle_batch(
        tensors=[img, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    # one_hot_labels = tf.reshape(label_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH])
    print("image_batch shape", image_batch.get_shape(), "label_batch shape", label_batch.get_shape())
    return image_batch, label_batch


class UNet:
    def __init__(self):
        print('Constructing New U-Net Network...')
        self.input_image = None
        self.input_label = None
        self.cast_image = None
        self.cast_label = None
        self.keep_prob = None
        self.lamb = None
        self.result_expand = None
        self.is_training = None
        self.loss, self.loss_mean, self.loss_all, self.train_step = [None] * 4
        self.prediction, self.correct_prediction, self.accuracy = [None] * 3
        self.result_conv = {}
        self.result_relu = {}
        self.result_maxpool = {}
        self.result_from_contracting = {}
        self.w = {}  # weight
        # self.b = {} # bias, because of BN, no need for bias

    def init_w(self, shape, name):
        with tf.name_scope('init_w'):
            # stddev = 0.01
            stddev = tf.sqrt(x=2 / (shape[0] * shape[1] * shape[2]))
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32), name=name)
            tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(w))
            return w

    @staticmethod
    def init_b(shape, name):
        with tf.name_scope('init_b'):
            return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)

    @staticmethod
    def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
        from tensorflow.python.training.moving_averages import assign_moving_average

        with tf.variable_scope(name):
            params_shape = x.shape[-1:]
            moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer,
                                         trainable=False)

            def mean_var_with_update():
                mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
                with tf.control_dependencies([
                    assign_moving_average(moving_mean, mean_this_batch, decay),
                    assign_moving_average(moving_var, variance_this_batch, decay)
                ]):
                    return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
            if affine:  # If you want to scale with beta and gamma
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
                                                   variance_epsilon=eps)
            else:
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                                   variance_epsilon=eps)
            return normed

    @staticmethod
    def merge_results_from_contracting_and_upsampling(result_from_contracting, result_from_upsampling):
        result_from_contracting_crop = result_from_contracting
        return tf.concat(values=[result_from_contracting_crop, result_from_upsampling], axis=-1)

    @staticmethod
    def conv(w, bias, ):
        return

    def build_up_unet(self, batch_size):
        # the contracting path
        # input layer
        with tf.name_scope('input'):
            # learning_rate = tf.train.exponential_decay()
            self.input_image = tf.placeholder(
                dtype=tf.float32,
                shape=[batch_size, FLAGS.input_shape[0], FLAGS.input_shape[1], FLAGS.input_shape[2]],
                name='input_images'
            )
            self.input_label = tf.placeholder(
                dtype=tf.int32, shape=[batch_size, FLAGS.input_shape[0], FLAGS.input_shape[1]],
                name='input_labels'
            )

            # keep_prob is to define how many neurons to keep when dropping out
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

            # lamb is used to control the size of the Regularization term,
            # and a larger value of lamb will constrain the complexity of the model to a greater extent
            # The purpose is to avoid model overfitting
            self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')

            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            normed_batch = self.batch_norm(x=self.input_image, is_training=self.is_training, name='input')

            print("input layer output shape", normed_batch.shape)

        # layer 1
        with tf.name_scope('layer_1'):
            # conv_1
            self.w[1] = self.init_w(shape=[3, 3, FLAGS.input_shape[2], 64], name='w_1')
            # self.b[1] = self.init_b(shape=[64], name='b_1')
            conv_1_result = tf.nn.conv2d(
                input=normed_batch, filter=self.w[1], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_1_conv_1')
            relu_1_result = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[2] = self.init_w(shape=[3, 3, 64, 64], name='w_2')
            # self.b[2] = self.init_b(shape=[64], name='b_2')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[2], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_1_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')
            self.result_from_contracting[1] = relu_2_result  # saved for up sampling below

            # maxpool
            maxpool_result = tf.nn.max_pool(
                value=relu_2_result, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            dropout_result = tf.nn.dropout(x=maxpool_result, keep_prob=self.keep_prob)

            print("layer 1 output shape", dropout_result.shape)

        # layer 2
        with tf.name_scope('layer_2'):
            # conv_1
            self.w[3] = self.init_w(shape=[3, 3, 64, 128], name='w_3')
            # self.b[3] = self.init_b(shape=[128], name='b_3')
            conv_1_result = tf.nn.conv2d(
                input=dropout_result, filter=self.w[3], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_2_conv_1')
            relu_1_result = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[4] = self.init_w(shape=[3, 3, 128, 128], name='w_4')
            # self.b[4] = self.init_b(shape=[128], name='b_4')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[4], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_2_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')
            self.result_from_contracting[2] = relu_2_result  # saved for up sampling below

            # maxpooling
            maxpool_result = tf.nn.max_pool(
                value=relu_2_result, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            dropout_result = tf.nn.dropout(x=maxpool_result, keep_prob=self.keep_prob)
            print("layer 2 output shape", dropout_result.shape)

        # layer 3
        with tf.name_scope('layer_3'):
            # conv_1
            self.w[5] = self.init_w(shape=[3, 3, 128, 256], name='w_5')
            # self.b[5] = self.init_b(shape=[256], name='b_5')
            conv_1_result = tf.nn.conv2d(
                input=dropout_result, filter=self.w[5], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_3_conv_1')
            relu_1_result = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[6] = self.init_w(shape=[3, 3, 256, 256], name='w_6')
            # self.b[6] = self.init_b(shape=[256], name='b_6')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[6], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_3_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')
            self.result_from_contracting[3] = relu_2_result  # saved for up sampling below

            # maxpool
            maxpool_result = tf.nn.max_pool(
                value=relu_2_result, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            dropout_result = tf.nn.dropout(x=maxpool_result, keep_prob=self.keep_prob)

            print("layer 3 output shape", dropout_result.shape)

        # layer 4
        with tf.name_scope('layer_4'):
            # conv_1
            self.w[7] = self.init_w(shape=[3, 3, 256, 512], name='w_7')
            # self.b[7] = self.init_b(shape=[512], name='b_7')
            conv_1_result = tf.nn.conv2d(
                input=dropout_result, filter=self.w[7], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_4_conv_1')
            relu_1_result = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[8] = self.init_w(shape=[3, 3, 512, 512], name='w_8')
            # self.b[8] = self.init_b(shape=[512], name='b_8')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[8], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_4_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')
            self.result_from_contracting[4] = relu_2_result  # saved for up sampling below

            # maxpool
            maxpool_result = tf.nn.max_pool(
                value=relu_2_result, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            dropout_result = tf.nn.dropout(x=maxpool_result, keep_prob=self.keep_prob)

            print("layer 4 output shape", dropout_result.shape)

        # the bottom
        # layer 5
        with tf.name_scope('layer_5'):
            # conv_1
            self.w[9] = self.init_w(shape=[3, 3, 512, 1024], name='w_9')
            # self.b[9] = self.init_b(shape=[1024], name='b_9')
            conv_1_result = tf.nn.conv2d(
                input=dropout_result, filter=self.w[9], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_5_conv_1')
            relu_1_result = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[10] = self.init_w(shape=[3, 3, 1024, 1024], name='w_10')
            # self.b[10] = self.init_b(shape=[1024], name='b_10')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[10], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_5_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')

            # up_sampling
            self.w[11] = self.init_w(shape=[2, 2, 512, 1024], name='w_11')
            # self.b[11] = self.init_b(shape=[512], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=relu_2_result, filter=self.w[11],
                output_shape=[batch_size, int(relu_2_result.shape[1]) * 2, int(relu_2_result.shape[2]) * 2, 512],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sampling')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_5_conv_up')
            relu_3_result = tf.nn.relu(features=normed_batch, name='relu_3')

            # dropout
            dropout_result = tf.nn.dropout(x=relu_3_result, keep_prob=self.keep_prob)

            print("layer 5 output shape", dropout_result.shape)

        # the expanding path
        # layer 6
        with tf.name_scope('layer_6'):
            # copy, crop and merge
            result_merge = self.merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[4], result_from_upsampling=dropout_result)

            # conv_1
            self.w[12] = self.init_w(shape=[3, 3, 1024, 512], name='w_12')
            # self.b[12] = self.init_b(shape=[512], name='b_12')
            conv_1_result = tf.nn.conv2d(
                input=result_merge, filter=self.w[12], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_6_conv_1')
            relu_1_result = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[13] = self.init_w(shape=[3, 3, 512, 512], name='w_10')
            # self.b[13] = self.init_b(shape=[512], name='b_10')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[13], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_6_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')
            # print(result_relu_2.shape[1])

            # up_sampling
            self.w[14] = self.init_w(shape=[2, 2, 256, 512], name='w_11')
            # self.b[14] = self.init_b(shape=[256], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=relu_2_result, filter=self.w[14],
                output_shape=[batch_size, int(relu_2_result.shape[1]) * 2, int(relu_2_result.shape[2]) * 2, 256],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sampling')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_6_conv_up')
            relu_3_result = tf.nn.relu(features=normed_batch, name='relu_3')

            # dropout
            dropout_result = tf.nn.dropout(x=relu_3_result, keep_prob=self.keep_prob)

            print("layer 6 output shape", dropout_result.shape)

        # layer 7
        with tf.name_scope('layer_7'):
            # copy, crop and merge
            result_merge = self.merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[3], result_from_upsampling=dropout_result)

            # conv_1
            self.w[15] = self.init_w(shape=[3, 3, 512, 256], name='w_12')
            # self.b[15] = self.init_b(shape=[256], name='b_12')
            conv_1_result = tf.nn.conv2d(
                input=result_merge, filter=self.w[15], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_7_conv_1')
            relu_1_result = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[16] = self.init_w(shape=[3, 3, 256, 256], name='w_10')
            # self.b[16] = self.init_b(shape=[256], name='b_10')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[16], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_7_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')

            # up_sampling
            self.w[17] = self.init_w(shape=[2, 2, 128, 256], name='w_11')
            # self.b[17] = self.init_b(shape=[128], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=relu_2_result, filter=self.w[17],
                output_shape=[batch_size, int(relu_2_result.shape[1]) * 2, int(relu_2_result.shape[2]) * 2, 128],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sampling')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_7_up')
            relu_3_result = tf.nn.relu(features=normed_batch, name='relu_3')

            # dropout
            dropout_result = tf.nn.dropout(x=relu_3_result, keep_prob=self.keep_prob)

            print("layer 7 output shape", dropout_result.shape)

        # layer 8
        with tf.name_scope('layer_8'):
            # copy, crop and merge
            result_merge = self.merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[2], result_from_upsampling=dropout_result)

            # conv_1
            self.w[18] = self.init_w(shape=[3, 3, 256, 128], name='w_12')
            # self.b[18] = self.init_b(shape=[128], name='b_12')
            conv_1_result = tf.nn.conv2d(
                input=result_merge, filter=self.w[18], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_8_conv_1')
            relu_1_result = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[19] = self.init_w(shape=[3, 3, 128, 128], name='w_10')
            # self.b[19] = self.init_b(shape=[128], name='b_10')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[19], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_8_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')

            # up_sampling
            self.w[20] = self.init_w(shape=[2, 2, 64, 128], name='w_11')
            # self.b[20] = self.init_b(shape=[64], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=relu_2_result, filter=self.w[20],
                output_shape=[batch_size, int(relu_2_result.shape[1]) * 2, int(relu_2_result.shape[2]) * 2, 64],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sampling')
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_8_up')
            relu_3_result = tf.nn.relu(features=normed_batch, name='relu_3')

            # dropout
            dropout_result = tf.nn.dropout(x=relu_3_result, keep_prob=self.keep_prob)

            print("layer 8 output shape", dropout_result.shape)

        # layer 9
        with tf.name_scope('layer_9'):
            # copy, crop and merge
            result_merge = self.merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[1], result_from_upsampling=dropout_result)

            # conv_1
            self.w[21] = self.init_w(shape=[3, 3, 128, 64], name='w_12')
            # self.b[21] = self.init_b(shape=[64], name='b_12')
            conv_1_result = tf.nn.conv2d(
                input=result_merge, filter=self.w[21], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=conv_1_result, is_training=self.is_training, name='layer_9_conv_1')
            relu_1_result = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[22] = self.init_w(shape=[3, 3, 64, 64], name='w_10')
            # self.b[22] = self.init_b(shape=[64], name='b_10')
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[22], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch = self.batch_norm(x=conv_2_result, is_training=self.is_training, name='layer_9_conv_2')
            relu_2_result = tf.nn.relu(features=normed_batch, name='relu_2')

            # convolution to [batch_size, OUTPIT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM]
            self.w[23] = self.init_w(shape=[1, 1, 64, 3], name='w_11')
            # self.b[23] = self.init_b(shape=[CLASS_NUM], name='b_11')
            result_conv_3 = tf.nn.conv2d(
                input=relu_2_result, filter=self.w[23],
                strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
            normed_batch = self.batch_norm(x=result_conv_3, is_training=self.is_training, name='layer_9_conv_3')
            # self.prediction = tf.nn.relu(tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='relu_3')
            # self.prediction = tf.nn.sigmoid(x=tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='sigmoid_1')
            self.prediction = normed_batch

            print("layer 9 output shape", self.prediction.shape)

        # softmax loss
        with tf.name_scope('softmax_loss'):
            # using one-hot
            # self.loss = \
            # 	tf.nn.softmax_cross_entropy_with_logits(labels=self.cast_label, logits=self.prediction, name='loss')

            # not using one-hot
            self.loss = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction,
                                                               name='loss')
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        # accuracy
        with tf.name_scope('accuracy'):
            # using one-hot
            # self.correct_prediction = tf.equal(tf.argmax(self.prediction, axis=3), tf.argmax(self.cast_label, axis=3))

            # not using one-hot
            self.correct_prediction = \
                tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)

        # Gradient Descent
        with tf.name_scope('Gradient_Descent'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)

    def train(self):
        train_file_path = FLAGS.dataset_dir + TRAIN_SET_NAME
        train_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=FLAGS.epoch_num, shuffle=True)
        ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        train_images, train_labels = read_image_batch(train_image_filename_queue, TRAIN_BATCH_SIZE)
        tf.summary.scalar("loss", self.loss_mean)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                epoch = 1
                while not coord.should_stop():
                    # run training
                    example, label = sess.run([train_images, train_labels])  # get image and label，type is numpy.ndarry
                    label = label.reshape(1, FLAGS.input_shape[0], FLAGS.input_shape[1])

                    # example = example.reshape(144, 112, 1)
                    # from keras_preprocessing import image
                    # img = image.array_to_img(example)
                    # print("example shape",example.shape)
                    # img.show()
                    # image.save_img(PROJECT_DIR + "/backup/" + str(epoch) + ".png", img)

                    lo, acc, summary_str = sess.run(
                        [self.loss_mean, self.accuracy, merged_summary],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_training: True}
                    )
                    summary_writer.add_summary(summary_str, epoch)
                    if epoch % 10 == 0:
                        print('num %d , loss: %.6f , accuracy: %.6f' % (epoch, lo, acc))
                    sess.run(
                        [self.train_step],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_training: True}
                    )
                    epoch += 1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                all_parameters_saver.save(sess=sess, save_path=ckpt_path)
                coord.request_stop()
            coord.join(threads)
        print("Done training")

    def validate(self):
        validation_file_path = FLAGS.dataset_dir + VALIDATE_SET_NAME
        validation_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(validation_file_path), num_epochs=1, shuffle=True)
        ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        validation_images, validation_labels = read_image_batch(validation_image_filename_queue, VALIDATION_BATCH_SIZE)
        # tf.summary.scalar("loss", self.loss_mean)
        # tf.summary.scalar('accuracy', self.accuracy)
        # merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                epoch = 1
                while not coord.should_stop():
                    example, label = sess.run([validation_images, validation_labels])
                    label = label.reshape(1, FLAGS.input_shape[0], FLAGS.input_shape[1])
                    lo, acc = sess.run(
                        [self.loss_mean, self.accuracy],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_training: False}
                    )
                    # summary_writer.add_summary(summary_str, epoch)
                    if epoch % 1 == 0:
                        print('num %d , loss: %.6f , accuracy: %.6f' % (epoch, lo, acc))
                    epoch += 1
            except tf.errors.OutOfRangeError:
                print('Done validating -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)
        print('Done validating')

    def test(self):
        import cv2
        test_file_path = FLAGS.dataset_dir + TEST_SET_NAME
        test_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(test_file_path), num_epochs=1, shuffle=True)
        ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        test_images, test_labels = read_image_batch(test_image_filename_queue, TEST_BATCH_SIZE)
        # tf.summary.scalar("loss", self.loss_mean)
        # tf.summary.scalar('accuracy', self.accuracy)
        # merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sum_acc = 0.0
            try:
                epoch = 0
                while not coord.should_stop():
                    example, label = sess.run([test_images, test_labels])
                    label = label.reshape(1, FLAGS.input_shape[0], FLAGS.input_shape[1])
                    img, acc = sess.run(
                        [tf.argmax(input=self.prediction, axis=3), self.accuracy],
                        feed_dict={
                            self.input_image: example, self.input_label: label,
                            self.keep_prob: 1.0, self.lamb: 0.004, self.is_training: False
                        }
                    )
                    sum_acc += acc
                    epoch += 1
                    cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.png' % epoch), img[0] * 255)
                    if epoch % 1 == 0:
                        print('num %d , accuracy: %.6f' % (epoch, acc))
            except tf.errors.OutOfRangeError:
                print(
                    'Done testing -- epoch limit reached \n Average accuracy: %.2f%%' % (sum_acc / TEST_SET_SIZE * 100))
            finally:
                coord.request_stop()
            coord.join(threads)
        print('Done testing')

    def predict(self):
        from keras.preprocessing import image
        import glob
        import numpy as np
        predict_file_path = glob.glob(os.path.join(ORIGIN_PREDICT_DIRECTORY, '*.png'))
        print("quantity of imgs used to predict is ", len(predict_file_path))
        if not os.path.lexists(PREDICT_SAVED_DIRECTORY):
            os.mkdir(PREDICT_SAVED_DIRECTORY)
        ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")  # CHECK_POINT_PATH
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            for index, image_path in enumerate(predict_file_path):
                original_img = image.load_img(image_path,
                                              target_size=(
                                                  FLAGS.output_shape[0], FLAGS.output_shape[1], FLAGS.output_shape[2]),
                                              color_mode="grayscale")
                original_img = image.img_to_array(original_img)
                img = np.expand_dims(original_img, axis=0)
                print("img shape:", img.shape)
                test = tf.argmax(input=self.prediction, axis=0)
                print("test shape", test.shape)
                print("test", test)
                predict_image = sess.run(
                    tf.argmax(input=self.prediction, axis=0),
                    feed_dict={
                        self.input_image: img, self.keep_prob: 1.0, self.lamb: 0.004, self.is_training: False
                    }
                )
                print("predict_image shape:", predict_image.shape)
                image.save_img(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.png' % index), predict_image)
        print('Done prediction')


def main():
    net = UNet()
    net.build_up_unet(TRAIN_BATCH_SIZE)
    print("❗️start training...")
    net.train()

    tf.reset_default_graph()
    net.build_up_unet(VALIDATION_BATCH_SIZE)
    print("❗️start validating...")
    net.validate()

    tf.reset_default_graph()
    net.build_up_unet(TEST_BATCH_SIZE)
    print("❗️start testing...")
    net.test()

    tf.reset_default_graph()
    net.build_up_unet(PREDICT_BATCH_SIZE)
    print("❗️start predicting...")
    net.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset dir
    parser.add_argument(
        '--dataset_dir', type=str, default=DATASET_DIR,
        help='path of dir storing input dataset')

    # dir for saving models
    parser.add_argument(
        '--model_dir', type=str, default=SAVED_MODELS,
        help='path of dir for saving models')

    # dir for saving logs
    parser.add_argument(
        '--log_dir', type=str, default=LOGS_DIR,
        help='path of dir for saving tensorboard logs')

    # input image shape
    parser.add_argument(
        '--input_shape', type=tuple, default=INPUT_SHAPE,
        help='shape of the input image, channel last, a tuple like (144,112,1)')

    # output image shape
    parser.add_argument(
        '--output_shape', type=tuple, default=OUTPUT_SHAPE,
        help='shape of the output image, channel last, a tuple like (144,112,3)')

    # epoch num
    parser.add_argument(
        '--epoch_num', type=int, default=EPOCH_NUM,
        help='number of epochs')

    FLAGS, _ = parser.parse_known_args()

    if not os.path.exists(DATASET_DIR + TRAIN_SET_NAME):
        convert_npy_to_tfrecords.main()

    main()
