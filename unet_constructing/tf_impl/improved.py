import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from unet_constructing.tf_impl.utils import *

FLAGS = tf.flags.FLAGS
INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL = eval(FLAGS.input_shape)


class UNet:
    def __init__(self):
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

    def init_w(self, shape, name):
        with tf.name_scope('init_w'):
            # stddev = 0.01
            stddev = tf.sqrt(x=2 / (shape[0] * shape[1] * shape[2]))
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32), name=name)
            tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(w))
            return w

    def build_up_unet(self, batch_size):
        # the contracting path
        # input layer
        with tf.name_scope('input'):
            self.input_image = tf.placeholder(
                dtype=tf.float32,
                shape=[batch_size, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL],
                name='input_images'
            )
            self.input_label = tf.placeholder(
                dtype=tf.int32, shape=[batch_size, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL],
                name='input_labels'
            )

            # keep_prob is to define how many neurons to keep when dropping out
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

            # lamb is used to control the size of the Regularization term,
            # and a larger value of lamb will constrain the complexity of the model to a greater extent
            # The purpose is to avoid model overfitting
            self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')

            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # layer 1
        with tf.name_scope('layer_1'):
            normed_batch = batch_norm(x=self.input_image, is_training=self.is_training, name='batch_normalization1_1')
            # conv_1
            self.w[1] = self.init_w(shape=[3, 3, INPUT_CHANNEL, int(64 / FLAGS.divisor)], name='w_1')
            conv_1_result = tf.nn.conv2d(
                input=normed_batch, filter=self.w[1], strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
            relu_1_result = tf.nn.relu(conv_1_result, name='relu1_1')

            normed_batch = batch_norm(x=relu_1_result, is_training=self.is_training, name='batch_normalization1_2')
            # conv_2
            self.w[2] = self.init_w(shape=[3, 3, int(64 / FLAGS.divisor), int(64 / FLAGS.divisor)], name='w_2')
            conv_2_result = tf.nn.conv2d(input=normed_batch, filter=self.w[2], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv1_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu1_2')
            self.result_from_contracting[1] = relu_2_result  # saved for up sampling below

            # maxpool
            maxpool_result = tf.nn.max_pool(value=relu_2_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='VALID', name='pool1')

        # layer 2
        with tf.name_scope('layer_2'):
            normed_batch = batch_norm(x=maxpool_result, is_training=self.is_training, name='batch_normalization2_1')
            # conv_1
            self.w[3] = self.init_w(shape=[3, 3, int(64 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='w_3')
            conv_1_result = tf.nn.conv2d(input=normed_batch, filter=self.w[3], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv2_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu2_1')

            normed_batch = batch_norm(x=relu_1_result, is_training=self.is_training, name='batch_normalization2_2')
            # conv_2
            self.w[4] = self.init_w(shape=[3, 3, int(128 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='w_4')
            conv_2_result = tf.nn.conv2d(input=normed_batch, filter=self.w[4], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv2_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu2_2')
            self.result_from_contracting[2] = relu_2_result  # saved for up sampling below

            # maxpooling
            maxpool_result = tf.nn.max_pool(value=relu_2_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='VALID', name='pool2')

        # layer 3
        with tf.name_scope('layer_3'):
            normed_batch = batch_norm(x=maxpool_result, is_training=self.is_training, name='batch_normalization3_1')
            # conv_1
            self.w[5] = self.init_w(shape=[3, 3, int(128 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='w_5')
            conv_1_result = tf.nn.conv2d(input=normed_batch, filter=self.w[5], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv3_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu3_1')

            normed_batch = batch_norm(x=relu_1_result, is_training=self.is_training, name='batch_normalization3_2')
            # conv_2
            self.w[6] = self.init_w(shape=[3, 3, int(256 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='w_6')
            conv_2_result = tf.nn.conv2d(input=normed_batch, filter=self.w[6], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv3_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu3_2')
            self.result_from_contracting[3] = relu_2_result  # saved for up sampling below

            # maxpool
            maxpool_result = tf.nn.max_pool(
                value=relu_2_result, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='pool3')

        # layer 4
        with tf.name_scope('layer_4'):
            normed_batch = batch_norm(x=maxpool_result, is_training=self.is_training, name='batch_normalization4_1')
            # conv_1
            self.w[7] = self.init_w(shape=[3, 3, int(256 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='w_7')
            conv_1_result = tf.nn.conv2d(input=normed_batch, filter=self.w[7], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv4_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu4_1')

            normed_batch = batch_norm(x=relu_1_result, is_training=self.is_training, name='batch_normalization4_2')
            # conv_2
            self.w[8] = self.init_w(shape=[3, 3, int(512 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='w_8')
            conv_2_result = tf.nn.conv2d(input=normed_batch, filter=self.w[8], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv4_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu4_2')
            self.result_from_contracting[4] = relu_2_result  # saved for up sampling below

            # maxpool
            maxpool_result = tf.nn.max_pool(
                value=relu_2_result, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='pool4')

        # the bottom
        # layer 5
        with tf.name_scope('layer_5'):
            normed_batch = batch_norm(x=maxpool_result, is_training=self.is_training, name='batch_normalization5_1')
            # conv_1
            self.w[9] = self.init_w(shape=[3, 3, int(512 / FLAGS.divisor), int(1024 / FLAGS.divisor)], name='w_9')
            conv_1_result = tf.nn.conv2d(input=normed_batch, filter=self.w[9], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv5_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu5_1')

            normed_batch = batch_norm(x=relu_1_result, is_training=self.is_training, name='batch_normalization5_2')
            # conv_2
            self.w[10] = self.init_w(shape=[3, 3, int(1024 / FLAGS.divisor), int(1024 / FLAGS.divisor)], name='w_10')
            conv_2_result = tf.nn.conv2d(input=normed_batch, filter=self.w[10], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv5_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu5_2')

            # up_sampling
            self.w[11] = self.init_w(shape=[2, 2, int(512 / FLAGS.divisor), int(1024 / FLAGS.divisor)], name='w_11')
            result_up = tf.nn.conv2d_transpose(value=relu_2_result, filter=self.w[11], strides=[1, 2, 2, 1],
                                               output_shape=[batch_size, int(relu_2_result.shape[1]) * 2,
                                                             int(relu_2_result.shape[2]) * 2, int(512 / FLAGS.divisor)],
                                               padding='VALID', name='Up_Sampling5')
            relu_3_result = tf.nn.relu(features=result_up, name='relu5_3')

        # the expanding path
        # layer 6
        with tf.name_scope('layer_6'):
            # copy, crop and merge
            result_merge = merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[4], result_from_upsampling=relu_3_result)

            # conv_1
            self.w[12] = self.init_w(shape=[3, 3, int(1024 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='w_12')
            conv_1_result = tf.nn.conv2d(input=result_merge, filter=self.w[12], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv6_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu6_1')
            # conv_2
            self.w[13] = self.init_w(shape=[3, 3, int(512 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='w_10')
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[13], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv6_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu6_2')

            # up_sampling
            self.w[14] = self.init_w(shape=[2, 2, int(256 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='w_11')
            result_up = tf.nn.conv2d_transpose(value=relu_2_result, filter=self.w[14], strides=[1, 2, 2, 1],
                                               output_shape=[batch_size, int(relu_2_result.shape[1]) * 2,
                                                             int(relu_2_result.shape[2]) * 2, int(256 / FLAGS.divisor)],
                                               padding='VALID', name='Up_Sampling6')
            relu_3_result = tf.nn.relu(features=result_up, name='relu6_3')

        # layer 7
        with tf.name_scope('layer_7'):
            # copy, crop and merge
            result_merge = merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[3], result_from_upsampling=relu_3_result)

            # conv_1
            self.w[15] = self.init_w(shape=[3, 3, int(512 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='w_12')
            conv_1_result = tf.nn.conv2d(input=result_merge, filter=self.w[15], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv7_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu7_1')
            # conv_2
            self.w[16] = self.init_w(shape=[3, 3, int(256 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='w_10')
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[16], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv7_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu7_2')

            # up_sampling
            self.w[17] = self.init_w(shape=[2, 2, int(128 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='w_11')
            result_up = tf.nn.conv2d_transpose(value=relu_2_result, filter=self.w[17], strides=[1, 2, 2, 1],
                                               output_shape=[batch_size, int(relu_2_result.shape[1]) * 2,
                                                             int(relu_2_result.shape[2]) * 2, int(128 / FLAGS.divisor)],
                                               padding='VALID', name='Up_Sampling7')
            relu_3_result = tf.nn.relu(features=result_up, name='relu7_3')

        # layer 8
        with tf.name_scope('layer_8'):
            # copy, crop and merge
            result_merge = merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[2], result_from_upsampling=relu_3_result)

            # conv_1
            self.w[18] = self.init_w(shape=[3, 3, int(256 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='w_12')
            conv_1_result = tf.nn.conv2d(input=result_merge, filter=self.w[18], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv8_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu8_1')
            # conv_2
            self.w[19] = self.init_w(shape=[3, 3, int(128 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='w_10')
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[19], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv8_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu8_2')

            # up_sampling
            self.w[20] = self.init_w(shape=[2, 2, int(64 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='w_11')
            result_up = tf.nn.conv2d_transpose(value=relu_2_result, filter=self.w[20], strides=[1, 2, 2, 1],
                                               output_shape=[batch_size, int(relu_2_result.shape[1]) * 2,
                                                             int(relu_2_result.shape[2]) * 2, int(64 / FLAGS.divisor)],
                                               padding='VALID', name='Up_Sampling8')
            relu_3_result = tf.nn.relu(features=result_up, name='relu8_3')

        # layer 9
        with tf.name_scope('layer_9'):
            # copy, crop and merge
            result_merge = merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[1], result_from_upsampling=relu_3_result)

            # conv_1
            self.w[21] = self.init_w(shape=[3, 3, int(128 / FLAGS.divisor), int(64 / FLAGS.divisor)], name='w_12')
            conv_1_result = tf.nn.conv2d(input=result_merge, filter=self.w[21], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv9_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu9_1')
            # conv_2
            self.w[22] = self.init_w(shape=[3, 3, int(64 / FLAGS.divisor), int(64 / FLAGS.divisor)], name='w_10')
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[22], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv9_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu9_2')

            # convolution to [batch_size, OUTPIT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM]
            self.w[23] = self.init_w(shape=[1, 1, int(64 / FLAGS.divisor), 3], name='w_11')
            result_conv_3 = tf.nn.conv2d(input=relu_2_result, filter=self.w[23], strides=[1, 1, 1, 1],
                                         padding='VALID', name='conv9_3')

        with tf.name_scope('softmax'):
            result = tf.nn.softmax(logits=result_conv_3)

        self.prediction = result

        # softmax loss
        with tf.name_scope('softmax_loss'):
            self.input_label = tf.reshape(self.input_label, [batch_size, INPUT_HEIGHT, INPUT_WIDTH])
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label,
                                                                       logits=self.prediction,
                                                                       name='loss')

            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        # accuracy
        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32),
                                               self.input_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)

        # Gradient Descent
        with tf.name_scope('Gradient_Descent'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)
