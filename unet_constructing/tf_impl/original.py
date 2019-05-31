import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from unet_constructing.tf_impl.utils import *

FLAGS = tf.flags.FLAGS
INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, INPUT_IMG_CHANNEL = eval(FLAGS.input_shape)


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

    def init_w(self, shape, name, scope_name):
        with tf.name_scope(scope_name):
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
                shape=[batch_size, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, INPUT_IMG_CHANNEL],
                name='input_images'
            )
            self.input_label = tf.placeholder(
                dtype=tf.int32, shape=[batch_size, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, INPUT_IMG_CHANNEL],
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
            # conv_1
            self.w[1] = self.init_w(shape=[3, 3, INPUT_IMG_CHANNEL, int(64 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv1_1")
            conv_1_result = tf.nn.conv2d(input=self.input_image, filter=self.w[1], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv1_1')
            relu_1_result = tf.nn.relu(conv_1_result, name='relu_1')
            # conv_2
            self.w[2] = self.init_w(shape=[3, 3, int(64 / FLAGS.divisor), int(64 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv1_2")
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[2], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv1_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')
            self.result_from_contracting[1] = relu_2_result  # saved for up sampling below
            # maxpool
            maxpool_result = tf.nn.max_pool(value=relu_2_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='VALID', name='pool1')

        # layer 2
        with tf.name_scope('layer_2'):
            # conv_1
            self.w[3] = self.init_w(shape=[3, 3, int(64 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv2_1")
            conv_1_result = tf.nn.conv2d(input=maxpool_result, filter=self.w[3], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv2_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu_1')
            # conv_2
            self.w[4] = self.init_w(shape=[3, 3, int(128 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv2_2")
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[4], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv2_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')
            self.result_from_contracting[2] = relu_2_result  # saved for up sampling below
            # maxpooling
            maxpool_result = tf.nn.max_pool(value=relu_2_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='VALID', name='pool2')

        # layer 3
        with tf.name_scope('layer_3'):
            # conv_1
            self.w[5] = self.init_w(shape=[3, 3, int(128 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv3_1")
            conv_1_result = tf.nn.conv2d(
                input=maxpool_result, filter=self.w[5], strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu_1')
            # conv_2
            self.w[6] = self.init_w(shape=[3, 3, int(256 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv3_2")
            conv_2_result = tf.nn.conv2d(
                input=relu_1_result, filter=self.w[6], strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')
            self.result_from_contracting[3] = relu_2_result  # saved for up sampling below
            # maxpool
            maxpool_result = tf.nn.max_pool(value=relu_2_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='VALID', name='pool3')

        # layer 4
        with tf.name_scope('layer_4'):
            # conv_1
            self.w[7] = self.init_w(shape=[3, 3, int(256 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv4_1")
            conv_1_result = tf.nn.conv2d(input=maxpool_result, filter=self.w[7], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv4_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu_1')
            # conv_2
            self.w[8] = self.init_w(shape=[3, 3, int(512 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv4_2")
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[8], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv4_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')
            self.result_from_contracting[4] = relu_2_result  # saved for up sampling below
            # maxpool
            maxpool_result = tf.nn.max_pool(value=relu_2_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='VALID', name='pool4')

        # the bottom
        # layer 5
        with tf.name_scope('layer_5'):
            # conv_1
            self.w[9] = self.init_w(shape=[3, 3, int(512 / FLAGS.divisor), int(1024 / FLAGS.divisor)], name='kernel',
                                    scope_name="conv5_1")
            conv_1_result = tf.nn.conv2d(input=maxpool_result, filter=self.w[9], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv5_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu_1')
            # conv_2
            self.w[10] = self.init_w(shape=[3, 3, int(1024 / FLAGS.divisor), int(1024 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv5_2")
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[10], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv5_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')

            # up_sampling
            self.w[11] = self.init_w(shape=[2, 2, int(512 / FLAGS.divisor), int(1024 / FLAGS.divisor)], name='kernel',
                                     scope_name="Up_Sampling_5")
            result_up = tf.nn.conv2d_transpose(value=relu_2_result, filter=self.w[11], strides=[1, 2, 2, 1],
                                               output_shape=[batch_size, int(relu_2_result.shape[1]) * 2,
                                                             int(relu_2_result.shape[2]) * 2, int(512 / FLAGS.divisor)],
                                               padding='VALID', name='Up_Sampling_5')
            relu_3_result = tf.nn.relu(features=result_up, name='relu_3')

        # the expanding path
        # layer 6
        with tf.name_scope('layer_6'):
            # copy, crop and merge
            result_merge = merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[4], result_from_upsampling=relu_3_result)

            # conv_1
            self.w[12] = self.init_w(shape=[3, 3, int(1024 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv6_1")
            conv_1_result = tf.nn.conv2d(input=result_merge, filter=self.w[12], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv6_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu_1')
            # conv_2
            self.w[13] = self.init_w(shape=[3, 3, int(512 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv6_2")
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[13], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv6_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')

            # up_sampling
            self.w[14] = self.init_w(shape=[2, 2, int(256 / FLAGS.divisor), int(512 / FLAGS.divisor)], name='kernel',
                                     scope_name="Up_Sampling_6")
            result_up = tf.nn.conv2d_transpose(value=relu_2_result, filter=self.w[14], strides=[1, 2, 2, 1],
                                               output_shape=[batch_size, int(relu_2_result.shape[1]) * 2,
                                                             int(relu_2_result.shape[2]) * 2, int(256 / FLAGS.divisor)],
                                               padding='VALID', name='Up_Sampling_6')
            relu_3_result = tf.nn.relu(features=result_up, name='relu_3')

        # layer 7
        with tf.name_scope('layer_7'):
            # copy, crop and merge
            result_merge = merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[3], result_from_upsampling=relu_3_result)

            # conv_1
            self.w[15] = self.init_w(shape=[3, 3, int(512 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv7_1")
            conv_1_result = tf.nn.conv2d(input=result_merge, filter=self.w[15], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv7_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu_1')
            # conv_2
            self.w[16] = self.init_w(shape=[3, 3, int(256 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv7_2")
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[16], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv7_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')

            # up_sampling
            self.w[17] = self.init_w(shape=[2, 2, int(128 / FLAGS.divisor), int(256 / FLAGS.divisor)], name='kernel',
                                     scope_name="Up_Sampling_7")
            result_up = tf.nn.conv2d_transpose(value=relu_2_result, filter=self.w[17], strides=[1, 2, 2, 1],
                                               output_shape=[batch_size, int(relu_2_result.shape[1]) * 2,
                                                             int(relu_2_result.shape[2]) * 2, int(128 / FLAGS.divisor)],
                                               padding='VALID', name='Up_Sampling_7')
            relu_3_result = tf.nn.relu(features=result_up, name='relu_3')

        # layer 8
        with tf.name_scope('layer_8'):
            # copy, crop and merge
            result_merge = merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[2], result_from_upsampling=relu_3_result)

            # conv_1
            self.w[18] = self.init_w(shape=[3, 3, int(256 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv8_1")
            conv_1_result = tf.nn.conv2d(input=result_merge, filter=self.w[18], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv8_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu_1')
            # conv_2
            self.w[19] = self.init_w(shape=[3, 3, int(128 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv8_2")
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[19], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv8_2')
            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')

            # up_sampling
            self.w[20] = self.init_w(shape=[2, 2, int(64 / FLAGS.divisor), int(128 / FLAGS.divisor)], name='kernel',
                                     scope_name="Up_Sampling_8")
            result_up = tf.nn.conv2d_transpose(value=relu_2_result, filter=self.w[20], strides=[1, 2, 2, 1],
                                               output_shape=[batch_size, int(relu_2_result.shape[1]) * 2,
                                                             int(relu_2_result.shape[2]) * 2, int(64 / FLAGS.divisor)],
                                               padding='VALID', name='Up_Sampling_8')
            relu_3_result = tf.nn.relu(features=result_up, name='relu_3')

        # layer 9
        with tf.name_scope('layer_9'):
            # copy, crop and merge
            result_merge = merge_results_from_contracting_and_upsampling(
                result_from_contracting=self.result_from_contracting[1], result_from_upsampling=relu_3_result)

            # conv_1
            self.w[21] = self.init_w(shape=[3, 3, int(128 / FLAGS.divisor), int(64 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv9_1")
            conv_1_result = tf.nn.conv2d(input=result_merge, filter=self.w[21], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv9_1')
            relu_1_result = tf.nn.relu(features=conv_1_result, name='relu_1')
            # conv_2
            self.w[22] = self.init_w(shape=[3, 3, int(64 / FLAGS.divisor), int(64 / FLAGS.divisor)], name='kernel',
                                     scope_name="conv9_2")
            conv_2_result = tf.nn.conv2d(input=relu_1_result, filter=self.w[22], strides=[1, 1, 1, 1],
                                         padding='SAME', name='conv9_2')

            relu_2_result = tf.nn.relu(features=conv_2_result, name='relu_2')
            self.w[23] = self.init_w(shape=[1, 1, int(64 / FLAGS.divisor), 3], name='kernel', scope_name="conv9_3")
            result_conv_3 = tf.nn.conv2d(input=relu_2_result, filter=self.w[23], strides=[1, 1, 1, 1],
                                         padding='VALID', name='conv9_3')

            self.prediction = result_conv_3

        # softmax
        with tf.name_scope('softmax_loss'):
            # using one-hot
            # self.loss = \
            # 	tf.nn.softmax_cross_entropy_with_logits(labels=self.cast_label, logits=self.prediction, name='loss')

            # not using one-hot
            # make self.input_label's rank -1
            self.input_label = tf.reshape(self.input_label, [batch_size, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH])
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label,
                                                                       logits=self.prediction,
                                                                       name='loss')
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        # accuracy
        with tf.name_scope('accuracy'):
            # using one-hot
            # self.correct_prediction = tf.equal(tf.argmax(self.prediction, axis=3), tf.argmax(self.cast_label, axis=3))

            # not using one-hot
            self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32),
                                               self.input_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)

        # Gradient Descent
        with tf.name_scope('Gradient_Descent'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)
