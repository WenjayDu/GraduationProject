import tensorflow as tf
from keras.layers import BatchNormalization

FLAGS = tf.flags.FLAGS


def forward_rn(inputs):
    # conv1
    with tf.name_scope('layer_1'):
        BN1 = BatchNormalization()(inputs)
        conv1 = tf.layers.conv2d(BN1, 64, 3, 1, activation=tf.nn.relu, name='conv1_1', padding="same",
                                 use_bias=False)
        BN1 = BatchNormalization()(conv1)
        conv1 = tf.layers.conv2d(BN1, 64, 3, 1, activation=tf.nn.relu, name='conv1_2', padding="same",
                                 use_bias=False)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name='pool1', padding="valid")
    # conv2
    with tf.name_scope('layer_2'):
        BN2 = BatchNormalization()(pool1)
        conv2 = tf.layers.conv2d(BN2, 128, 3, 1, activation=tf.nn.relu, name='conv2_1', padding="same",
                                 use_bias=False)
        BN2 = BatchNormalization()(conv2)
        conv2 = tf.layers.conv2d(BN2, 128, 3, 1, activation=tf.nn.relu, name='conv2_2', padding="same",
                                 use_bias=False)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='pool2', padding="valid")
    # conv3
    with tf.name_scope('layer_3'):
        BN3 = BatchNormalization()(pool2)
        conv3 = tf.layers.conv2d(BN3, 256, 3, 1, activation=tf.nn.relu, name='conv3_1', padding="same",
                                 use_bias=False)
        BN3 = BatchNormalization()(conv3)
        conv3 = tf.layers.conv2d(BN3, 256, 3, 1, activation=tf.nn.relu, name='conv3_2', padding="same",
                                 use_bias=False)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), name='pool3', padding="valid")
    # conv4
    with tf.name_scope('layer_4'):
        BN4 = BatchNormalization()(pool3)
        conv4 = tf.layers.conv2d(BN4, 512, 3, 1, activation=tf.nn.relu, name='conv4_1', padding="same",
                                 use_bias=False)
        BN4 = BatchNormalization()(conv4)
        conv4 = tf.layers.conv2d(BN4, 512, 3, 1, activation=tf.nn.relu, name='conv4_2', padding="same",
                                 use_bias=False)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='pool4', padding="valid")
    # conv5
    with tf.name_scope('layer_5'):
        BN5 = BatchNormalization()(pool4)
        conv5 = tf.layers.conv2d(BN5, 1024, 3, 1, activation=tf.nn.relu, name='conv5_1', padding="same",
                                 use_bias=False)
        BN5 = BatchNormalization()(conv5)
        conv5 = tf.layers.conv2d(BN5, 1024, 3, 1, activation=tf.nn.relu, name='conv5_2', padding="same",
                                 use_bias=False)

        up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        conc5 = tf.keras.layers.Concatenate(axis=3)([up5, conv4])

    # conv6
    with tf.name_scope('layer_6'):
        conv6 = tf.layers.conv2d(conc5, 512, 3, 1, activation=tf.nn.relu, name='conv6_1', padding="same",
                                 use_bias=False)
        conv6 = tf.layers.conv2d(conv6, 512, 3, 1, activation=tf.nn.relu, name='conv6_2', padding="same",
                                 use_bias=False)

        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        conc6 = tf.keras.layers.Concatenate(axis=3)([up6, conv3])

    # conv7
    with tf.name_scope('layer_7'):
        conv7 = tf.layers.conv2d(conc6, 256, 3, 1, activation=tf.nn.relu, name='conv7_1', padding="same",
                                 use_bias=False)
        conv7 = tf.layers.conv2d(conv7, 256, 3, 1, activation=tf.nn.relu, name='conv7_2', padding="same",
                                 use_bias=False)

        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
        conc7 = tf.keras.layers.Concatenate(axis=3)([up7, conv2])

    # conv8
    with tf.name_scope('layer_8'):
        conv8 = tf.layers.conv2d(conc7, 128, 3, 1, activation=tf.nn.relu, name='conv8_1', padding="same",
                                 use_bias=False)
        conv8 = tf.layers.conv2d(conv8, 128, 3, 1, activation=tf.nn.relu, name='conv8_2', padding="same",
                                 use_bias=False)

        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
        conc8 = tf.keras.layers.Concatenate(axis=3)([up8, conv1])

    # conv9
    with tf.name_scope('layer_9'):
        conv9 = tf.layers.conv2d(conc8, 64, 3, 1, activation=tf.nn.relu, name='conv9_1', padding="same",
                                 use_bias=False)
        conv9 = tf.layers.conv2d(conv9, 64, 3, 1, activation=tf.nn.relu, name='conv9_2', padding="same",
                                 use_bias=False)
        conv9 = tf.layers.conv2d(conv9, 3, 3, 1, activation=tf.nn.relu, name='conv9_3', padding="same",
                                 use_bias=False)
    return conv9