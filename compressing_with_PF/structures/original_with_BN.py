import tensorflow as tf

FLAGS = tf.flags.FLAGS
INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL = eval(FLAGS.input_shape)
EPS = 10e-5
LAMB = 0.004


def kernel_init(shape):
    return tf.truncated_normal_initializer(stddev=tf.sqrt(x=1 / (shape[0] * shape[1] * shape[2])))


def forward_fn(inputs):
    kernel_regu = tf.contrib.layers.l2_regularizer(LAMB)

    # conv1
    with tf.name_scope('layer_1'):
        BN1 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization1_1')(inputs)
        conv1 = tf.layers.conv2d(BN1, int(64 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv1_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(64 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        BN1 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization1_2')(conv1)
        conv1 = tf.layers.conv2d(BN1, int(64 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv1_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(64 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name='pool1', padding="valid")
    # conv2
    with tf.name_scope('layer_2'):
        BN2 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization2_1')(pool1)
        conv2 = tf.layers.conv2d(BN2, int(128 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv2_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(128 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        BN2 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization2_2')(conv2)
        conv2 = tf.layers.conv2d(BN2, int(128 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv2_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(128 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='pool2', padding="valid")
    # conv3
    with tf.name_scope('layer_3'):
        BN3 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization3_1')(pool2)
        conv3 = tf.layers.conv2d(BN3, int(256 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv3_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(256 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        BN3 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization3_2')(conv3)
        conv3 = tf.layers.conv2d(BN3, int(256 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv3_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(256 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), name='pool3', padding="valid")
    # conv4
    with tf.name_scope('layer_4'):
        BN4 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization4_1')(pool3)
        conv4 = tf.layers.conv2d(BN4, int(512 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv4_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(512 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        BN4 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization4_2')(conv4)
        conv4 = tf.layers.conv2d(BN4, int(512 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv4_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(512 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='pool4', padding="valid")
    # conv5
    with tf.name_scope('layer_5'):
        BN5 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization5_1')(pool4)
        conv5 = tf.layers.conv2d(BN5, int(1024 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv5_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(1024 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        BN5 = tf.layers.BatchNormalization(epsilon=EPS, fused=False, name='batch_normalization5_2')(conv5)
        conv5 = tf.layers.conv2d(BN5, int(1024 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv5_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(1024 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)

        up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        conc5 = tf.keras.layers.Concatenate(axis=3)([up5, conv4])
    # conv6
    with tf.name_scope('layer_6'):
        conv6 = tf.layers.conv2d(conc5, int(512 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv6_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(512 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        conv6 = tf.layers.conv2d(conv6, int(512 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv6_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(512 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)

        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        conc6 = tf.keras.layers.Concatenate(axis=3)([up6, conv3])
    # conv7
    with tf.name_scope('layer_7'):
        conv7 = tf.layers.conv2d(conc6, int(256 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv7_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(256 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        conv7 = tf.layers.conv2d(conv7, int(256 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv7_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(256 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)

        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
        conc7 = tf.keras.layers.Concatenate(axis=3)([up7, conv2])
    # conv8
    with tf.name_scope('layer_8'):
        conv8 = tf.layers.conv2d(conc7, int(128 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv8_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(128 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        conv8 = tf.layers.conv2d(conv8, int(128 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv8_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(128 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)

        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
        conc8 = tf.keras.layers.Concatenate(axis=3)([up8, conv1])
    # conv9
    with tf.name_scope('layer_9'):
        conv9 = tf.layers.conv2d(conc8, int(64 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv9_1',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(64 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        conv9 = tf.layers.conv2d(conv9, int(64 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv9_2',
                                 padding="same", kernel_initializer=kernel_init([3, 3, int(64 / FLAGS.divisor)]),
                                 kernel_regularizer=kernel_regu)
        conv9 = tf.layers.conv2d(conv9, 3, 3, 1, activation=tf.nn.relu, name='conv9_3', padding="same",
                                 kernel_initializer=kernel_init([3, 3, 3]),
                                 kernel_regularizer=kernel_regu)
        return conv9
