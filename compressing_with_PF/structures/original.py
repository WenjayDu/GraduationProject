import tensorflow as tf

FLAGS = tf.flags.FLAGS
INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL = eval(FLAGS.input_shape)

LAMB = 0.004


def kernel_init(shape):
    return tf.truncated_normal_initializer(stddev=tf.sqrt(x=1 / (shape[0] * shape[1] * shape[2])))


def forward_fn(inputs):
    kernel_regu = tf.contrib.layers.l2_regularizer(LAMB)
    # conv1
    with tf.name_scope('layer_1'):
        conv1 = tf.layers.conv2d(inputs, int(64 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv1_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(64 / FLAGS.divisor)]))
        conv1 = tf.layers.conv2d(conv1, int(64 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv1_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(64 / FLAGS.divisor)]))
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name='pool1', padding="valid")
    # conv2
    with tf.name_scope('layer_2'):
        conv2 = tf.layers.conv2d(pool1, int(128 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv2_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(128 / FLAGS.divisor)]))
        conv2 = tf.layers.conv2d(conv2, int(128 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv2_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(128 / FLAGS.divisor)]))
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='pool2', padding="valid")
    # conv3
    with tf.name_scope('layer_3'):
        conv3 = tf.layers.conv2d(pool2, int(256 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv3_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(256 / FLAGS.divisor)]))
        conv3 = tf.layers.conv2d(conv3, int(256 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv3_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(256 / FLAGS.divisor)]))
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), name='pool3', padding="valid")
    # conv4
    with tf.name_scope('layer_4'):
        conv4 = tf.layers.conv2d(pool3, int(512 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv4_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(512 / FLAGS.divisor)]))
        conv4 = tf.layers.conv2d(conv4, int(512 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv4_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(512 / FLAGS.divisor)]))
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='pool4', padding="valid")
    # conv5
    with tf.name_scope('layer_5'):
        conv5 = tf.layers.conv2d(pool4, int(1024 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv5_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(1024 / FLAGS.divisor)]))
        conv5 = tf.layers.conv2d(conv5, int(1024 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv5_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(1024 / FLAGS.divisor)]))
        up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        conc5 = tf.keras.layers.Concatenate(axis=3)([up5, conv4])
    # conv6
    with tf.name_scope('layer_6'):
        conv6 = tf.layers.conv2d(conc5, int(512 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv6_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(512 / FLAGS.divisor)]))
        conv6 = tf.layers.conv2d(conv6, int(512 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv6_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(512 / FLAGS.divisor)]))

        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        conc6 = tf.keras.layers.Concatenate(axis=3)([up6, conv3])
    # conv7
    with tf.name_scope('layer_7'):
        conv7 = tf.layers.conv2d(conc6, int(256 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv7_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(256 / FLAGS.divisor)]))
        conv7 = tf.layers.conv2d(conv7, int(256 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv7_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(256 / FLAGS.divisor)]))

        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
        conc7 = tf.keras.layers.Concatenate(axis=3)([up7, conv2])
    # conv8
    with tf.name_scope('layer_8'):
        conv8 = tf.layers.conv2d(conc7, int(128 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv8_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(128 / FLAGS.divisor)]))
        conv8 = tf.layers.conv2d(conv8, int(128 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv8_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(128 / FLAGS.divisor)]))

        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
        conc8 = tf.keras.layers.Concatenate(axis=3)([up8, conv1])
    # conv9
    with tf.name_scope('layer_9'):
        conv9 = tf.layers.conv2d(conc8, int(64 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv9_1',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(64 / FLAGS.divisor)]))
        conv9 = tf.layers.conv2d(conv9, int(64 / FLAGS.divisor), 3, 1, activation=tf.nn.relu, name='conv9_2',
                                 padding="same", kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, int(64 / FLAGS.divisor)]))
        conv9 = tf.layers.conv2d(conv9, 3, 3, 1, activation=tf.nn.relu, name='conv9_3', padding="same",
                                 kernel_regularizer=kernel_regu,
                                 kernel_initializer=kernel_init([3, 3, 3]))

    return conv9
