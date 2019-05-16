import tensorflow as tf
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, UpSampling2D, Concatenate, Input

FLAGS = tf.flags.FLAGS
INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL = eval(FLAGS.input_shape)


def forward_rn(inputs):
    IN = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL), tensor=inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(IN)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=None)(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up5 = UpSampling2D(size=(2, 2))(conv5)
    conc5 = Concatenate(axis=3)([up5, conv4])
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conc5)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up6 = UpSampling2D(size=(2, 2))(conv6)
    conc6 = Concatenate(axis=3)([up6, conv3])
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conc6)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up7 = UpSampling2D(size=(2, 2))(conv7)
    conc7 = Concatenate(axis=3)([up7, conv2])
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conc7)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up8 = UpSampling2D(size=(2, 2))(conv8)
    conc8 = Concatenate(axis=3)([up8, conv1])
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conc8)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = Convolution2D(3, 3, 3, activation='relu', border_mode='same')(conv9)
    return conv9
