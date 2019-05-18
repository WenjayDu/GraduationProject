import tensorflow as tf

from module_minc_keras.minc_keras import *

FLAGS = tf.flags.FLAGS


class UNet:
    @staticmethod
    def build_up_unet(data, class_num):
        """
        Define the structure of U-Net
        """

        IN = Input(shape=(data['image_dim'][1], data['image_dim'][2], 1))

        BN1 = BatchNormalization()(IN)
        conv1 = Convolution2D(64 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN1)
        BN1 = BatchNormalization()(conv1)
        conv1 = Convolution2D(64 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=None)(conv1)

        BN2 = BatchNormalization()(pool1)
        conv2 = Convolution2D(128 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN2)
        BN2 = BatchNormalization()(conv2)
        conv2 = Convolution2D(128 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        BN3 = BatchNormalization()(pool2)
        conv3 = Convolution2D(256 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN3)
        BN3 = BatchNormalization()(conv3)
        conv3 = Convolution2D(256 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        BN4 = BatchNormalization()(pool3)
        conv4 = Convolution2D(512 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN4)
        BN4 = BatchNormalization()(conv4)
        conv4 = Convolution2D(512 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        BN5 = BatchNormalization()(pool4)
        conv5 = Convolution2D(1024 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN5)
        BN5 = BatchNormalization()(conv5)
        conv5 = Convolution2D(1024 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(BN5)

        up5 = UpSampling2D(size=(2, 2))(conv5)
        conc5 = Concatenate(axis=3)([up5, conv4])
        conv6 = Convolution2D(512 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(conc5)
        conv6 = Convolution2D(512 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(conv6)

        up6 = UpSampling2D(size=(2, 2))(conv6)
        conc6 = Concatenate(axis=3)([up6, conv3])
        conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conc6)
        conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

        up7 = UpSampling2D(size=(2, 2))(conv7)
        conc7 = Concatenate(axis=3)([up7, conv2])
        conv8 = Convolution2D(128 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(conc7)
        conv8 = Convolution2D(128 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(conv8)

        up8 = UpSampling2D(size=(2, 2))(conv8)
        conc8 = Concatenate(axis=3)([up8, conv1])
        conv9 = Convolution2D(64 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(conc8)
        conv9 = Convolution2D(64 / FLAGS.divisor, 3, 3, activation='relu', border_mode='same')(conv9)

        conv10 = Convolution2D(class_num, 1, 1, activation='softmax')(conv9)

        model = keras.models.Model(input=[IN], output=conv10)
        print(model.summary())
        return model
