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

        conv1 = Convolution2D(filters=32, kernel_size=3, activation='relu', border_mode='same')(BN1)
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

        conv10 = Convolution2D(class_num, 1, 1, activation='softmax')(conv9)

        model = keras.models.Model(input=[IN], output=conv10)

        print(model.summary())
        return model

# shape of each layer
# 🚩IN shape (?, 144, 112, 1)
# 🚩BN1 shape (?, 144, 112, 1)
# 🚩️conv1—— shape (?, 144, 112, 32)
# 🚩️conv1 shape (?, 144, 112, 32)
# 🚩pool1 shape (?, 72, 56, 32)
# 🚩️conv2 shape (?, 72, 56, 64)
# 🚩️conv2 shape (?, 72, 56, 64)
# 🚩pool2 shape (?, 36, 28, 64)
# 🚩conv3 shape (?, 36, 28, 128)
# 🚩️conv3 shape (?, 36, 28, 128)
# 🚩pool3 shape (?, 18, 14, 128)
# 🚩️conv4 shape (?, 18, 14, 256)
# 🚩️conv4 shape (?, 18, 14, 256)
# 🚩pool4 shape (?, 9, 7, 256)
# 🚩️conv5 shape (?, 9, 7, 512)
# 🚩️conv5 shape (?, 9, 7, 512)
# 🚩up5 shape (?, 18, 14, 512)
# 🚩conc5 shape (?, 18, 14, 768)
# 🚩️conv6 shape (?, 18, 14, 256)
# 🚩️conv6 shape (?, 18, 14, 256)
# 🚩up6 shape (?, 36, 28, 256)
# 🚩️conc6 shape (?, 36, 28, 384)
# 🚩️conv7 shape (?, 36, 28, 128)
# 🚩️conv7 shape (?, 36, 28, 128)
# 🚩up7 shape (?, 72, 56, 128)
# 🚩️conc7 shape (?, 72, 56, 192)
# 🚩️conv8 shape (?, 72, 56, 64)
# 🚩️conv8 shape (?, 72, 56, 64)
# 🚩up8 shape (?, 144, 112, 64)
# 🚩️conc8 shape (?, 144, 112, 96)
# 🚩️conv9 shape (?, 144, 112, 32)
# 🚩️conv9 shape (?, 144, 112, 32)
# 🚩️conv10 shape (?, 144, 112, 3)
