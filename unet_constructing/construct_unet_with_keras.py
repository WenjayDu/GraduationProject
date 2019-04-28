import os
import sys
import subprocess
from keras.callbacks import TensorBoard

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pickle
from config_and_utils import GlobalVar
from data_processing.prepare_datasets import prepare_mri_dataset
from module_minc_keras.minc_keras import *

PROJECT_DIR = GlobalVar.PROJECT_PATH
DATASET_DIR = GlobalVar.DATASET_PATH
OUTPUT_DIR = GlobalVar.OUTPUT_PATH + "/keras_implementation"
SERIALIZE_FILE = GlobalVar.DATASET_PATH + "/mri_pad_4_results/prepare_mri_dataset_return"

LOGS_DIR = OUTPUT_DIR + "/logs"
SAVED_MODELS_DIR = OUTPUT_DIR + "/saved_models"


def main():
    if os.path.exists(SERIALIZE_FILE):
        with open(SERIALIZE_FILE, "rb") as f:
            print("Done deserializing file:", SERIALIZE_FILE)
            [images_mri_pad_4, data_mri_pad_4] = pickle.load(f)
    else:
        [images_mri_pad_4, data_mri_pad_4] = prepare_mri_dataset()

    # Load data
    Y_validate_mri_pad_4 = np.load(data_mri_pad_4["validate_y_fn"] + '.npy')
    nlabels_mri_pad_4 = len(np.unique(Y_validate_mri_pad_4))

    X_train_mri_pad_4 = np.load(data_mri_pad_4["train_x_fn"] + '.npy')
    Y_train_mri_pad_4 = np.load(data_mri_pad_4["train_y_fn"] + '.npy')
    X_validate_mri_pad_4 = np.load(data_mri_pad_4["validate_x_fn"] + '.npy')

    X_test_mri_pad_4 = np.load(data_mri_pad_4["test_x_fn"] + '.npy')
    Y_test_mri_pad_4 = np.load(data_mri_pad_4["test_y_fn"] + '.npy')

    Y_test_mri_pad_4 = to_categorical(Y_test_mri_pad_4)
    Y_train_mri_pad_4 = to_categorical(Y_train_mri_pad_4, num_classes=nlabels_mri_pad_4)
    Y_validate_mri_pad_4 = to_categorical(Y_validate_mri_pad_4, num_classes=nlabels_mri_pad_4)

    # if you change the number of times you downsample with max_pool,
    # then you need to rerun prepare_data() with pad_base=<number of downsample nodes>
    model_saving_path = SAVED_MODELS_DIR + "/model_of_unet_at_mri.hdf5"

    # Define the architecture of neural network
    IN = Input(shape=(data_mri_pad_4['image_dim'][1], data_mri_pad_4['image_dim'][2], 1))

    BN1 = BatchNormalization()(IN)

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(BN1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

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
    # up6 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc5 = Concatenate(axis=3)([up5, conv4])
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conc5)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up6 = UpSampling2D(size=(2, 2))(conv6)
    # up6 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc6 = Concatenate(axis=3)([up6, conv3])
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conc6)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up7 = UpSampling2D(size=(2, 2))(conv7)
    # up7 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv7)
    conc7 = Concatenate(axis=3)([up7, conv2])
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conc7)  # (up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up8 = UpSampling2D(size=(2, 2))(conv8)
    # up8 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv8)
    conc8 = Concatenate(axis=3)([up8, conv1])
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conc8)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(nlabels_mri_pad_4, 1, 1, activation='softmax')(conv9)

    model = keras.models.Model(input=[IN], output=conv10)

    print(model.summary())

    # set compiler
    ada = keras.optimizers.Adam(0.0001)
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['acc'])
    # fit model
    history = model.fit([X_train_mri_pad_4],
                        Y_train_mri_pad_4,
                        validation_data=([X_validate_mri_pad_4], Y_validate_mri_pad_4),
                        epochs=3,
                        callbacks=[TensorBoard(log_dir=LOGS_DIR)])
    # save model
    model.save(model_saving_path)
    # test model
    test_score = model.evaluate(X_test_mri_pad_4, Y_test_mri_pad_4)
    print("Test :", test_score)


if __name__ == "__main__":
    subprocess.call("mkdir -p " + SAVED_MODELS_DIR, shell=True)
    main()
