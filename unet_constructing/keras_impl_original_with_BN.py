import os
import sys

from keras.callbacks import TensorBoard

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pickle
from config_and_utils import GlobalVar
from data_processing.prepare_datasets import prepare_dataset
from module_minc_keras.minc_keras import *

PROJECT_DIR = GlobalVar.PROJECT_PATH
DATASET_DIR = GlobalVar.DATASET_PATH
OUTPUT_DIR = GlobalVar.OUTPUT_PATH + "/keras_impl_original_with_BN"
LOGS_DIR = OUTPUT_DIR + "/logs"
SAVED_MODELS_DIR = OUTPUT_DIR + "/saved_models"
DATASET_NAME = "mri"
EPOCH_NUM = 3


def main():
    print("🚩Use", DATASET_NAME, "dataset to train", EPOCH_NUM, "epoches")
    serialized_file = GlobalVar.DATASET_PATH + "/" + DATASET_NAME + "_pad_4_results/serialized_dataset_object"
    if os.path.exists(serialized_file):
        with open(serialized_file, "rb") as f:
            print("🚩Done deserializing file:", serialized_file)
            [images_mri_pad_4, data_mri_pad_4] = pickle.load(f)
    else:
        [images_mri_pad_4, data_mri_pad_4] = prepare_dataset(dataset_name=DATASET_NAME)

    # Load data
    Y_validate_mri_pad_4 = np.load(data_mri_pad_4["validate_y_fn"] + '.npy')
    nlabels_mri_pad_4 = len(np.unique(Y_validate_mri_pad_4))  # class num

    X_train_mri_pad_4 = np.load(data_mri_pad_4["train_x_fn"] + '.npy')
    Y_train_mri_pad_4 = np.load(data_mri_pad_4["train_y_fn"] + '.npy')
    X_validate_mri_pad_4 = np.load(data_mri_pad_4["validate_x_fn"] + '.npy')

    X_test_mri_pad_4 = np.load(data_mri_pad_4["test_x_fn"] + '.npy')
    Y_test_mri_pad_4 = np.load(data_mri_pad_4["test_y_fn"] + '.npy')

    print("Y_test_mri_pad_4 shape", Y_test_mri_pad_4.shape)
    Y_test_mri_pad_4 = to_categorical(Y_test_mri_pad_4)
    Y_train_mri_pad_4 = to_categorical(Y_train_mri_pad_4, num_classes=nlabels_mri_pad_4)
    Y_validate_mri_pad_4 = to_categorical(Y_validate_mri_pad_4, num_classes=nlabels_mri_pad_4)

    # if you change the number of times you downsample with max_pool,
    # then you need to rerun prepare_data() with pad_base=<number of downsample nodes>
    model_saving_path = SAVED_MODELS_DIR + "/unet_model_on_" + DATASET_NAME + ".hdf5"

    # Define the architecture of neural network
    IN = Input(shape=(data_mri_pad_4['image_dim'][1], data_mri_pad_4['image_dim'][2], 1))
    # print("🚩IN shape", IN.shape)

    BN1 = BatchNormalization()(IN)
    # print("🚩BN1 shape", BN1.shape)
    # 32, 3, 3 are nb_filters, nb_row, nb_col. 3, 3 can also be write as kernel_size=3. strides default to (1,1).
    conv1 = Convolution2D(filters=64, kernel_size=3, activation='relu', border_mode='same')(BN1)
    # print("🚩️conv1 shape", conv1.shape)
    BN1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(BN1)
    # print("🚩️conv1 shape", conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=None)(conv1)  # strides is None, it will default to pool_size
    # print("🚩pool1 shape", pool1.shape)

    BN2 = BatchNormalization()(pool1)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(BN2)
    # print("🚩️conv2 shape", conv2.shape)
    BN2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(BN2)
    # print("🚩️conv2 shape", conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print("🚩pool2 shape", pool2.shape)

    BN3 = BatchNormalization()(pool2)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(BN3)
    # print("🚩conv3 shape", conv3.shape)
    BN3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(BN3)
    # print("🚩️conv3 shape", conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # print("🚩pool3 shape", pool3.shape)

    BN4 = BatchNormalization()(pool3)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(BN4)
    # print("🚩️conv4 shape", conv4.shape)
    BN4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(BN4)
    # print("🚩️conv4 shape", conv4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # print("🚩pool4 shape", pool4.shape)

    BN5 = BatchNormalization()(pool4)
    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(BN5)
    # print("🚩️conv5 shape", conv5.shape)
    BN5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(BN5)
    # print("🚩️conv5 shape", conv5.shape)

    up5 = UpSampling2D(size=(2, 2))(conv5)
    # print("🚩up5 shape", up5.shape)
    # up6 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc5 = Concatenate(axis=3)([up5, conv4])
    # print("🚩conc5 shape", conc5.shape)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conc5)
    # print("🚩️conv6 shape", conv6.shape)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)
    # print("🚩️conv6 shape", conv6.shape)

    up6 = UpSampling2D(size=(2, 2))(conv6)
    # print("🚩up6 shape", up6.shape)
    # up6 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc6 = Concatenate(axis=3)([up6, conv3])
    # print("🚩️conc6 shape", conc6.shape)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conc6)
    # print("🚩️conv7 shape", conv7.shape)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)
    # print("🚩️conv7 shape", conv7.shape)

    up7 = UpSampling2D(size=(2, 2))(conv7)
    # print("🚩up7 shape", up7.shape)
    # up7 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv7)
    conc7 = Concatenate(axis=3)([up7, conv2])
    # print("🚩️conc7 shape", conc7.shape)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conc7)  # (up8)
    # print("🚩️conv8 shape", conv8.shape)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)
    # print("🚩️conv8 shape", conv8.shape)

    up8 = UpSampling2D(size=(2, 2))(conv8)
    # print("🚩up8 shape", up8.shape)
    # up8 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv8)
    conc8 = Concatenate(axis=3)([up8, conv1])
    # print("🚩️conc8 shape", conc8.shape)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conc8)
    # print("🚩️conv9 shape", conv9.shape)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)
    # print("🚩️conv9 shape", conv9.shape)

    conv10 = Convolution2D(nlabels_mri_pad_4, 1, 1, activation='softmax')(conv9)
    # print("🚩️conv10 shape", conv10.shape)

    model = keras.models.Model(input=[IN], output=conv10)

    print(model.summary())

    # set the optimizer
    ada = keras.optimizers.Adam(0.0001)
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['acc'])
    # fit model
    history = model.fit([X_train_mri_pad_4],
                        Y_train_mri_pad_4,
                        validation_data=([X_validate_mri_pad_4], Y_validate_mri_pad_4),
                        epochs=EPOCH_NUM,
                        callbacks=[TensorBoard(log_dir=LOGS_DIR)])
    # save model
    model.save(model_saving_path)
    # test model
    test_score = model.evaluate(X_test_mri_pad_4, Y_test_mri_pad_4)
    print("🚩Test :", test_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset name
    parser.add_argument(
        '--dataset_name', type=str, default=DATASET_NAME,
        help='name of dataset you want to use')
    # epoch num
    parser.add_argument(
        '--epoch_num', type=int, default=EPOCH_NUM,
        help='epoch num')
    FLAGS, _ = parser.parse_known_args()

    DATASET_NAME = FLAGS.dataset_name
    EPOCH_NUM = FLAGS.epoch_num

    if not os.path.exists(SAVED_MODELS_DIR):
        os.makedirs(SAVED_MODELS_DIR)

    main()
