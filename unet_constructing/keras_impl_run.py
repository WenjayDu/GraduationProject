import os
import sys

import tensorflow as tf
from keras.callbacks import TensorBoard

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pickle
from config_and_utils import GlobalVar, logging
from module_minc_keras.minc_keras import *
from unet_constructing.keras_impl import (original, original_with_BN, smaller, smaller_with_BN)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(name='dataset_dir_path', default=GlobalVar.DATASET_PATH + "/mri_pad_4",
                       help='path of the dataset dir you want to use')
tf.flags.DEFINE_string(name='structure', default="original",
                       help="structure of U-Net you want to use, like original, smaller")
tf.flags.DEFINE_integer(name='epoch_num', default=3, help='epoch num')

ROOT_OUTPUT_DIR = FLAGS.dataset_dir_path + '/models/keras_impl/'

REAL_OUTPUT_DIR = ROOT_OUTPUT_DIR + '/' + FLAGS.structure

LOGS_DIR = REAL_OUTPUT_DIR + "/logs"
MODEL_SAVE_DIR = REAL_OUTPUT_DIR + "/saved_model"


def choose_unet(structure_name=FLAGS.structure):
    switcher = {
        "original": original.UNet(),
        "original_with_BN": original_with_BN.UNet(),
        "smaller": smaller.UNet(),
        "smaller_with_BN": smaller_with_BN.UNet()
    }
    return switcher.get(structure_name)


def main():
    logging.info("🚩Use " + FLAGS.dataset_dir_path + " dataset to train " + str(FLAGS.epoch_num) + " epoches")
    serialized_file = FLAGS.dataset_dir_path + "/serialized_dataset_object"
    if os.path.exists(serialized_file):
        with open(serialized_file, "rb") as f:
            logging.info("🚩Done deserializing file: " + serialized_file)
            [_, data] = pickle.load(f)
    else:
        logging.error("❌" + FLAGS.dataset_dir_path +
                      " cannot be recognized, please use data_processing.prepare_datasets to generate.")

    # Load data
    Y_validate_mri_pad_4 = np.load(data["validate_y_fn"] + '.npy')
    nlabels_mri_pad_4 = len(np.unique(Y_validate_mri_pad_4))  # class num

    X_train_mri_pad_4 = np.load(data["train_x_fn"] + '.npy')
    Y_train_mri_pad_4 = np.load(data["train_y_fn"] + '.npy')
    X_validate_mri_pad_4 = np.load(data["validate_x_fn"] + '.npy')

    X_test_mri_pad_4 = np.load(data["test_x_fn"] + '.npy')
    Y_test_mri_pad_4 = np.load(data["test_y_fn"] + '.npy')

    Y_test_mri_pad_4 = to_categorical(Y_test_mri_pad_4)
    Y_train_mri_pad_4 = to_categorical(Y_train_mri_pad_4, num_classes=nlabels_mri_pad_4)
    Y_validate_mri_pad_4 = to_categorical(Y_validate_mri_pad_4, num_classes=nlabels_mri_pad_4)

    # construct U-Net
    unet = choose_unet(structure_name=FLAGS.structure)
    unet_model = unet.build_up_unet(data=data, class_num=nlabels_mri_pad_4)

    # set the optimizer
    ada = keras.optimizers.Adam(0.0001)
    # compile the model
    unet_model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['acc'])
    # fit model
    history = unet_model.fit([X_train_mri_pad_4],
                             Y_train_mri_pad_4,
                             validation_data=([X_validate_mri_pad_4], Y_validate_mri_pad_4),
                             epochs=FLAGS.epoch_num,
                             callbacks=[TensorBoard(log_dir=LOGS_DIR)])
    # save model
    model_name = "model_of_" + FLAGS.structure + ".hdf5"
    model_save_path = MODEL_SAVE_DIR + '/' + model_name
    unet_model.save(model_save_path)
    # test model
    test_score = unet_model.evaluate(X_test_mri_pad_4, Y_test_mri_pad_4)
    logging.info("🚩Test : " + str(test_score))
    logging.info("🚩model has been saved as " + model_save_path)


if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    main()
