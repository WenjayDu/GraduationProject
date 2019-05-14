import os
import sys

import keras

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from config_and_utils import GlobalVar, logging, get_sorted_files
from unet_constructing.tf_impl.utils import predict
from unet_constructing.tf_impl_run import choose_unet
from data_processing.img_utils import to_hot_cmap

PROJECT_PATH = GlobalVar.PROJECT_PATH
DATASET_PATH = GlobalVar.DATASET_PATH
OUTPUT_PATH = GlobalVar.OUTPUT_PATH
IMG_SIZE = [144, 112, 3]
IMG_PATH = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_anat_rsl/53.png"
LABEL_PATH = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_variant-seg_rsl/53.png"

KERAS_MODEL_PATH = OUTPUT_PATH + "/keras_impl_original/unet_model_on_mri.hdf5"
KERAS_PREDICTION_SAVE_DIR = OUTPUT_PATH + "/prediction_of_keras_model"

# for model implemented with tf
PREDICT_IMG_DIR = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_anat_rsl"
TF_PREDICTION_SAVE_DIR = OUTPUT_PATH + "/prediction_of_tf_model"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(name="ckpt_path", default=DATASET_PATH + '/mri_pad_4/models/tf_impl/original/unet_model.ckpt',
                       help="path of checkpoint")


def safe_load_model(model_path):
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        return model
    else:
        logging.error("❌️Error: cannot find model, please check " + model_path)
        sys.exit(1)


def predict_with_keras_model(model_path, img_path, img_size, prediction_save_dir=None):
    if type(img_size) == tuple:
        img_size = list(img_size)
    logging.info("🚩️loading the model " + model_path)
    model = safe_load_model(model_path)
    # prepare image
    img = image.load_img(img_path, target_size=img_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # get prediction
    predict_result = model.predict(img_array)
    to_hot_cmap(img=predict_result, save_path=prediction_save_dir + "/prediction.png")


def predict_with_tf_model(ckpt_path, structure, prediction_save_dir=None):
    unet = choose_unet(structure_name=structure)
    tf.reset_default_graph()
    unet.build_up_unet(batch_size=1)
    logging.info("🚩️start predicting...")
    if prediction_save_dir is None:
        prediction_save_dir = ckpt_path.split('/unet_model.ckpt')[0] + "/predictions"
    predict(unet, ckpt_path=ckpt_path, prediction_save_dir=prediction_save_dir)


if __name__ == "__main__":
    if not os.path.exists(KERAS_PREDICTION_SAVE_DIR):
        os.makedirs(KERAS_PREDICTION_SAVE_DIR)
        logging.warning("❗️Creating non-existent dir" + KERAS_PREDICTION_SAVE_DIR)
    if not os.path.exists(TF_PREDICTION_SAVE_DIR):
        os.makedirs(TF_PREDICTION_SAVE_DIR)
        logging.warning("❗️Creating non-existent dir" + TF_PREDICTION_SAVE_DIR)
    predict_with_keras_model(model_path=FLAGS.model_path)
    predict_with_tf_model(ckpt_path=FLAGS.ckpt_path, structure="original", )
