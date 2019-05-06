import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

from config_and_utils import GlobalVar
from module_minc_keras.utils import normalize
from unet_constructing.construct_unet_with_tf import UNet

PROJECT_PATH = GlobalVar.PROJECT_PATH
DATASET_PATH = GlobalVar.DATASET_PATH
OUTPUT_PATH = GlobalVar.OUTPUT_PATH
IMG_SIZE = (144, 112, 3)
IMG_PATH = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_anat_rsl/53.png"
LABEL_PATH = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_variant-seg_rsl/53.png"

MODEL_PATH = PROJECT_PATH + "/unet_at_mri/trained_models/keras_implementation/model_of_unet_at_mri.hdf5"
KERAS_PREDICTION_SAVED_DIR = OUTPUT_PATH + "/keras_model"

# for model implemented with tf
CHECKPOINT_PATH = PROJECT_PATH + "/unet_at_mri/trained_models/tf_implementation/model.ckpt"
PREDICT_IMG_DIR = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_anat_rsl"
TF_PREDICTION_SAVED_DIR = OUTPUT_PATH + "/tf_model"
PREDICT_BATCH_SIZE = 1


def to_hot_cmap(img, save_path, argmax_axis=2):
    """
    convert an img to a hot colormap
    :param img: path of an img to be converted or an numpy array
    :param save_path: save path
    :param argmax_axis:
    :return:
    """
    if type(img) == str:
        img = image.load_img(img)
        img = image.img_to_array(img)
    img = np.argmax(img, axis=argmax_axis)
    if len(list(img.shape)) == 3:
        img = img.reshape(list(img.shape[1:3]))
    img = normalize(img)
    plt.imshow(img, cmap="hot")
    plt.axis('off')
    plt.savefig(save_path)
    print("🚩Saved prediction to", save_path)
    del img


def safe_load_model(model_path):
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        return model
    else:
        print("❗️Error: cannot find model, please check", model_path)


def predict_with_keras_model(model_path=MODEL_PATH, img_path=IMG_PATH, img_size=IMG_SIZE):
    print("loading the model " + model_path + " ......")
    model = safe_load_model(model_path)
    # prepare image
    img = image.load_img(img_path, target_size=img_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # get prediction
    predict_result = model.predict(img_array)
    to_hot_cmap(predict_result, argmax_axis=3, save_path=KERAS_PREDICTION_SAVED_DIR + "/prediction.png")


def predict_with_tf_model():
    unet = UNet()
    tf.reset_default_graph()
    unet.build_up_unet(PREDICT_BATCH_SIZE)
    print("❗️start predicting...")
    unet.predict(ckpt_path=CHECKPOINT_PATH)


if __name__ == "__main__":
    if not os.path.exists(KERAS_PREDICTION_SAVED_DIR):
        os.makedirs(KERAS_PREDICTION_SAVED_DIR)
        print("Creating non-existent dir", KERAS_PREDICTION_SAVED_DIR)
    if not os.path.exists(TF_PREDICTION_SAVED_DIR):
        os.makedirs(TF_PREDICTION_SAVED_DIR)
        print("Creating non-existent dir", TF_PREDICTION_SAVED_DIR)
    predict_with_keras_model()
    predict_with_tf_model()
