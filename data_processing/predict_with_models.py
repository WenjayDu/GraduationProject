import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

from config_and_utils import GlobalVar, get_sorted_files
from module_minc_keras.utils import normalize
from unet_constructing.tf_impl_original import UNet

PROJECT_PATH = GlobalVar.PROJECT_PATH
DATASET_PATH = GlobalVar.DATASET_PATH
OUTPUT_PATH = GlobalVar.OUTPUT_PATH
IMG_SIZE = (144, 112, 3)
IMG_PATH = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_anat_rsl/53.png"
LABEL_PATH = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_variant-seg_rsl/53.png"

MODEL_PATH = OUTPUT_PATH + "/keras_implementation/model.hdf5"
KERAS_PREDICTION_SAVED_DIR = OUTPUT_PATH + "/prediction_of_keras_model"

# for model implemented with tf
CHECKPOINT_PATH = OUTPUT_PATH + "/tf_implementation/model.ckpt"
PREDICT_IMG_DIR = DATASET_PATH + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_anat_rsl"
TF_PREDICTION_SAVED_DIR = OUTPUT_PATH + "/prediction_of_tf_model"
PREDICT_BATCH_SIZE = 1


def show_argmax_result(img, argmax_axis=2):
    """
    convert an img to a hot colormap
    :param img: path of an img to be converted or an numpy array
    :param save_path: save path
    :param argmax_axis:
    :return:
    """
    if type(img) == str:
        img = image.load_img(img)
        img.show()
        img = image.img_to_array(img)
    print("image.shape", img.shape)
    img = np.argmax(img, axis=argmax_axis)
    img = np.expand_dims(img, axis=argmax_axis)
    img = image.array_to_img(img)
    img.show()


def to_hot_cmap(img, save_path=None, argmax_axis=2):
    """
    convert an img to a hot colormap
    :param img: path of an img or a dir containing png images to be converted or an numpy array
    :param save_path: save path
    :param argmax_axis:
    :return:
    """
    if type(img) == str:
        if os.path.isfile(img):
            img = image.load_img(img)
            img = image.img_to_array(img)
        elif os.path.isdir(img):
            print("🚩Converting all .png files in", img)
            image_list = get_sorted_files(img, "png")
            example = image.load_img(image_list[0])
            example = image.img_to_array(example)
            fig = plt.figure(frameon=False,
                             figsize=(example.shape[1] / 500, example.shape[0] / 500),  # figsize(width, height)
                             dpi=500)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if not os.path.exists(img + "/to_hot_cmap"):
                os.makedirs(img + "/to_hot_cmap")
            for i in image_list:
                save_path = img + "/to_hot_cmap/" + i.split('/')[-1]
                img_arr = image.load_img(i)
                img_arr = image.img_to_array(img_arr)
                img_arr = np.argmax(img_arr, axis=argmax_axis)
                img_arr = normalize(img_arr)
                ax.imshow(img_arr, cmap="hot")
                fig.savefig(save_path)
                ax.clear()
            print("🚩Done converting, converted images are saved to", img + "/to_hot_cmap")
            plt.close()
            return 0
    img = np.argmax(img, axis=argmax_axis)
    if len(list(img.shape)) == 3:  # this means the img is a tensor, in which the 1st dim is the num of samples
        img = img.reshape(list(img.shape[1:3]))
    img = normalize(img)

    fig = plt.figure(frameon=False,
                     figsize=(img.shape[1] / 500, img.shape[0] / 500),  # figsize(width, height)
                     dpi=500)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap="hot")
    fig.savefig(save_path)
    print("🚩Saved prediction to", save_path)
    plt.close()
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
