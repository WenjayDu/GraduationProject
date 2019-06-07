import os
import sys

import keras

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from config_and_utils import GlobalVar, logging, get_sorted_files, get_dir_containing_file
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
tf.flags.DEFINE_string(name="img_path", default=IMG_PATH, help="path of img to predict")
tf.flags.DEFINE_list(name="img_size", default=IMG_SIZE, help="size of output img")


def safe_load_model(model_path):
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        return model
    else:
        logging.error("❌️Error: cannot find model, please check " + model_path)
        sys.exit(1)


def predict_with_keras_model(model_path, img_path, input_shape, prediction_save_dir=None):
    if type(input_shape) == tuple:
        input_shape = list(input_shape)
    if input_shape.__len__() == 2:
        input_shape = input_shape + [3]
    if prediction_save_dir is None:
        prediction_save_dir = get_dir_containing_file(model_path) + "/predictions"
    if not os.path.exists(prediction_save_dir):
        os.makedirs(prediction_save_dir)
    logging.info("🚩️loading the model " + model_path)
    model = safe_load_model(model_path)
    # prepare image
    if os.path.isdir(img_path):
        image_list = get_sorted_files(img_path, "png")
        logging.info("🚩️" + str(len(image_list)) + " images to be predicted, will be saved to " + prediction_save_dir)
        for index, image_path in enumerate(image_list):
            original_img = image.load_img(image_path, target_size=input_shape, color_mode="grayscale")
            original_img = image.img_to_array(original_img)
            img = np.expand_dims(original_img, axis=0)
            prediction = model.predict(img)
            prediction = prediction.reshape(input_shape)
            image.save_img(os.path.join(prediction_save_dir, '%d.png' % index), prediction * 255)
        logging.info("🚩️Predictions are saved, now converting them to 'hot' color map")
        to_hot_cmap(prediction_save_dir)
        logging.warning('❗️Done prediction')
    else:
        img = image.load_img(img_path, target_size=input_shape, color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # get prediction
        prediction = model.predict(img_array, batch_size=1)
        to_hot_cmap(img=prediction, save_path=prediction_save_dir + "/prediction.png")


def predict_with_tf_model(ckpt_path, structure, img_path, prediction_save_dir=None):
    unet = choose_unet(structure_name=structure)
    tf.reset_default_graph()
    unet.build_up_unet(batch_size=1)
    logging.info("🚩️start predicting...")
    if prediction_save_dir is None:
        prediction_save_dir = get_dir_containing_file(ckpt_path) + "/predictions"
    if not os.path.exists(prediction_save_dir):
        os.makedirs(prediction_save_dir)
    predict(unet, ckpt_path=ckpt_path, original_img_path=img_path, prediction_save_dir=prediction_save_dir)


def predict_with_pb_model(pb_file_path, img_path, input_shape=[144, 112, 3]):
    if type(input_shape) is not list:
        sys.exit("Error: input_shape must be a list")
    img = image.load_img(img_path, target_size=input_shape, color_mode='grayscale')
    img = image.img_to_array(img)
    img = img.reshape([1] + input_shape[0: 2] + [1])
    with tf.Graph().as_default() as graph:
        sess = tf.Session()

        # restore the model
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_file_path, 'rb') as i_file:
            graph_def.ParseFromString(i_file.read())
        tf.import_graph_def(graph_def)

        # obtain input & output nodes and then test the model
        net_input = graph.get_tensor_by_name('import/net_input:0')
        net_output = graph.get_tensor_by_name('import/net_output:0')
        result = sess.run(net_output, feed_dict={net_input: img})
        return result


if __name__ == "__main__":
    if not os.path.exists(KERAS_PREDICTION_SAVE_DIR):
        os.makedirs(KERAS_PREDICTION_SAVE_DIR)
        logging.warning("❗️Creating non-existent dir" + KERAS_PREDICTION_SAVE_DIR)
    if not os.path.exists(TF_PREDICTION_SAVE_DIR):
        os.makedirs(TF_PREDICTION_SAVE_DIR)
        logging.warning("❗️Creating non-existent dir" + TF_PREDICTION_SAVE_DIR)
    predict_with_keras_model(model_path=FLAGS.model_path, img_path=FLAGS.img_path, input_shape=FLAGS.img_size)
    predict_with_tf_model(ckpt_path=FLAGS.ckpt_path, structure="original", img_path=FLAGS.img_path)
