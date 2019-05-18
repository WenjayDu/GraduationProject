import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from module_minc_keras.minc_keras import *
from data_processing import convert_npy_to_tfrecords
from unet_constructing.tf_impl.utils import *
from unet_constructing.tf_impl import (original, original_with_BN)

FLAGS = tf.flags.FLAGS


def choose_unet(structure_name=FLAGS.structure):
    switcher = {
        "original": original.UNet(),
        "original_with_BN": original_with_BN.UNet()
    }
    return switcher.get(structure_name)


def main():
    unet = choose_unet(structure_name=FLAGS.structure)
    if FLAGS.to_train == "yes":
        unet.build_up_unet(batch_size=FLAGS.train_batch_size)
        logging.info("🚩️start training...")
        train(unet)
    if FLAGS.to_validate == "yes":
        tf.reset_default_graph()
        unet.build_up_unet(batch_size=FLAGS.validation_batch_size)
        logging.info("🚩️start validating...")
        validate(unet)
    if FLAGS.to_test == "yes":
        tf.reset_default_graph()
        unet.build_up_unet(batch_size=FLAGS.test_batch_size)
        logging.info("🚩️start testing...")
        test(unet)
    if FLAGS.to_predict == "yes":
        tf.reset_default_graph()
        unet.build_up_unet(batch_size=1)
        logging.info("🚩️start predicting...")
        predict(unet)


if __name__ == '__main__':
    if not os.path.exists(FLAGS.dataset_dir_path):
        logging.error("❌dir %s does not exist, please correct the path of dataset dir used to train",
                      FLAGS.dataset_dir_path)
        sys.exit(1)

    if not os.path.exists(TEST_SAVE_DIR):
        os.system("mkdir -p " + TEST_SAVE_DIR)
    if not os.path.exists(PREDICTION_SAVE_DIR):
        os.system("mkdir -p " + PREDICTION_SAVE_DIR)

    if not os.path.exists(FLAGS.dataset_dir_path + "/tfrecords/train.tfrecords"):
        logging.warning("❗️.tfrecords files used to train do not exist, generating now...")
        convert_npy_to_tfrecords.convert_whole_dataset(FLAGS.dataset_dir_path)

    main()
