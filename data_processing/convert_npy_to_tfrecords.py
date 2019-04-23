import os
import numpy as np
import tensorflow as tf
import project_config as config
from keras_preprocessing import image

PROJECT_DIR = config.get_project_path()
DATASET_DIR = PROJECT_DIR + "/datasets/mri_pad_4_results/data/"
TARGET_PATH_DIR = PROJECT_DIR + "/datasets/tfrecords/"


def convert_to_tfrecords(data_npy, label_npy, target_path):
    with tf.python_io.TFRecordWriter(target_path) as writer:
        for i in range(len(data_npy)):
            features = tf.train.Features(
                feature={
                    # convert to string, which is byte data type
                    "data": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data_npy[i].tostring()])),
                    "label": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[label_npy[i].tostring()]))
                }
            )
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def parse_function(example_proto):
    features = {"data": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], np.float16)
    label = tf.decode_raw(parsed_features['label'], np.float16)
    return data, label


def read_from_tfrecords(srcfile):
    """
    this function is created to help me test reading content from written tfrecords files.
    :param srcfile:
    :return:
    """
    sess = tf.Session()
    dataset = tf.data.TFRecordDataset(srcfile)  # load tfrecord file
    dataset = dataset.map(parse_function)  # parse data into tensor
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    for i in range(25):
        try:
            data, label = sess.run(next_data)
            if i == 24:
                data = data.reshape(144, 112, 1)
                label = label.reshape(144, 112, 1)
                img = image.array_to_img(data)
                lab = image.array_to_img(label)
                img.show()
                lab.show()
        except tf.errors.OutOfRangeError:
            pass
        finally:
            sess.close()


def main():
    if not os.path.exists(TARGET_PATH_DIR):
        print("mkdir " + TARGET_PATH_DIR)
        os.makedirs(TARGET_PATH_DIR)
    if os.path.exists(DATASET_DIR + "train_x.npy"):
        train_data = np.load(DATASET_DIR + "train_x.npy")
        train_label = np.load(DATASET_DIR + "train_y.npy")
        convert_to_tfrecords(train_data, train_label, TARGET_PATH_DIR + "train.tfrecords")
        print("converting the train dataset done!")

        test_data = np.load(DATASET_DIR + "test_x.npy")
        test_label = np.load(DATASET_DIR + "test_y.npy")
        convert_to_tfrecords(test_data, test_label, TARGET_PATH_DIR + "test.tfrecords")
        print("converting the test dataset done!")

        validate_data = np.load(DATASET_DIR + "validate_x.npy")
        validate_label = np.load(DATASET_DIR + "validate_y.npy")
        convert_to_tfrecords(validate_data, validate_label, TARGET_PATH_DIR + "validate.tfrecords")
        print("converting the validate dataset done!")
    else:
        print("default .npy files do not exists, please run ./prepare_datasets.py to generate them")
        return 1


if __name__ == "__main__":
    main()
