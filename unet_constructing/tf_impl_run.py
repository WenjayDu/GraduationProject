import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from config_and_utils import GlobalVar, logging, get_sorted_files
from module_minc_keras.minc_keras import *
from data_processing import convert_npy_to_tfrecords, predict_with_models
from unet_constructing.tf_impl.utils import *
from unet_constructing.tf_impl import (original, original_with_BN, smaller, smaller_with_BN)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(name='dataset_dir_path', default=GlobalVar.DATASET_PATH + "/mri_pad_4",
                       help='path of the dataset dir you want to use')
tf.flags.DEFINE_string(name='structure', default="original",
                       help="structure of U-Net you want to use, like original, smaller")
tf.flags.DEFINE_integer(name='epoch_num', default=3, help='epoch num')

tf.flags.DEFINE_integer('train_batch_size', default=8, help='train batch size')
tf.flags.DEFINE_integer('validation_batch_size', default=8, help='validation batch size')
tf.flags.DEFINE_integer('test_batch_size', default=8, help='test batch size')

tf.flags.DEFINE_string('to_train', default="yes", help='whether to train, yes/no')
tf.flags.DEFINE_string('to_validate', default="yes", help='whether to validate, yes/on')
tf.flags.DEFINE_string('to_test', default="yes", help='whether to test, yes/no')
tf.flags.DEFINE_string('to_predict', default="yes", help='whether to predict, yes/no')
tf.flags.DEFINE_list('input_shape', default=[144, 112, 1], help='shape of input data')
tf.flags.DEFINE_list('output_shape', default=[144, 112, 3], help='shape of input data')

ROOT_OUTPUT_DIR = FLAGS.dataset_dir_path + '/models/tf_impl/'
REAL_OUTPUT_DIR = ROOT_OUTPUT_DIR + '/' + FLAGS.structure

LOG_DIR = REAL_OUTPUT_DIR + "/logs"
MODEL_SAVE_DIR = REAL_OUTPUT_DIR + "/saved_model"
CKPT_PATH = MODEL_SAVE_DIR + "/unet_model.ckpt"

DATASET_DIR = GlobalVar.DATASET_PATH

# the path of dir stroing imgs for predicting operation
ORIGINAL_IMG_DIR = DATASET_DIR + "/examples/extracted_images/sub-00031_task-01_ses-01_T1w_anat_rsl"

# the path of dir for saving imgs output from predicting operation
PREDICTION_SAVE_DIR = REAL_OUTPUT_DIR + "/predictions"
TEST_SAVE_DIR = REAL_OUTPUT_DIR + "/test_saved"

[INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, INPUT_IMG_CHANNEL] = FLAGS.input_shape
[OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH, OUTPUT_IMG_CHANNEL] = FLAGS.output_shape

TRAIN_BATCH_SIZE = FLAGS.train_batch_size
VALIDATION_BATCH_SIZE = FLAGS.validation_batch_size
TEST_BATCH_SIZE = FLAGS.test_batch_size

TRAIN_SET_PATH = FLAGS.dataset_dir_path + "/tfrecords/train.tfrecords"
VALIDATE_SET_PATH = FLAGS.dataset_dir_path + "/tfrecords/validate.tfrecords"
TEST_SET_PATH = FLAGS.dataset_dir_path + "/tfrecords/test.tfrecords"


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


def choose_unet(structure_name=FLAGS.structure):
    switcher = {
        "original": original.UNet(),
        "original_with_BN": original_with_BN.UNet(),
        "smaller": smaller.UNet(),
        "smaller_with_BN": smaller_with_BN.UNet()
    }
    return switcher.get(structure_name)


def train(unet):
    train_image_filename_queue = tf.train.string_input_producer(
        string_tensor=tf.train.match_filenames_once(TRAIN_SET_PATH), num_epochs=FLAGS.epoch_num, shuffle=True)
    train_images, train_labels = read_image_batch(train_image_filename_queue, TRAIN_BATCH_SIZE)
    tf.summary.scalar("loss", unet.loss_mean)
    tf.summary.scalar('accuracy', unet.accuracy)
    merged_summary = tf.summary.merge_all()
    all_parameters_saver = tf.train.Saver()
    divisor = step_size_of_showing_result(TRAIN_BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        tf.summary.FileWriter(MODEL_SAVE_DIR, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            epoch = 0
            while not coord.should_stop():
                # run training
                example, label = sess.run([train_images, train_labels])  # get image and label，type is numpy.ndarry
                label = label.reshape(TRAIN_BATCH_SIZE, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH)

                lo, acc, summary_str = sess.run(
                    [unet.loss_mean, unet.accuracy, merged_summary],
                    feed_dict={
                        unet.input_image: example,
                        unet.input_label: label,
                        unet.keep_prob: 1.0,
                        unet.lamb: 0.004,
                        unet.is_training: True}
                )
                summary_writer.add_summary(summary_str, epoch)
                sess.run([unet.train_step], feed_dict={unet.input_image: example, unet.input_label: label,
                                                       unet.keep_prob: 1.0, unet.lamb: 0.004, unet.is_training: True}
                         )
                epoch += 1
                if epoch % divisor == 0:
                    print('num %d , loss: %.6f , accuracy: %.6f' % (epoch * TRAIN_BATCH_SIZE, lo, acc))
        except tf.errors.OutOfRangeError:
            logging.info('❗️Done training -- epoch limit reached')
        finally:
            all_parameters_saver.save(sess=sess, save_path=CKPT_PATH)
            coord.request_stop()
        coord.join(threads)
    print('❗️Done training. Total: num %d , loss: %.6f , accuracy: %.6f\n'
          % (epoch * TRAIN_BATCH_SIZE, lo, acc))


def validate(unet):
    validation_image_filename_queue = tf.train.string_input_producer(
        string_tensor=tf.train.match_filenames_once(VALIDATE_SET_PATH), num_epochs=1, shuffle=True)
    validation_images, validation_labels = read_image_batch(validation_image_filename_queue,
                                                            VALIDATION_BATCH_SIZE)
    # tf.summary.scalar("loss", self.loss_mean)
    # tf.summary.scalar('accuracy', self.accuracy)
    # merged_summary = tf.summary.merge_all()
    all_parameters_saver = tf.train.Saver()
    divisor = step_size_of_showing_result(VALIDATION_BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
        # tf.summary.FileWriter(SAVED_MODELS_DIR, sess.graph)
        all_parameters_saver.restore(sess=sess, save_path=CKPT_PATH)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            epoch = 0
            while not coord.should_stop():
                example, label = sess.run([validation_images, validation_labels])
                label = label.reshape(VALIDATION_BATCH_SIZE, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH)
                lo, acc = sess.run(
                    [unet.loss_mean, unet.accuracy],
                    feed_dict={
                        unet.input_image: example,
                        unet.input_label: label,
                        unet.keep_prob: 1.0,
                        unet.lamb: 0.004,
                        unet.is_training: False}
                )
                # summary_writer.add_summary(summary_str, epoch)
                epoch += 1
                if epoch % divisor == 0:
                    print('num %d , loss: %.6f , accuracy: %.6f' % (epoch * VALIDATION_BATCH_SIZE, lo, acc))
        except tf.errors.OutOfRangeError:
            print('❗️Done validating -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
    print('❗️Done validating. Total: num %d , loss: %.6f , accuracy: %.6f\n'
          % (epoch * VALIDATION_BATCH_SIZE, lo, acc))


def test(unet):
    import cv2
    test_image_filename_queue = tf.train.string_input_producer(
        string_tensor=tf.train.match_filenames_once(TEST_SET_PATH), num_epochs=1, shuffle=True)
    test_images, test_labels = read_image_batch(test_image_filename_queue, TEST_BATCH_SIZE)
    # tf.summary.scalar("loss", self.loss_mean)
    # tf.summary.scalar('accuracy', self.accuracy)
    # merged_summary = tf.summary.merge_all()
    all_parameters_saver = tf.train.Saver()
    divisor = step_size_of_showing_result(TEST_BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
        # tf.summary.FileWriter(SAVED_MODELS_DIR, sess.graph)
        all_parameters_saver.restore(sess=sess, save_path=CKPT_PATH)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sum_loss = 0.0
        sum_acc = 0.0
        try:
            epoch = 0
            while not coord.should_stop():
                example, label = sess.run([test_images, test_labels])
                label = label.reshape(TEST_BATCH_SIZE, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH)
                img, loss, acc = sess.run(
                    [tf.argmax(input=unet.prediction, axis=3), unet.loss_mean, unet.accuracy],
                    feed_dict={
                        unet.input_image: example,
                        unet.input_label: label,
                        unet.keep_prob: 1.0,
                        unet.lamb: 0.004,
                        unet.is_training: False
                    }
                )
                sum_loss += loss
                sum_acc += acc
                cv2.imwrite(os.path.join(TEST_SAVE_DIR, '%d.png' % epoch), img[0] * 255)
                epoch += 1
                if epoch % divisor == 0:
                    print('num %d ,  accuracy: %.6f' % (epoch * TEST_BATCH_SIZE, acc))
        except tf.errors.OutOfRangeError:
            print('❗️Done testing -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
    print('❗️Done testing. Average loss: %.6f , accuracy: %.6f\n' % (sum_loss / epoch, sum_acc / epoch))


def predict(unet, prediction_save_dir=PREDICTION_SAVE_DIR, ckpt_path=CKPT_PATH):
    import cv2
    from keras.preprocessing import image
    import numpy as np
    image_list = get_sorted_files(ORIGINAL_IMG_DIR, "png")
    print("🚩️" + str(len(image_list)) + " images to be predicted, will be saved to", prediction_save_dir)
    if not os.path.lexists(PREDICTION_SAVE_DIR):
        os.mkdir(PREDICTION_SAVE_DIR)
    all_parameters_saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
        # tf.summary.FileWriter(SAVED_MODELS_DIR, sess.graph)
        all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
        for index, image_path in enumerate(image_list):
            original_img = image.load_img(image_path,
                                          target_size=(
                                              OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH, OUTPUT_IMG_CHANNEL),
                                          color_mode="grayscale")
            original_img = image.img_to_array(original_img)
            img = np.expand_dims(original_img, axis=0)
            predict_image = sess.run(unet.prediction,
                                     feed_dict={
                                         unet.input_image: img,
                                         unet.keep_prob: 1.0,
                                         unet.lamb: 0.004,
                                         unet.is_training: False
                                     }
                                     )
            # save_path = os.path.join(PREDICTION_SAVED_DIRECTORY, '%d.png' % index)
            # predict_with_models.to_hot_cmap(predict_image, save_path, argmax_axis=3)
            predict_image = predict_image.reshape(OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH, OUTPUT_IMG_CHANNEL)
            # image.save_img(os.path.join(PREDICTION_SAVED_DIRECTORY, '%d.png' % index), predict_image * 255)
            cv2.imwrite(os.path.join(PREDICTION_SAVE_DIR, '%d.png' % index), predict_image * 255)
    print("🚩Predictions are saved, now converting them to 'hot' color map")
    predict_with_models.to_hot_cmap(PREDICTION_SAVE_DIR)
    print('❗️Done prediction\n')


if __name__ == '__main__':
    if not os.path.exists(TEST_SAVE_DIR):
        os.system("mkdir -p " + TEST_SAVE_DIR)
    if not os.path.exists(PREDICTION_SAVE_DIR):
        os.system("mkdir -p " + PREDICTION_SAVE_DIR)

    if not os.path.exists(FLAGS.dataset_dir_path + "/tfrecords/train.tfrecords"):
        logging.warning("❗️.tfrecords files used to train do not exist, generating now...")
        convert_npy_to_tfrecords.convert_whole_dataset(FLAGS.dataset_dir_path)

    main()
