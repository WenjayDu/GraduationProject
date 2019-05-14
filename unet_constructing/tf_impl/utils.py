import os
import sys

import numpy as np
import tensorflow as tf

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from config_and_utils import GlobalVar, logging, get_sorted_files
from data_processing.img_utils import to_hot_cmap

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

# these flags are added because error is arisen for they undefined when importing this module from
# data_processing.predict_with_models, therefore, define them here
tf.flags.DEFINE_string(name="mode", default="client", help="just to avoid err")
tf.flags.DEFINE_integer(name="port", default=65533, help="just to avoid err")

ROOT_OUTPUT_DIR = FLAGS.dataset_dir_path + '/models/tf_impl'
REAL_OUTPUT_DIR = ROOT_OUTPUT_DIR + '/' + FLAGS.structure

LOG_DIR = REAL_OUTPUT_DIR + "/logs"
MODEL_SAVE_DIR = REAL_OUTPUT_DIR + "/saved_model"
CKPT_PATH = MODEL_SAVE_DIR + "/model_of_" + FLAGS.structure + ".ckpt"

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

# EPS below is the value added to denominator in BN operation,
# to prevent the 0 operation when dividing by the variance,
# may be different in different frameworks.
EPS = 10e-5


def read_image(file_queue, shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    data = tf.decode_raw(features['data'], np.float16)
    data = tf.reshape(data, shape)

    label = tf.decode_raw(features['label'], np.float16)
    label = tf.reshape(label, shape)

    return data, label


def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue, shape=FLAGS.input_shape)
    min_after_dequeue = 500
    capacity = 510
    # image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
    image_batch, label_batch = tf.train.shuffle_batch(
        tensors=[img, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    # one_hot_labels = tf.reshape(label_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH])
    # print("image_batch shape", image_batch.get_shape(), "label_batch shape", label_batch.get_shape())
    return image_batch, label_batch


def step_size_of_showing_result(batch_size):
    if batch_size == 1 or batch_size == 2:
        return 1000
    elif batch_size < 32:
        return 100
    else:
        return 10


def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
    from tensorflow.python.training.moving_averages import assign_moving_average

    with tf.variable_scope(name):
        params_shape = x.shape[-1:]
        moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer,
                                     trainable=False)

        def mean_var_with_update():
            mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
            with tf.control_dependencies([
                assign_moving_average(moving_mean, mean_this_batch, decay),
                assign_moving_average(moving_var, variance_this_batch, decay)
            ]):
                return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

        mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
        if affine:  # If you want to scale with beta and gamma
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
                                               variance_epsilon=eps)
        else:
            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                               variance_epsilon=eps)
        return normed


def merge_results_from_contracting_and_upsampling(result_from_contracting, result_from_upsampling):
    result_from_contracting_crop = result_from_contracting
    return tf.concat(values=[result_from_contracting_crop, result_from_upsampling], axis=-1)


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

                lo, acc, summary_str = sess.run([unet.loss_mean, unet.accuracy, merged_summary],
                                                feed_dict={unet.input_image: example,
                                                           unet.input_label: label,
                                                           unet.keep_prob: 1.0,
                                                           unet.lamb: 0.004,
                                                           unet.is_training: True}
                                                )
                summary_writer.add_summary(summary_str, epoch)
                sess.run([unet.train_step], feed_dict={unet.input_image: example, unet.input_label: label,
                                                       unet.keep_prob: 1.0, unet.lamb: 0.004, unet.is_training: True})
                epoch += 1
                if epoch % divisor == 0:
                    logging.info('num %d , loss: %.6f , accuracy: %.6f' % (epoch * TRAIN_BATCH_SIZE, lo, acc))
        except tf.errors.OutOfRangeError:
            logging.warning('❗️Done training -- epoch limit reached')
        finally:
            all_parameters_saver.save(sess=sess, save_path=CKPT_PATH)
            coord.request_stop()
        coord.join(threads)
    logging.warning('❗️Done training. TOTAL: num: %d , loss: %.6f , accuracy: %.6f\n'
                    % (epoch * TRAIN_BATCH_SIZE, lo, acc))
    logging.info("🚩model has been saved as " + CKPT_PATH)


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
                lo, acc = sess.run([unet.loss_mean, unet.accuracy],
                                   feed_dict={unet.input_image: example,
                                              unet.input_label: label,
                                              unet.keep_prob: 1.0,
                                              unet.lamb: 0.004,
                                              unet.is_training: False}
                                   )
                # summary_writer.add_summary(summary_str, epoch)
                epoch += 1
                if epoch % divisor == 0:
                    logging.info('num %d , loss: %.6f , accuracy: %.6f' % (epoch * VALIDATION_BATCH_SIZE, lo, acc))
        except tf.errors.OutOfRangeError:
            logging.warning('❗️Done validating -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
    logging.warning('❗️Done validating. TOTAL: num: %d , loss: %.6f , accuracy: %.6f'
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
                img, loss, acc = sess.run([tf.argmax(input=unet.prediction, axis=3), unet.loss_mean, unet.accuracy],
                                          feed_dict={unet.input_image: example,
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
                    logging.info('num %d ,  accuracy: %.6f' % (epoch * TEST_BATCH_SIZE, acc))
        except tf.errors.OutOfRangeError:
            logging.warning('❗️Done testing -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
    logging.warning('❗️Done testing. Average loss: %.6f , accuracy: %.6f' % (sum_loss / epoch, sum_acc / epoch))


def predict(unet, prediction_save_dir=PREDICTION_SAVE_DIR, ckpt_path=CKPT_PATH, original_img_path=ORIGINAL_IMG_DIR):
    from keras.preprocessing import image
    import numpy as np
    image_list = get_sorted_files(original_img_path, "png")
    logging.info("🚩️" + str(len(image_list)) + " images to be predicted, will be saved to " + prediction_save_dir)
    if not os.path.lexists(prediction_save_dir):
        os.mkdir(prediction_save_dir)
    all_parameters_saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
        # tf.summary.FileWriter(SAVED_MODELS_DIR, sess.graph)
        all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
        for index, image_path in enumerate(image_list):
            original_img = image.load_img(image_path,
                                          target_size=(OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH, OUTPUT_IMG_CHANNEL),
                                          color_mode="grayscale")
            original_img = image.img_to_array(original_img)
            img = np.expand_dims(original_img, axis=0)
            predict_image = sess.run(unet.prediction,
                                     feed_dict={unet.input_image: img,
                                                unet.keep_prob: 1.0,
                                                unet.lamb: 0.004,
                                                unet.is_training: False
                                                }
                                     )
            # save_path = os.path.join(prediction_save_dir, '%d.png' % index)
            # predict_with_models.to_hot_cmap(predict_image, save_path, argmax_axis=3)
            predict_image = predict_image.reshape(OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDTH, OUTPUT_IMG_CHANNEL)
            image.save_img(os.path.join(prediction_save_dir, '%d.png' % index), predict_image * 255)
            # cv2.imwrite(os.path.join(prediction_save_dir, '%d.png' % index), predict_image)

    logging.info("🚩️Predictions are saved, now converting them to 'hot' color map")
    to_hot_cmap(prediction_save_dir)
    logging.warning('❗️Done prediction')
