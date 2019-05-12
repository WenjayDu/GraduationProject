import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

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
