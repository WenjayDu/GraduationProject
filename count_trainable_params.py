import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('ckpt_path', '', 'path of checkpoint')
tf.flags.DEFINE_string('meta_file_path', '', 'path of meta file')


def restore_saver_and_graph(ckpt_path, meta_file_path):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file_path)
        saver.restore(sess, ckpt_path)
        graph = tf.get_default_graph()
        return saver, graph


def count():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


if __name__ == '__main__':
    saver, graph = restore_saver_and_graph(
        ckpt_path=FLAGS.ckpt_path,
        meta_file_path=FLAGS.meta_file_path)
    num_of_trainable = count()
    print('🚩num of trainable parameters is', num_of_trainable)
