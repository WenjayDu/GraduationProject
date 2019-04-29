import traceback
import tensorflow as tf
from learners.learner_utils import create_learner
from model_compressing.unet_at_mri_for_pf import ModelHelper
from model_compressing.config import GlobalPath

OUTPUT_DIR = GlobalPath.OUTPUT_PATH + "/tf_model_compressing"
LOGS_DIR = OUTPUT_DIR + "/logs"
SAVED_MODELS_DIR = OUTPUT_DIR + "/saved_models"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', LOGS_DIR, 'logging directory')
tf.app.flags.DEFINE_boolean('enbl_multi_gpu', False, 'enable multi-GPU training')
tf.app.flags.DEFINE_string('learner', 'full-prec', 'learner\'s name')
tf.app.flags.DEFINE_string('exec_mode', 'train', 'execution mode: train / eval')
tf.app.flags.DEFINE_boolean('debug', False, 'debugging information')


def main(unused_arg):
    """Main entry."""

    try:
        # setup the TF logging routine
        if FLAGS.debug:
            tf.logging.set_verbosity(tf.logging.DEBUG)
        else:
            tf.logging.set_verbosity(tf.logging.INFO)
        sm_writer = tf.summary.FileWriter(FLAGS.log_dir)

        # display FLAGS's values
        tf.logging.info('FLAGS:')
        for key, value in FLAGS.flag_values_dict().items():
            tf.logging.info('{}: {}'.format(key, value))

        # build the model helper & learner
        print("❗️creating model_helper...")
        model_helper = ModelHelper()
        print("❗️creating learner...")
        learner = create_learner(sm_writer, model_helper)

        # execute the learner
        if FLAGS.exec_mode == 'train':
            print("❗️start training")
            learner.train()
        elif FLAGS.exec_mode == 'eval':
            print("❗️start downloading the model...")
            learner.download_model()
            print("❗done downloading, start evaluating...")
            learner.evaluate()
        else:
            raise ValueError('unrecognized execution mode: ' + FLAGS.exec_mode)

        # exit normally
        return 0
    except ValueError:
        traceback.print_exc()
        # exit with error
        return 1


if __name__ == '__main__':
    tf.app.run()
