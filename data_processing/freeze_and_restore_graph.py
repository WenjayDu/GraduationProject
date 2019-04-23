import argparse
import tensorflow as tf
import project_config as config

PROJECT_DIR = config.get_project_path()
OUTPUT_DIR = PROJECT_DIR + "/output/tf_implementation"
SAVED_MODELS = OUTPUT_DIR + "/saved_models"
FROZEN_GRAPH_FILE = SAVED_MODELS + "frozen_graph.pb"

OUTPUT_NODE_NAMES = "accuracy/Mean"


def freeze_graph(model_dir=SAVED_MODELS, output_node_names=OUTPUT_NODE_NAMES):
    """
    freeze the current graph into a .pb file according to specified output nodes

    :param model_dir: the dir containing the ckpt state file
    :param output_node_names: all the output node's names
    :return: the output graph
    """
    if not tf.gfile.Exists(model_dir):
        print("Model directory containing ckpt doesn't exists.")
        exit(1)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # clear devices to allow TF to control on which device it will load operations
    clear_devices = True

    with tf.Session(graph=tf.Graph()) as sess:
        # import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # restore the weights
        saver.restore(sess, input_checkpoint)

        # export variables to constants
        output_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names.split(",")
        )

        # serialize the output graph
        with tf.gfile.GFile(FROZEN_GRAPH_FILE, "wb") as f:
            f.write(output_graph.SerializeToString())
        print("a tatal of %d ops in the output graph." % len(output_graph.node))

    return output_graph


def restore_graph(frozen_graph_filename=FROZEN_GRAPH_FILE):
    # read from the frozen graph .pb file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import into a new Graph and return
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def get_all_output_nodes():
    for op in tf.get_default_graph().get_operations():
        print(op.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=SAVED_MODELS,
                        help="Model folder containing ckpt to export")
    parser.add_argument("--output_node_names", type=str, default=OUTPUT_NODE_NAMES,
                        help="The name of the output nodes, comma separated.")
    FLAGS, _ = parser.parse_known_args()

    freeze_graph(FLAGS.model_dir, FLAGS.output_node_names)
