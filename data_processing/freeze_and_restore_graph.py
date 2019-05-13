import sys

import tensorflow as tf

from config_and_utils import GlobalVar

PROJECT_DIR = GlobalVar.PROJECT_PATH
OUTPUT_DIR = GlobalVar.OUTPUT_PATH + "/tf_impl_original"
SAVED_MODELS = OUTPUT_DIR + "/saved_models"
FROZEN_GRAPH_FILE = SAVED_MODELS + "/frozen_graph.pb"

OUTPUT_NODE_NAMES = "accuracy/Mean"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(name="--model_dir", default=SAVED_MODELS, help="Model folder containing ckpt to export")
tf.flags.DEFINE_string(name="--output_node_names", default=OUTPUT_NODE_NAMES,
                       help="The name of the output nodes, comma separated.")


def freeze_graph(model_dir=SAVED_MODELS, output_node_names=OUTPUT_NODE_NAMES):
    """
    freeze the current graph into a .pb file according to specified output nodes

    :param model_dir: the dir containing the ckpt state file
    :param output_node_names: all the output node's names
    :return: the output graph
    """
    if not tf.gfile.Exists(model_dir):
        print("Model directory containing ckpt doesn't exists.")
        sys.exit(1)

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
        print("🚩Done freezing the graph to ", FROZEN_GRAPH_FILE)
        print("🚩A tatal of %d ops in the output graph." % len(output_graph.node))

    return output_graph


def restore_graph(frozen_graph_filename=FROZEN_GRAPH_FILE):
    # read from the frozen graph .pb file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # for fixing the bug of batch norm
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    # import into a new Graph and return
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    print("🚩Done restoring the graph from the ", frozen_graph_filename)
    return graph


def get_all_output_nodes():
    for op in tf.get_default_graph().get_operations():
        print(op.name)


if __name__ == '__main__':
    freeze_graph(FLAGS.model_dir, FLAGS.output_node_names)
