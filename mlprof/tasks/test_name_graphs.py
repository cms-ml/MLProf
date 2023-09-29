"""
Collection of test tasks.
"""

import os

import tensorflow as tf
import luigi
import law

from mlprof.tasks.base import BaseTask


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        # print(f.read())
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and return it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.graph_util.import_graph_def(graph_def, name="prefix")
    return graph


class PrintLayersNamesAndSizesTFFrozenGraph(BaseTask, law.tasks.RunOnceTask):
    """
    Task to print the names and sizes of the tensorflow layers of the given frozen graph
    Two options are provided to reduce the output and get only the

    """

    graph_path = luigi.Parameter(default="/afs/cern.ch/user/n/nprouvos/public/graph.pb",
                                 description="path to frozen graph",
                                 )

    no_model_layers = luigi.BoolParameter(default=False,
                                          description="choose to print the layers indicated by the prefix "
                            "'model' = intermediate layers from a model created without sequential in keras",
                                          )

    no_sequential_layers = luigi.BoolParameter(default=False,
                                        description="choose to print the layers indicated by the prefix "
                                        "'sequential' = intermediate layers from a sequential model in keras",
                                               )

    sandbox = "bash::$MLP_BASE/sandboxes/venv_tf.sh"

    @law.tasks.RunOnceTask.complete_on_success
    def run(self):
        graph = load_graph(self.graph_path)
        for op in graph.get_operations():
            if self.no_model_layers:
                if op.name.split(os.sep)[1] == "model":
                    continue
            if self.no_sequential_layers:
                if op.name.split(os.sep)[1] == "sequential":
                    continue
            print(op.name, op.outputs)
