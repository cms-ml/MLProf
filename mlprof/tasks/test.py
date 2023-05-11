# coding: utf-8

"""
Collection of test tasks.
"""

import os

import luigi
import law

from mlprof.tasks.base import BaseTask

# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import subprocess
# from law.contrib.matplotlib.formatter import MatplotlibFormatter
# from law.contrib.csv.formatter import PandasCsvFormatter
from law.target.local import LocalFileTarget
from mlprof.plotting.plotter import plot_batchsize
from mlprof.tools.tools import create_corrected_cfg, all_elements_list_as_one_string, create_name_and_size_vectors
# from IPython import embed

class TestTask(BaseTask):
    """
    Some dummy task to show how to write a task creating a json file using the law dump json-formatter
    """
    i = luigi.IntParameter(default=1)

    def output(self):
        return self.local_target(f"the_test_file_{self.i}.json")

    def run(self):
        j = self.i * 3
        self.output().dump({"j": j}, indent=4, formatter="json")


class CMSSWTestTask(BaseTask, law.tasks.RunOnceTask):
    """
    Some dummy task to show how to write a test to be run in the cmssw sandbox
    """

    i = TestTask.i

    sandbox = "bash::$MLP_BASE/sandboxes/cmssw_default.sh"

    def requires(self):
        return TestTask.req(self)

    @law.tasks.RunOnceTask.complete_on_success
    def run(self):
        # print the content of the input
        print(self.input().load(formatter="json"))

        # print the cmssw version
        print(os.getenv("CMSSW_VERSION"))


class RuntimeParametersTask(BaseTask):
    """
    Task to provide the parameters for the runtime measurement tasks
    """

    graph_path = luigi.Parameter(default="/afs/cern.ch/user/n/nprouvos/public/graph.pb",
                                description="path of the graph to be tested; "
                                "default: /afs/cern.ch/user/n/nprouvos/public/graph.pb",
                                 )
    input_files = luigi.Parameter(default="/afs/cern.ch/user/n/nprouvos/public/testfile.root",
                                 description="the absolute path of the input files in root format "
                                 "to be openened in CMSSW, might not be used for the input values, "
                                 "depending on the input type argument; "
                                 "default: /afs/cern.ch/user/n/nprouvos/public/testfile.root",
                                  )
    # input_sizes = luigi.IntParameter(default=10,
    #                                 description="The size of the input tensor in the network; default: 10",
    #                                 )
    output_size = luigi.IntParameter(default=1,
                                     description="The size of the output tensor of the network; default: 1",
                                     )
    input_type = luigi.Parameter(default="random",
                                description="Type of input values to be used, either random or incremental "
                                "(custom type to be added?); default: random",
                                 )
    # input_tensor_names = luigi.Parameter(default="input",
    #                                     description="Tensorflow name of the input into the given "
    #                                     "network; default: input",
    #                                     )
    # output_tensor_name = luigi.Parameter(default="Identity",
    #                                      description="Tensorflow name of the output of the given "
    #                                      "network; default:Identity",
    #                                      )
    number_runs = luigi.IntParameter(default=500,
                                     description="The number of batches to be evaluated and "
                                     "measurement averaged upon; default: 500",
                                     )
    number_warm_ups = luigi.IntParameter(default=50,
                                         description="The number of batches to be evaluated before "
                                         "the actual measurement; default: 50",
                                         )
    output_directory = luigi.Parameter(default=law.NO_STR,
                                       description="The path to the folder to save the csv file "
                                       "with the results, standard law path will be used if value "
                                       "is law.NO_STR; default:law.NO_STR",
                                       )

    input_shapes = law.CSVParameter(
        default=("input:10"),
        description="the name of the input layers followed by their shapes, separated by a comma. "
        "The format is 'name_input_tensor1:first_dimension_shape-second_dim_shape,name_input_tensor2:...'"
        " Therefore, the name of the layer may not contain ':'. default is ('input:10')."
        )       # TODO?: Allow for input names with ":" in name???? Choose delimiter oneself?

    output_tensor_names = law.CSVParameter(
        default=("Identity"),
        description="the name of the output nodes, separated by a comma. "
        "The format is 'name_output_tensor1,name_output_tensor2...'"
        "default is ('Identity')."
        )





class CreateConfig(RuntimeParametersTask):
    def output(self):
        return LocalFileTarget(path=os.path.join(os.environ.get("MLP_BASE"), "MLProf", "RuntimeModule", "test", "my_plugin_runtime_cfg.py"))

    def run(self):
        if self.output_directory == law.NO_STR:
            self.output_directory = os.path.join(os.sep + os.path.join(*self.local_target().path.split(os.sep)[:-2]), "RuntimeMeasurement", self.version) + os.sep
        print(self.output_directory)
        self.inputs, self.input_tensor_names, self.input_sizes, self.inputs_dimensions = create_name_and_size_vectors(self.input_shapes)

        parameter_dict = {"GRAPH_PATH_PLACEHOLDER": self.graph_path,
                          "INPUT_FILES_PLACEHOLDER": self.input_files,
                          "INPUT_SIZE_PLACEHOLDER": self.input_sizes,
                          "INPUT_CLASS_DIMENSION_PLACEHOLDER": self.inputs_dimensions,
                          "INPUT_TYPE_PLACEHOLDER": self.input_type,
                          "INPUT_TENSOR_NAME_PLACEHOLDER": self.input_tensor_names,
                          "OUTPUT_TENSOR_NAME_PLACEHOLDER": list(self.output_tensor_names),
                          "NUMBER_RUNS_PLACEHOLDER": self.number_runs,
                          "NUMBER_WARM_UPS_PLACEHOLDER": self.number_warm_ups,
                          #"BATCH_SIZES_PLACEHOLDER": self.batch_sizes,
                          "OUTPUT_DIRECTORY_PLACEHOLDER": self.output_directory,
                          }
        create_corrected_cfg(os.path.join(os.environ.get("MLP_BASE"), "MLProf", "utils", "my_plugin_runtime_cfg_template.py"),
                         self.output().path,
                         parameter_dict,
                         )


class CheckPath(BaseTask, law.tasks.RunOnceTask):
    @law.tasks.RunOnceTask.complete_on_success
    def run(self):
        # print the content of the input
        print(os.getcwd())
        print(os.environ.get("MLP_BASE"))
        print(os.path.join(os.environ.get("MLP_BASE"), "MLProf", "RuntimeModule", "test", "my_plugin_runtime_cfg.py"))
        path=os.path.join(os.environ.get("MLP_BASE"), "MLProf", "RuntimeModule", "test", "my_plugin_runtime_cfg.py")
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i<5:
                    print(line)

class RuntimeMeasurementOneBatchSize(RuntimeParametersTask):
    """
    Task to provide the time measurements of the inference of a network in cmssw, given the input parameters
    and a single batch size

    Output is a result_batch_size_{batch_size}.csv file. """
    batch_size = luigi.IntParameter(default=1,
                                    description="The size of the batch for which the runtime measurement is done",
                                    )

    def requires(self):
        return CreateConfig.req(self)

    def output(self):
        if self.output_directory != law.NO_STR:
            return LocalFileTarget(path=os.path.join(self.output_directory, "results_batch_size_{}.csv".format(self.batch_size)))
        else:
            return self.local_target("results_batch_size_{}.csv".format(self.batch_size))

    def run(self):
        if not os.path.isdir(self.output().dirname):
            os.makedirs(self.output().dirname)
        embed()
        process = subprocess.Popen("source " + os.path.join(os.environ.get("MLP_BASE"), "mlprof", "tools", "runtime.sh")+
                                   + " batchsizes=" + str(self.batchsize) + " filename=" + self.output.split(os.sep)[-1],
                               shell=True,
                               stdout=None,
                               stderr=None)
        process.wait()




class RuntimeMeasurement(RuntimeParametersTask):
    """
    Task to provide the time measurements of the inference of a network in cmssw, given the input parameters

    Output is the result.csv file.
    """

    batch_sizes = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(1, 2, 4),
        max_len=20,
        description="the different batchsizes to be tested; default: 1,2,4)",
    )

    sandbox = "bash::$MLP_BASE/sandboxes/mlprof_gcc900_12_2_3.sh"

    # def requires(self):
    #     return [RuntimeMeasurementOneBatchSize.req(self, batch_size=i) for i in self.batch_sizes]
    def requires(self):
        return CreateConfig.req(self)

    def output(self):
        if self.output_directory != law.NO_STR:
            return LocalFileTarget(path=os.path.join(self.output_directory, "results.csv"))
        else:
            return self.local_target("results.csv")

    def run(self):
        if not os.path.isdir(self.output().dirname):
            os.makedirs(self.output().dirname)

        batchsizes_list = list(self.batch_sizes)
        batchsizes_argument_varparser = batchsizes_list.copy()
        for i, batchsize in enumerate(batchsizes_argument_varparser):
            batchsizes_argument_varparser[i] = "batchsizes=" + str(batchsize)
        # embed()
        process = subprocess.Popen("source " + os.path.join(os.environ.get("MLP_BASE"), "mlprof", "tools", "runtime.sh") +
                                   all_elements_list_as_one_string(batchsizes_argument_varparser),
                                   shell=True,
                                   stdout=None,
                                   stderr=None)
        process.wait()


class RuntimePlotterTask(RuntimeParametersTask, law.tasks.RunOnceTask):
    """
    Task to plot the results from the runtime measurements depending on the batch sizes given as parameters,
    default are 1, 2 and 4.
    """
    number_runs = RuntimeMeasurement.number_runs
    output_directory_plot = luigi.Parameter(default=law.NO_STR,
                                       description="The path to the folder to save the pdf file "
                                       "with the plots called runtime_plot_different_batchsizes.pdf, "
                                       "standard law path will be used if value "
                                       "is law.NO_STR; default:law.NO_STR",
                                            )
    output_directory = RuntimeMeasurement.output_directory
    batch_sizes = RuntimeMeasurement.batch_sizes

    def requires(self):
        return RuntimeMeasurement.req(self)

    def output(self):
        if self.output_directory_plot != law.NO_STR:
            return LocalFileTarget(path=os.path.join(self.output_directory_plot,
                                                     "runtime_plot_different_batchsizes.pdf"))
        else:
            return self.local_target("runtime_plot_different_batchsizes.pdf")

    def run(self):
        if not os.path.isdir(self.output().dirname):
            os.makedirs(self.output().dirname)
        plot_batchsize(np.array(self.batch_sizes), self.number_runs, self.input().path, self.output().path)
        print("plot saved")
