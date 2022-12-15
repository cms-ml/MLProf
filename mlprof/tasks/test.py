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
from mlprof.tools.tools import all_elements_list_as_one_string


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


class RuntimeMeasurement(BaseTask):
    """
    Task to provide the time measurements of the inference of a network in cmssw, given the input parameters

    Output is the result.csv file.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/mlprof_gcc900_12_2_3.sh"

    graph_path = luigi.Parameter(default="/afs/cern.ch/user/n/nprouvos/public/graph.pb",
                                description="path of the graph to be tested; "
                                "default: /afs/cern.ch/user/n/nprouvos/public/graph.pb",
                                 )
    input_files = luigi.Parameter(default="/afs/cern.ch/user/n/nprouvos/public/testfile.root",
                                 description="the absolute path of the input_files in root format; "
                                 "default: /afs/cern.ch/user/n/nprouvos/public/testfile.root",
                                  )
    input_size = luigi.IntParameter(default=10,
                                    description="The size of the input tensor in the network; default: 10",
                                    )
    output_size = luigi.IntParameter(default=1,
                                     description="The size of the output tensor of the network; default: 1",
                                     )
    input_tensor_name = luigi.Parameter(default="input",
                                        description="Tensorflow name of the input into the given "
                                        "network; default: input",
                                        )
    output_tensor_name = luigi.Parameter(default="Identity",
                                         description="Tensorflow name of the output of the given "
                                         "network; default:Identity",
                                         )
    number_runs = luigi.IntParameter(default=200,
                                     description="The number of batches to be evaluated and "
                                     "measurement averaged upon; default: 200",
                                     )
    number_warm_ups = luigi.IntParameter(default=50,
                                         description="The number of batches to be evaluated before "
                                         "the actual measurement; default: 50",
                                         )
    batch_sizes = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(1, 2, 4),
        max_len=20,
        description="the different batchsizes to be tested; default: 1,2,4)",
    )
    output_directory = luigi.Parameter(default=law.NO_STR,
                                       description="The path to the folder to save the csv file "
                                       "with the results, standard law path will be used if value "
                                       "is law.NO_STR; default:law.NO_STR",
                                       )

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
        process = subprocess.Popen("source /afs/desy.de/user/p/prouvost/xxl/af-cms/MLProf/mlprof/tools/runtime.sh" +
                               " graphPath=" + self.graph_path +
                               " inputFiles=file://" + self.input_files + " inputSize=" + str(self.input_size) +
                               " outputSize=" + str(self.output_size) +
                               " inputTensorName=" + self.input_tensor_name +
                               " outputTensorName=" + self.output_tensor_name +
                               " numberRuns=" + str(self.number_runs) + " numberWarmUps=" + str(self.number_warm_ups) +
                               " outputPath=" + self.output().path + " " + str(len(self.batch_sizes)) +
                               all_elements_list_as_one_string(batchsizes_argument_varparser),
                               shell=True,
                               stdout=None,
                               stderr=None)
        process.wait()


class RuntimePlotterTask(BaseTask, law.tasks.RunOnceTask):
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
