# coding: utf-8

"""
Collection of test tasks.
"""

import os
import subprocess

import luigi
import law
from law.target.local import LocalFileTarget

from mlprof.tasks.base import BaseTask
from mlprof.plotting.plotter import plot_batchsize
from mlprof.tools.tools import all_elements_list_as_one_string, merge_csv_files


class CMSSWParameters(BaseTask):

    cmssw_version = luigi.Parameter(
        default="CMSSW_12_2_4",
        description="CMSSW version; default: CMSSW_12_2_4",
    )
    scram_arch = luigi.Parameter(
        default="slc7_amd64_gcc10",
        description="SCRAM architecture; default: slc7_amd64_gcc10",
    )

    def store_parts(self):
        parts = super().store_parts()

        cmssw_repr = [
            self.cmssw_version,
            self.scram_arch,
        ]
        parts.insert_before("version", "cmssw", "__".join(cmssw_repr))

        return parts


class CMSSWSandboxTask(CMSSWParameters):

    @property
    def sandbox(self):
        return f"cmssw::{self.cmssw_version}::arch={self.scram_arch}"


class RuntimeParameters(BaseTask):

    model_file = luigi.Parameter(
        default="$MLP_BASE/examples/model1/model.json",
        description="json file containing information of model to be tested; "
        "default: $MLP_BASE/examples/model1/model.json",
    )
    model_name = luigi.Parameter(
        default=law.NO_STR,
        description="when set, use this name for storing outputs instead of a hashed version of "
        "--model-file; default: empty",
    )
    input_files = law.CSVParameter(
        default=(),
        description="comma-separeted list of absolute paths of input files for the CMSSW analyzer; "
        "when empty, random input values will be used; default: empty",
    )
    events = luigi.IntParameter(
        default=1,
        description="number of events to read from each input file for averaging measurements; "
        "default: 1",
    )
    repetitions = luigi.IntParameter(
        default=100,
        description="number of repetitions to be performed per evaluation for averaging; "
        "default: 100",
    )
    warmup = luigi.IntParameter(
        default=10,
        significant=False,
        description="number of evaluations to be performed before averaging; default: 10",
    )

    @property
    def abs_input_files(self):
        return [
            os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
            for path in self.input_files
        ]

    @property
    def full_model_name(self):
        if self.model_name not in (None, law.NO_STR):
            return self.model_name

        # create a hash
        model_file = os.path.expandvars(os.path.expanduser(self.model_file))
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        return f"{model_name}{law.util.create_hash(model_file)}"

    def store_parts(self):
        parts = super().store_parts()

        # build a combined string that represents the significant parameters
        params = [
            f"model_{self.full_model_name}",
            f"inputs_{law.util.create_hash(sorted(self.abs_input_files)) if self.input_files else 'empty'}",
            f"events_{self.events}",
            f"repeat_{self.repetitions}",
        ]
        parts.insert_before("version", "model_params", "__".join(params))

        return parts


class CreateRuntimeConfig(RuntimeParameters, CMSSWParameters):

    default_cmssw_files = {
        "CMSSW_*": ["/afs/cern.ch/user/n/nprouvos/public/testfile.root"],
    }

    def find_default_cmssw_files(self):
        for version_pattern, files in self.default_cmssw_files.items():
            if law.util.multi_match(self.cmssw_version, version_pattern):
                return files

        raise Exception(f"no default cmssw files found for version '{self.cmssw_version}'")

    def output(self):
        return self.local_target("cfg.py")

    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        # open the model file
        model_data = law.LocalFileTarget(self.model_file).load(formatter="json")

        # determine input files
        input_files = self.abs_input_files
        input_type = "file"
        if not input_files:
            input_files = self.find_default_cmssw_files()
            input_type = "random"

        # prepare template variables
        template_vars = {
            "GRAPH_PATH_PLACEHOLDER": model_data["file"],
            "INPUT_FILES_PLACEHOLDER": [
                law.target.file.add_scheme(path, "file://")
                for path in input_files
            ],
            "NUMBER_EVENTS_TAKEN": self.events,
            "INPUT_SIZE_PLACEHOLDER": sum(
                (inp["shape"] for inp in model_data["inputs"]),
                [],
            ),
            "INPUT_CLASS_DIMENSION_PLACEHOLDER": [
                len(inp["shape"])
                for inp in model_data["inputs"]
            ],
            "INPUT_TYPE_PLACEHOLDER": input_type,
            "INPUT_TENSOR_NAME_PLACEHOLDER": [inp["name"] for inp in model_data["inputs"]],
            "OUTPUT_TENSOR_NAME_PLACEHOLDER": [outp["name"] for outp in model_data["outputs"]],
            "NUMBER_RUNS_PLACEHOLDER": self.repetitions,
            "NUMBER_WARM_UPS_PLACEHOLDER": self.warmup,
        }

        # load the template content
        template = law.LocalFileTarget("$MLP_BASE/MLProf/RuntimeModule/test/runtime_template_cfg.py")
        content = template.load(formatter="text")

        # replace variables
        for key, value in template_vars.items():
            content = content.replace(key, str(value))

        # write the output
        output.dump(content, formatter="text")



















class RuntimeMeasurementOneBatchSize(RuntimeParameters):
    """
    Task to provide the time measurements of the inference of a network in cmssw, given the input parameters
    and a single batch size

    Output is a result_batch_size_{batch_size}.csv file. """
    batch_size = luigi.IntParameter(default=1,
                                    description="The size of the batch for which the runtime measurement is done",
                                    )

    sandbox = "bash::$MLP_BASE/sandboxes/mlprof_gcc900_12_2_3.sh"

    def requires(self):
        return CreateConfigRuntime.req(self)

    def output(self):
        if self.output_directory != law.NO_STR:
            return LocalFileTarget(path=os.path.join(self.output_directory, "results_batch_size_{}.csv".format(self.batch_size)))
        else:
            return self.local_target("results_batch_size_{}.csv".format(self.batch_size))

    def run(self):
        if not os.path.isdir(self.output().dirname):
            os.makedirs(self.output().dirname)
        # embed()

        process = subprocess.Popen("source " + os.path.join(os.environ.get("MLP_BASE"), "mlprof", "tools", "runtime.sh") +
                                   " batchsizes=" + str(self.batch_size) + " filename=" + self.output().path.split(os.sep)[-1],
                               shell=True,
                               stdout=None,
                               stderr=None)
        process.wait()


class RuntimeMeasurement(RuntimeParameters):
    """
    Task to provide the time measurements of the inference of a network in cmssw, given the input parameters
    This task merges the results from the several occurences of RuntimeMeasurementOneBatchSize.

    Output is the results_batchsize_{all_batch_sizes}.csv file.
    """

    batch_sizes = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(1, 2, 4),
        # max_len=20,
        description="the different batchsizes to be tested; default: (1,2,4)",
    )

    def requires(self):
        return [RuntimeMeasurementOneBatchSize.req(self, batch_size=i) for i in self.batch_sizes]

    def output(self):
        if self.output_directory != law.NO_STR:
            return LocalFileTarget(path=os.path.join(self.output_directory, "results_batchsizes" +
                                                     all_elements_list_as_one_string(self.batch_sizes, "_") + ".csv"))
        else:
            return self.local_target("results_batchsizes" + all_elements_list_as_one_string(self.batch_sizes, "_") + ".csv")

    def run(self):
        if not os.path.isdir(self.output().dirname):
            os.makedirs(self.output().dirname)
        # embed()
        # implement merging csv files here
        input_paths = []
        for input_target in self.input():
            input_paths = input_paths + [input_target.path]
        merge_csv_files(input_paths, self.output().path)


# class RuntimeMeasurement(RuntimeParameters):
#     """
#     Task to provide the time measurements of the inference of a network in cmssw, given the input parameters

#     Output is the result.csv file.
#     """

#     batch_sizes = law.CSVParameter(
#         cls=luigi.IntParameter,
#         default=(1, 2, 4),
#         max_len=20,
#         description="the different batchsizes to be tested; default: 1,2,4)",
#     )

#     sandbox = "bash::$MLP_BASE/sandboxes/mlprof_gcc900_12_2_3.sh"

#     # def requires(self):
#     #     return [RuntimeMeasurementOneBatchSize.req(self, batch_size=i) for i in self.batch_sizes]
#     def requires(self):
#         return CreateConfigRuntime.req(self)

#     def output(self):
#         if self.output_directory != law.NO_STR:
#             return LocalFileTarget(path=os.path.join(self.output_directory, "results.csv"))
#         else:
#             return self.local_target("results.csv")

#     def run(self):
#         if not os.path.isdir(self.output().dirname):
#             os.makedirs(self.output().dirname)

#         batchsizes_list = list(self.batch_sizes)
#         batchsizes_argument_varparser = batchsizes_list.copy()
#         for i, batchsize in enumerate(batchsizes_argument_varparser):
#             batchsizes_argument_varparser[i] = "batchsizes=" + str(batchsize)

#         # embed()
#         process = subprocess.Popen("source " + os.path.join(os.environ.get("MLP_BASE"), "mlprof", "tools", "runtime.sh") +
#                                    all_elements_list_as_one_string(batchsizes_argument_varparser, " "),
#                                    shell=True,
#                                    stdout=None,
#                                    stderr=None)
#         process.wait()


class RuntimePlotterTask(RuntimeParameters, law.tasks.RunOnceTask):
    """
    Task to plot the results from the runtime measurements depending on the batch sizes given as parameters,
    default are 1, 2 and 4.
    """
    # number_runs = RuntimeMeasurement.number_runs
    output_directory_plot = luigi.Parameter(default=law.NO_STR,
                                       description="The path to the folder to save the pdf file "
                                       "with the plots called runtime_plot_different_batchsizes.pdf, "
                                       "standard law path will be used if value "
                                       "is law.NO_STR; default:law.NO_STR",
                                            )
    # output_directory = RuntimeMeasurementSingleBatchSizes.output_directory
    batch_sizes = RuntimeMeasurement.batch_sizes

    def requires(self):
        return RuntimeMeasurement.req(self)

    def output(self):
        if self.output_directory_plot != law.NO_STR:
            return LocalFileTarget(path=os.path.join(self.output_directory_plot,
                                                     "runtime_plot_different_batchsizes" +
                                                     all_elements_list_as_one_string(self.batch_sizes, "_") + ".pdf"))
        else:
            return self.local_target("runtime_plot_different_batchsizes" + all_elements_list_as_one_string(self.batch_sizes, "_") + ".pdf")

    def run(self):
        import numpy as np

        if not os.path.isdir(self.output().dirname):
            os.makedirs(self.output().dirname)
        # embed()
        plot_batchsize(np.array(self.batch_sizes), self.number_runs, self.input().path, self.output().path)
        print("plot saved")
