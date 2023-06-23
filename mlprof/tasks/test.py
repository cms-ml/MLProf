# coding: utf-8

"""
Collection of test tasks.
"""

import os

import luigi
import law

from mlprof.tasks.base import BaseTask, CommandTask, PlotTask, view_output_plots
from mlprof.plotting.plotter import plot_batchsize


class CMSSWParameters(BaseTask):
    """
    TODO:
        - move to base (or even better: a different name)
    """

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
    """
    TODO:
        - move to base (or even better: a different name)
    """

    @property
    def sandbox(self):
        parts = [
            "cmssw",
            self.cmssw_version,
            f"arch={self.scram_arch}",
            "setup=$MLP_BASE/cmssw/install_sandbox.sh",
            "dir=$MLP_CMSSW_BASE",
        ]
        return "::".join(parts)


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


class BatchSizesParameters(BaseTask):
    """
    TODO:
        - move to base (or even better: a different name)
    """

    batch_sizes = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(1, 2, 4),
        sort=True,
        description="comma-separated list of batch sizes to be tested; default: 1,2,4",
    )

    @property
    def batch_sizes_repr(self):
        return "_".join(map(str, self.batch_sizes))


class CreateRuntimeConfig(RuntimeParameters, CMSSWParameters):
    """
    TODO:
        - rename template vars
    """

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
        template = "$MLP_BASE/cmssw/MLProf/RuntimeModule/test/runtime_template_cfg.py"
        content = law.LocalFileTarget(template).load(formatter="text")

        # replace variables
        for key, value in template_vars.items():
            content = content.replace(key, str(value))

        # write the output
        output.dump(content, formatter="text")


class MeasureRuntime(CommandTask, RuntimeParameters, CMSSWSandboxTask):
    """
    Task to provide the time measurements of the inference of a network in cmssw, given the input
    parameters and a single batch size

    Output is a result_batch_size_{batch_size}.csv file.
    """

    batch_size = luigi.IntParameter(
        default=1,
        description="the batch size to measure the runtime for; default: 1",
    )

    def requires(self):
        return CreateRuntimeConfig.req(self)

    def output(self):
        return self.local_target(f"runtime_bs_{self.batch_size}.csv")

    def build_command(self):
        return [
            "cmsRun",
            self.input().path,
            f"batchSizes={self.batch_size}",
            f"csvFile={self.output().path}",
        ]


class MergeRuntimes(RuntimeParameters, CMSSWParameters, BatchSizesParameters):

    def requires(self):
        return [
            MeasureRuntime.req(self, batch_size=batch_size)
            for batch_size in self.batch_sizes
        ]

    def output(self):
        return self.local_target(f"runtimes_bs_{self.batch_sizes_repr}.csv")

    def run(self):
        # merge files
        lines = [
            inp.load(formatter="text")
            for inp in self.input()
        ]

        # remove empty lines
        lines = [_line for _line in (line.strip() for line in lines) if _line]

        # save it
        self.output().dump(
            "\n".join(lines),
            formatter="text",
        )


class PlotRuntimes(RuntimeParameters, CMSSWParameters, BatchSizesParameters, PlotTask):
    """
    Task to plot the results from the runtime measurements depending on the batch sizes given as parameters,
    default are 1, 2 and 4.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/plotting.sh"

    def requires(self):
        return MergeRuntimes.req(self)

    def output(self):
        return self.local_target(f"runtime_plot_different_batchsizes_{self.batch_sizes_repr}.pdf")

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        # create the plot
        plot_batchsize(self.batch_sizes, self.repetitions, self.input().path, output.path)
        print("plot saved")
