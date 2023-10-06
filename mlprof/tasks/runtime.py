# coding: utf-8

"""
Collection of test tasks.
"""

import luigi
import law

from mlprof.tasks.base import CommandTask, PlotTask, view_output_plots
from mlprof.tasks.parameters import RuntimeParameters, CMSSWParameters, BatchSizesParameters, PlotCustomParameters
from mlprof.tasks.sandboxes import CMSSWSandboxTask
from mlprof.plotting.plotter import plot_batchsize, plot_batchsize_several_measurements


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


class PlotRuntimes(RuntimeParameters, CMSSWParameters, BatchSizesParameters, PlotTask, PlotCustomParameters):
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

        # get name network for legend
        model_data = law.LocalFileTarget(self.model_file).load(formatter="json")
        network_name = model_data["network_name"]

        # create the plot
        plot_batchsize_several_measurements(self.batch_sizes, [self.input().path], output.path, [network_name], self.custom_plot_params)
        print("plot saved")


class PlotRuntimesSeveralNetworks(RuntimeParameters, CMSSWParameters, BatchSizesParameters, PlotTask, PlotCustomParameters):
    """
    Task to plot the results from the runtime measurements depending on the batch sizes given as parameters,
    default are 1, 2 and 4.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/plotting.sh"

    model_files = law.CSVParameter(
        description="comma-separated list of json files containing information of models to be tested"
    )

    def requires(self):
        return [MergeRuntimes.req(self, model_file=model_file) for model_file in self.model_files]

    def output(self):
        network_names = []
        for model_file in self.model_files:
            model_data = law.LocalFileTarget(model_file).load(formatter="json")
            network_names += [model_data["network_name"]]
        network_names_repr = "_".join(network_names)
        return self.local_target(f"runtime_plot_networks_{network_names_repr}_different_batchsizes_{self.batch_sizes_repr}.pdf")

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        # create the plot
        network_names = []
        input_paths = []
        for model_file in self.model_files:
            model_data = law.LocalFileTarget(model_file).load(formatter="json")
            network_names += [model_data["network_name"]]
        for input_task in self.input():
            input_paths += [input_task.path]
        plot_batchsize_several_measurements(self.batch_sizes, input_paths,
                                        output.path, network_names, self.custom_plot_params)
        # plot_batchsize_several_measurements(self.batch_sizes, [self.input()[0].path, self.input()[0].path],
        #                                output.path, ["model_1", "model_2"])
        print("plot saved")


class PlotRuntimesMultipleCMSSW(RuntimeParameters, CMSSWParameters, BatchSizesParameters, PlotTask, PlotCustomParameters):
    """
    Task to plot the results from the runtime measurements depending on the batch sizes given as parameters,
    default are 1, 2 and 4.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/plotting.sh"

    cmssw_versions = law.CSVParameter(
        cls=luigi.Parameter,
        default=("CMSSW_12_2_4","CMSSW_12_2_2"),
        description="comma-separated list of CMSSW versions; default: ('CMSSW_12_2_4','CMSSW_12_2_2')",
        brace_expand=True,
    )

    def requires(self):
        return [MergeRuntimes.req(self, cmssw_version=cmssw_version) for cmssw_version in self.cmssw_versions]

    def output(self):
        cmssw_versions_repr = "_".join(self.cmssw_versions)
        return self.local_target(f"runtime_plot__multiple_cmssw_{cmssw_versions_repr}_different_batchsizes_{self.batch_sizes_repr}.pdf")

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        # create the plot
        input_paths = []
        for input_task in self.input():
            input_paths += [input_task.path]
        from IPython import embed; embed()
        plot_batchsize_several_measurements(self.batch_sizes, input_paths,
                                        output.path, self.cmssw_versions, self.custom_plot_params)
        # plot_batchsize_several_measurements(self.batch_sizes, [self.input()[0].path, self.input()[0].path],
        #                                 output.path, ["CMSSW_12_2_4", "CMSSW_12_2_4"])
        print("plot saved")
