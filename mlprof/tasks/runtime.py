# coding: utf-8

"""
Collection of test tasks.
"""

import os
import itertools

import luigi
import law

from mlprof.tasks.base import CommandTask, PlotTask, view_output_plots
from mlprof.tasks.parameters import RuntimeParameters, CMSSWParameters, BatchSizesParameters, CustomPlotParameters
from mlprof.tasks.sandboxes import CMSSWSandboxTask
from mlprof.plotting.plotter import plot_batch_size_several_measurements


class CreateRuntimeConfig(RuntimeParameters, CMSSWParameters):

    default_input_files = {
        "CMSSW_*": ["/afs/cern.ch/user/n/nprouvos/public/testfile.root"],
    }

    def find_default_input_files(self):
        for version_pattern, files in self.default_input_files.items():
            if law.util.multi_match(self.cmssw_version, version_pattern):
                return files

        raise Exception(f"no default input files found for '{self.cmssw_version}'")

    def output(self):
        return self.local_target("cfg.py")

    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        # get model data
        model_data = self.model_data

        # resolve the graph path relative to the model file
        graph_path = os.path.expandvars(os.path.expanduser(model_data["file"]))
        graph_path = os.path.join(os.path.dirname(self.model_file), graph_path)

        # determine input files
        if self.input_file:
            input_files = [self.input_file]
            input_type = "random"
        else:
            input_files = self.find_default_input_files()
            input_type = self.input_type

        # prepare template variables
        template_vars = {
            "GRAPH_PATH": graph_path,
            "INPUT_FILES": [
                law.target.file.add_scheme(path, "file://")
                for path in input_files
            ],
            "N_EVENTS": self.n_events,
            "INPUT_TYPE": input_type,
            "INPUT_RANKS": [
                len(inp["shape"])
                for inp in model_data["inputs"]
            ],
            "FLAT_INPUT_SIZES": sum(
                (inp["shape"] for inp in model_data["inputs"]),
                [],
            ),
            "INPUT_TENSOR_NAMES": [inp["name"] for inp in model_data["inputs"]],
            "OUTPUT_TENSOR_NAMES": [outp["name"] for outp in model_data["outputs"]],
            "N_CALLS": self.n_calls,
        }

        # load the template content
        template = "$MLP_BASE/cmssw/MLProf/RuntimeMeasurement/test/tf_runtime_template_cfg.py"
        content = law.LocalFileTarget(template).load(formatter="text")

        # replace variables
        for key, value in template_vars.items():
            content = content.replace(f"__{key}__", str(value))

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


class PlotRuntimes(RuntimeParameters, CMSSWParameters, BatchSizesParameters, PlotTask, CustomPlotParameters):
    """
    Task to plot the results from the runtime measurements depending on the batch sizes given as parameters,
    default are 1, 2 and 4.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/plotting.sh"

    def requires(self):
        return MergeRuntimes.req(self)

    def output(self):
        return self.local_target(f"runtime_plot_different_batch_sizes_{self.batch_sizes_repr}.pdf")

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        # get name network for legend
        model_data = self.model_data
        network_name = model_data["network_name"]

        # create the plot
        plot_batch_size_several_measurements(
            self.batch_sizes,
            [self.input().path],
            output.path,
            [network_name],
            self.custom_plot_params,
        )
        print("plot saved")


class PlotRuntimesMultipleNetworks(
    RuntimeParameters,
    CMSSWParameters,
    BatchSizesParameters,
    PlotTask,
    CustomPlotParameters,
):
    """
    Task to plot the results from the runtime measurements for several networks, depending on the batch sizes given as
    parameters, default are 1, 2 and 4.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/plotting.sh"

    model_files = law.CSVParameter(
        description="comma-separated list of json files containing information of models to be tested",
    )

    def requires(self):
        return [
            MergeRuntimes.req(self, model_file=model_file)
            for model_file in self.model_files
        ]

    def output(self):
        network_names = [req.model_data["network_name"] for req in self.requires()]
        network_names_repr = "_".join(network_names)
        return self.local_target(
            f"runtime_plot_networks_{network_names_repr}_different_batch_sizes_{self.batch_sizes_repr}.pdf",
        )

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        # create the plot
        network_names = [req.model_data["network_name"] for req in self.requires()]
        input_paths = [inp.path for inp in self.input()]
        plot_batch_size_several_measurements(
            self.batch_sizes,
            input_paths,
            output.path,
            network_names,
            self.custom_plot_params,
        )


class PlotRuntimesMultipleCMSSW(
    RuntimeParameters,
    CMSSWParameters,
    BatchSizesParameters,
    PlotTask,
    CustomPlotParameters,
):
    """
    Task to plot the results from the runtime measurements for inferences performed in multiple cmssw versions,
    depending on the batch sizes given as parameters, default are 1, 2 and 4.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/plotting.sh"

    cmssw_versions = law.CSVParameter(
        cls=luigi.Parameter,
        default=("CMSSW_12_2_4", "CMSSW_12_2_2"),
        description="comma-separated list of CMSSW versions; default: CMSSW_12_2_4,CMSSW_12_2_2",
        brace_expand=True,
    )

    def requires(self):
        return [
            MergeRuntimes.req(self, cmssw_version=cmssw_version)
            for cmssw_version in self.cmssw_versions
        ]

    def output(self):
        cmssw_versions_repr = "_".join(self.cmssw_versions)
        return self.local_target(
            f"runtime_plot_multiple_cmssw_{cmssw_versions_repr}_different_batch_sizes_{self.batch_sizes_repr}.pdf",
        )

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        # create the plot
        input_paths = [inp.path for inp in self.input()]
        plot_batch_size_several_measurements(
            self.batch_sizes,
            input_paths,
            output.path,
            self.cmssw_versions,
            self.custom_plot_params,
        )


class PlotRuntimesMultipleParams(
    RuntimeParameters,
    CMSSWParameters,
    BatchSizesParameters,
    PlotTask,
    CustomPlotParameters,
):
    """
    Task to plot the results from the runtime measurements for several parameters, e.g. networks
    or cmssw versions, depending on the batch sizes
    given as parameters, default are 1, 2 and 4.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/plotting.sh"

    model_files = law.CSVParameter(
        description="comma-separated list of json files containing information of models to be tested",
        default=None,
    )

    cmssw_versions = law.CSVParameter(
        cls=luigi.Parameter,
        default=None,
        description="comma-separated list of CMSSW versions; default: ('CMSSW_12_2_4','CMSSW_12_2_2')",
        brace_expand=True,
    )

    # create params_to_write if model_files or cmssw_versions is None? -> gets difficult with itertools product if only one param is changed

    def requires(self):
        self.fill_undefined_param_values()
        all_params = list(itertools.product(self.model_files, self.cmssw_versions))
        return [MergeRuntimes.req(self, model_file=params[0], cmssw_version=params[1]) for params in all_params]

    def output(self):
        self.fill_undefined_param_values()
        all_params = self.factorize_params()
        all_params_list = ["_".join(all_params_item) for all_params_item in all_params]
        all_params_repr = "_".join(all_params_list)
        return self.local_target(f"runtime_plot_params_{all_params_repr}_different_batch_sizes_{self.batch_sizes_repr}.pdf")  # noqa

    def factorize_params(self):
        # get additional parameters plotting
        network_names = []
        for model_file in self.model_files:
            model_data = law.LocalFileTarget(model_file).load(formatter="json")
            network_names += [model_data["network_name"]]

        # combine all parameters together
        all_params = list(itertools.product(network_names, self.cmssw_versions))
        return all_params

    def fill_undefined_param_values(self):
        if self.model_files is None:
            self.model_files = tuple(self.model_file)

        if self.cmssw_versions is None:
            self.cmssw_versions = tuple(self.cmssw_version)

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        self.fill_undefined_param_values()

        input_paths = [inp.path for inp in self.input()]
        print(input_paths)
        all_params = self.factorize_params()

        # create the plot
        plot_batch_size_several_measurements(self.batch_sizes, input_paths,
                                        output.path, all_params, self.custom_plot_params)
