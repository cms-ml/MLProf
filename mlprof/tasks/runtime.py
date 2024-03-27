# coding: utf-8

"""
Collection of test tasks.
"""

import os
import itertools

import luigi
import law

from mlprof.tasks.base import CommandTask, PlotTask, view_output_plots
from mlprof.tasks.parameters import (
    RuntimeParameters, ModelParameters, MultiModelParameters, CMSSWParameters, BatchSizesParameters,
    CustomPlotParameters,
)
from mlprof.tasks.sandboxes import CMSSWSandboxTask
from mlprof.plotting.plotter import plot_batch_size_several_measurements


class CreateRuntimeConfig(RuntimeParameters, ModelParameters, CMSSWParameters):

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
        model_data = self.model.data

        # resolve the graph path relative to the model file
        graph_path = os.path.expandvars(os.path.expanduser(model_data["file"]))
        model_file = os.path.expandvars(os.path.expanduser(self.model_file))
        graph_path = os.path.join(os.path.dirname(model_file), graph_path)

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
        if model_data["inference_engine"] == "tf":
            template = "$MLP_BASE/cmssw/MLProf/RuntimeMeasurement/test/tf_runtime_template_cfg.py"
        elif model_data["inference_engine"] == "onnx":
            template = "$MLP_BASE/cmssw/MLProf/RuntimeMeasurement/test/onnx_runtime_template_cfg.py"
        else:
            raise Exception("The only inference_engine supported are 'tf' and 'onnx'")

        content = law.LocalFileTarget(template).load(formatter="text")
        # replace variables
        for key, value in template_vars.items():
            content = content.replace(f"__{key}__", str(value))

        # write the output
        output.dump(content, formatter="text")


class MeasureRuntime(CommandTask, RuntimeParameters, ModelParameters, CMSSWSandboxTask):
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
        return self.local_target(f"runtime_bs{self.batch_size}.csv")

    def build_command(self):
        return [
            "cmsRun",
            self.input().path,
            f"batchSize={self.batch_size}",
            f"csvFile={self.output().path}",
        ]


class MergeRuntimes(RuntimeParameters, ModelParameters, CMSSWParameters, BatchSizesParameters):

    def requires(self):
        return [
            MeasureRuntime.req(self, batch_size=batch_size)
            for batch_size in self.batch_sizes
        ]

    def output(self):
        return self.local_target(f"runtimes_bs{self.batch_sizes_repr}.csv")

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


class PlotRuntimes(
    RuntimeParameters,
    ModelParameters,
    CMSSWParameters,
    BatchSizesParameters,
    PlotTask,
    CustomPlotParameters,
):
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

        # create the plot
        plot_batch_size_several_measurements(
            self.batch_sizes,
            [self.input().path],
            output.path,
            [self.model.full_model_label],
            self.custom_plot_params,
        )
        print("plot saved")


class PlotRuntimesMultipleParams(
    RuntimeParameters,
    MultiModelParameters,
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

    cmssw_versions = law.CSVParameter(
        default=(CMSSWParameters.cmssw_version._default,),
        description=f"comma-separated list of CMSSW versions; default: ({CMSSWParameters.cmssw_version._default},)",
        brace_expand=True,
    )
    scram_archs = law.CSVParameter(
        default=(CMSSWParameters.scram_arch._default,),
        description=f"comma-separated list of SCRAM architectures; default: ({CMSSWParameters.scram_arch._default},)",
        brace_expand=True,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check that, if given, the number of model names matches that of model names
        n_models = len(self.model_files)
        if len(self.model_names) not in (n_models, 0):
            raise ValueError("the number of model names does not match the number of model files")

        # list of sequences over which the product is performed for the requirements
        self.product_names_req = ["model_file", "model_name", "cmssw_version", "scram_arch"]
        self.product_sequences_req = [
            list(zip(self.model_files, self.model_names or (n_models * [None]))),
            self.cmssw_versions,
            self.scram_archs,
        ]

        # list of sequences over which the product is performed for the output file name
        self.product_names_out = ["model_name", "cmssw_version", "scram_arch"]
        self.product_sequences_out = [
            tuple([model.full_name for model in self.models]),
            self.cmssw_versions,
            self.scram_archs,
        ]

        # list of sequences over which the product is performed for the labels in plot
        self.product_names_labels = ["model_label", "cmssw_version", "scram_arch"]
        self.product_sequences_labels = [
            tuple([model.full_model_label for model in self.models]),
            self.cmssw_versions,
            self.scram_archs,
        ]

        # define output product
        self.output_product = list(itertools.product(*self.product_sequences_out))
        self.output_product_dict = [dict(zip(self.product_names_out, values)) for values in self.output_product]

        # retrieve the names of the params to be put in output
        self.params_to_write_outputs = []
        for iparam, param in enumerate(self.product_names_out):
            if len(self.product_sequences_out[iparam]) > 1:
                self.params_to_write_outputs += [param]

        # create output representation to be used in output file name
        self.output_product_params_to_write = [
            combination_dict[key_to_write]
            for combination_dict in self.output_product_dict
            for key_to_write in self.params_to_write_outputs
        ]

        self.out_params_repr = "_".join(self.output_product_params_to_write)

        # define label product
        self.labels_products = list(itertools.product(*self.product_sequences_labels))
        self.labels_products_dict = [dict(zip(self.product_names_labels, values)) for values in self.labels_products]

        # retrieve the names of the params to be put in labels
        self.params_to_write_labels = []
        for iparam, param in enumerate(self.product_names_labels):
            if len(self.product_sequences_labels[iparam]) > 1:
                self.params_to_write_labels += [param]

        # create list of labels to plot
        self.params_product_params_to_write = [
            tuple([combination_dict[key_to_write] for key_to_write in self.params_to_write_labels])
            for combination_dict in self.labels_products_dict
        ]

    def flatten_tuple(self, value):
        for x in value:
            if isinstance(x, tuple):
                yield from self.flatten_tuple(x)
            else:
                yield x

    def requires(self):
        flattened_product = [
            tuple(self.flatten_tuple(tuple_of_args)) for tuple_of_args in itertools.product(*self.product_sequences_req)
        ]
        return [
            MergeRuntimes.req(self, **dict(zip(self.product_names_req, values)))
            for values in flattened_product
        ]

    def output(self):
        return self.local_target(
            f"runtime_plot_{self.out_params_repr}_different_batch_sizes_{self.batch_sizes_repr}.pdf",
        )

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        input_paths = [inp.path for inp in self.input()]

        # create the plot

        plot_batch_size_several_measurements(
            self.batch_sizes,
            input_paths,
            output.path,
            self.params_product_params_to_write,
            self.custom_plot_params,
        )
