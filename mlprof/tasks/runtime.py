# coding: utf-8

"""
Collection of test tasks.
"""

import os
import itertools

import luigi  # type: ignore[import-untyped]
import law  # type: ignore[import-untyped]

from mlprof.tasks.base import CMSRunCommandTask, PlotTask, view_output_plots
from mlprof.tasks.parameters import (
    RuntimeParameters, ModelParameters, MultiModelParameters, CMSSWParameters, MultiCMSSWParameters,
    BatchSizesParameters, CustomPlotParameters,
)
from mlprof.tasks.sandboxes import CMSSWSandboxTask
from mlprof.plotting.plotter import plot_batch_size_several_measurements
from mlprof.util import expand_path


class RemoveCMSSWSandbox(CMSSWParameters, ModelParameters, law.tasks.RunOnceTask):

    @law.tasks.RunOnceTask.complete_on_success
    def run(self):
        sandbox_task = CMSSWSandboxTask.req(self)
        install_dir = os.path.join("$MLP_CMSSW_BASE", sandbox_task.cmssw_install_dir)
        law.LocalDirectoryTarget(install_dir).remove(silent=True)


class MeasureRuntime(
    CMSRunCommandTask,
    RuntimeParameters,
    CMSSWSandboxTask,
):
    """
    Task to provide the time measurements of the inference of a network in cmssw, given the input
    parameters and a single batch size.
    """

    renew_cmssw_sandbox = luigi.BoolParameter(
        default=False,
        description="remove the cmssw sandbox corresponding to the inference engine of the requested model first; "
        "default: False",
    )

    def requires(self):
        return RemoveCMSSWSandbox.req(self) if self.renew_cmssw_sandbox else []

    def output(self):
        return self.local_target(f"runtime_bs{self.batch_size}.csv")

    def build_command(self):
        # determine the config to run
        engine = self.model.data["inference_engine"]
        config_file = f"$MLP_BASE/cmssw/MLProf/RuntimeMeasurement/test/{engine}_runtime_cfg.py"

        # build cmsRun command options
        options = {
            "inputFiles": law.target.file.add_scheme(self.input_file, "file://"),
            "batchSize": self.batch_size,
            "csvFile": self.output().path,
            "inputType": "random" if self.input_data == "file" else self.input_data,
            "maxEvents": self.n_events,
            "nCalls": self.n_calls,
        }

        # engine specific options
        if engine in {"tf", "onnx"}:
            graph_path = expand_path(self.model.data["file"])
            model_dir = expand_path(self.model_file, dir=True)
            options.update({
                "graphPath": os.path.normpath(os.path.join(model_dir, graph_path)),
                "inputTensorNames": [inp["name"] for inp in self.model.data["inputs"]],
                "outputTensorNames": [outp["name"] for outp in self.model.data["outputs"]],
                "inputRanks": [len(inp["shape"]) for inp in self.model.data["inputs"]],
                "flatInputSizes": sum((inp["shape"] for inp in self.model.data["inputs"]), []),
            })
        elif engine == "tfaot":
            if self.tfaot_batch_rules != law.NO_STR:
                options["batchRules"] = self.tfaot_batch_rules_option

        return self.build_cmsrun_command(expand_path(config_file), options)


class MergeRuntimes(
    BatchSizesParameters,
    RuntimeParameters,
    ModelParameters,
    CMSSWParameters,
):

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
    BatchSizesParameters,
    RuntimeParameters,
    ModelParameters,
    CMSSWParameters,
    CustomPlotParameters,
    PlotTask,
):
    """
    Task to plot the results from the runtime measurements depending on the batch sizes given as parameters,
    default are 1, 2 and 4.
    """

    sandbox = "bash::$MLP_BASE/sandboxes/plotting.sh"

    def requires(self):
        return MergeRuntimes.req(self)

    def output(self):
        return self.local_target(f"runtimes_bs{self.batch_sizes_repr}.pdf")

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
            [self.model.color],
            self.custom_plot_params,
        )
        print("plot saved")


class PlotMultiRuntimes(
    BatchSizesParameters,
    RuntimeParameters,
    MultiModelParameters,
    MultiCMSSWParameters,
    CustomPlotParameters,
    PlotTask,
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

        # TODO: refactor the combinatorics below

        # list of sequences over which the product is performed for the requirements
        self.product_names_req = ["model_file", "model_name", "cmssw_version", "scram_arch"]
        self.product_sequences_req = [
            list(zip(self.model_files, self.model_names or (n_models * [None]))),
            self.cmssw_versions,
            self.scram_archs,
        ]

        # list of sequences over which the product is performed for the output file name
        self.product_names_out = ["cmssw_version", "scram_arch"]
        self.product_sequences_out = [
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
        return [
            MergeRuntimes.req(self, **dict(zip(self.product_names_req, self.flatten_tuple(values))))
            for values in itertools.product(*self.product_sequences_req)
        ]

    def output(self):
        return self.local_target(f"runtimes_{self.out_params_repr}_bs{self.batch_sizes_repr}.pdf")

    @view_output_plots
    def run(self):
        # prepare the output directory
        output = self.output()
        output.parent.touch()

        input_paths = [inp.path for inp in self.input()]

        # create the plot
        plot_batch_size_several_measurements(
            batch_sizes=self.batch_sizes,
            input_paths=input_paths,
            output_path=output.path,
            measurements=self.params_product_params_to_write,
            color_list=[model.color for model in self.models],
            plot_params=self.custom_plot_params,
        )
