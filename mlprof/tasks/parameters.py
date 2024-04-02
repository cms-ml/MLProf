# coding: utf-8

"""
Collection of the recurrent luigi parameters for different tasks.
"""

import os

import luigi
import law

from mlprof.tasks.base import BaseTask


class Model(object):

    def __init__(self, model_file: str, name, label, **kwargs):

        super().__init__(**kwargs)

        self.model_file = os.path.abspath(os.path.expandvars(os.path.expanduser(model_file)))
        self.name = name
        self.label = label

        # cached data
        self._all_data = None
        self._data = None

    @property
    def data(self):
        if self._data is None:
            all_data = law.LocalFileTarget(self.model_file).load(formatter="yaml")
            if "model" not in all_data:
                raise Exception(f"model file '{self.model_file}' is missing 'model' field")
            self._data = all_data["model"]
            self._all_data = all_data
        return self._data

    @property
    def full_name(self):
        if self.name:
            return self.name

        # create a hash
        name = os.path.splitext(os.path.basename(self.model_file))[0]
        return f"{name}{law.util.create_hash(self.model_file)}"

    @property
    def full_model_label(self):
        if self.label:
            return self.label

        # get the model.name field in the model data
        model_name = self.data.get("name")
        if model_name:
            return model_name

        # fallback to the full model name
        return self.full_name


class CMSSWParameters(BaseTask):
    """
    Parameters related to the CMSSW environment
    """

    cmssw_version = luigi.Parameter(
        default="CMSSW_14_1_X_2024-04-01-1100",
        description="CMSSW version; default: CMSSW_14_1_X_2024-04-01-1100",
    )
    scram_arch = luigi.Parameter(
        default="slc7_amd64_gcc12",
        description="SCRAM architecture; default: slc7_amd64_gcc12",
    )

    def store_parts(self):
        parts = super().store_parts()

        cmssw_repr = [self.cmssw_version, self.scram_arch]
        parts.insert_before("version", "cmssw", "__".join(cmssw_repr))

        return parts


class RuntimeParameters(BaseTask):
    """
    General parameters for the model definition and the runtime measurement.
    """

    input_type = luigi.Parameter(
        default="random",
        description="either 'random', 'incremental', 'zeros', 'ones', or a path to a root file; "
        "default: random",
    )
    n_events = luigi.IntParameter(
        default=1,
        description="number of events to be processed; default: 1",
    )
    n_calls = luigi.IntParameter(
        default=100,
        description="number of evaluation calls for averaging; default: 100",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # verify the input type
        self.input_file = None
        if self.input_type not in ("random", "incremental", "zeros"):
            self.input_file = os.path.abspath(os.path.expandvars(os.path.expanduser(self.input_type)))
            if not os.path.exists(self.input_file):
                raise ValueError(
                    f"input type '{self.input_type}' is neither 'random' nor 'incremental' nor 'zeros' nor "
                    f"a path to an existing root file",
                )

    def store_parts(self):
        parts = super().store_parts()

        # build a combined string that represents the significant parameters
        params = [
            f"input_{law.util.create_hash(self.input_file) if self.input_file else self.input_type}",
            f"nevents_{self.n_events}",
            f"ncalls_{self.n_calls}",
        ]
        parts.insert_before("version", "runtime_params", "__".join(params))

        return parts


class ModelParameters(BaseTask):
    """
    General parameters for the model definition and the runtime measurement.
    """

    model_file = luigi.Parameter(
        default="$MLP_BASE/examples/simple_dnn/model_tf.json",
        description="json file containing information of model to be tested; "
        "default: $MLP_BASE/examples/simple_dnn/model_tf.json",
    )
    model_name = luigi.Parameter(
        default=law.NO_STR,
        description="when set, use this name for storing outputs instead of a hashed version of "
        "--model-file; default: empty",
    )
    model_label = luigi.Parameter(
        default=law.NO_STR,
        description="when set, use this label in plots; when empty, the 'network_name' field in the model json data is "
        "used when existing, and full_name otherwise; default: empty",
    )

    @classmethod
    def modify_param_values(cls, params) -> dict:
        params = super().modify_param_values(params)

        if params.get("model_file"):
            params["model_file"] = os.path.abspath(os.path.expandvars(os.path.expanduser(params["model_file"])))

        return params

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = Model(
            model_file=self.model_file,
            name=self.model_name if self.model_name != law.NO_STR else None,
            label=self.model_label if self.model_label != law.NO_STR else None,
        )

    def store_parts(self):
        parts = super().store_parts()

        # build a combined string that represents the significant parameters
        params = [
            f"model_{self.model.full_name}",
        ]
        parts.insert_before("version", "model_params", "__".join(params))

        return parts


class MultiModelParameters(BaseTask):
    """
    General parameters for the model definition and the runtime measurement.
    """

    model_files = law.CSVParameter(
        description="comma-separated list of json files containing information of models to be tested",
        brace_expand=True,
    )
    model_names = law.CSVParameter(
        default=law.NO_STR,
        description="comma-separated list of names of models defined in --model-files to use in output paths "
        "instead of a hashed version of model_files; when set, the number of names must match the number of "
        "model files; default: ()",
    )
    model_labels = law.CSVParameter(
        default=law.NO_STR,
        description="when set, use this label in plots; when empty, the 'network_name' field in the "
        "model json data is used when existing, and full_model_name otherwise; default: empty",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check that lengths match if initialized
        if self.model_names[0] == law.NO_STR:
            if (self.model_labels[0] != law.NO_STR) and (len(self.model_files) != len(self.model_labels)):
                raise ValueError("the lengths of model_files and model_labels must be the same")
        elif self.model_labels[0] == law.NO_STR:
            if len(self.model_files) != len(self.model_names):
                raise ValueError("the lengths of model_files and model_names must be the same")
        elif len({len(self.model_files), len(self.model_names), len(self.model_labels)}) != 1:
            raise ValueError("the lengths of model_names, model_files and model_labels must be the same")

        # if not initialized, change size objects for them to match
        if len(self.model_names) != len(self.model_files):
            self.model_names = (law.NO_STR,) * len(self.model_files)
        if len(self.model_labels) != len(self.model_files):
            self.model_labels = (law.NO_STR,) * len(self.model_files)

        # define Model objects
        self.models = [
            Model(
                model_file=x,
                name=y if y != law.NO_STR else None,
                label=z if z != law.NO_STR else None,
            )
            for x, y, z in zip(self.model_files, self.model_names, self.model_labels)
        ]

    def store_parts(self):
        parts = super().store_parts()

        # build a combined string that represents the significant parameters
        params = [
            f"model_{model.full_name}" for model in self.models
        ]
        parts.insert_before("version", "model_params", "__".join(params))

        return parts


class BatchSizesParameters(BaseTask):
    """
    Parameter to add several batch sizes to perform the measurement on
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


class CustomPlotParameters(BaseTask):
    """
    Parameters for customization of plotting
    """

    log_y = luigi.BoolParameter(
        default=False,
        description="plot the y-axis values logarithmically; default: False",
    )
    bs_normalized = luigi.BoolParameter(
        default=True,
        description="normalize the measured values with the batch size; default: True",
    )
    filling = luigi.BoolParameter(
        default=True,
        description="plot the errors as error bands instead of error bars; default: True",
    )
    top_right_label = luigi.Parameter(
        default="",
        description="stick a label over the top right corner of the plot",
    )

    @property
    def custom_plot_params(self):
        return {
            "log_y": self.log_y,
            "bs_normalized": self.bs_normalized,
            "filling": self.filling,
            "top_right_label": self.top_right_label,
        }
