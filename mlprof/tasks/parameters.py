# coding: utf-8

"""
Collection of the recurrent luigi parameters for different tasks.
"""

import os

import luigi
import law

from mlprof.tasks.base import BaseTask
from mlprof.util import expand_path


class Model(object):

    def __init__(self, model_file: str, name: str, label: str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model_file = expand_path(model_file, abs=True)
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
        return f"{name}_{law.util.create_hash(self.model_file)}"

    @property
    def full_model_label(self):
        if self.label:
            return self.label

        # get the model.label field in the model data
        model_label = self.data.get("label")
        if model_label:
            return model_label

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
        default="CMSSW_14_1_X_2024-04-04-2300",
        description="CMSSW version; default: CMSSW_14_1_X_2024-04-04-2300",
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


class MultiCMSSWParameters(BaseTask):

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


class RuntimeParameters(BaseTask):
    """
    General parameters for the model definition and the runtime measurement.
    """

    input_data = luigi.Parameter(
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
    batch_size = luigi.IntParameter(
        default=1,
        description="the batch size to measure the runtime for; default: 1",
    )
    tfaot_batch_rules = law.Parameter(
        default=law.NO_STR,
        description="dash-separated tfaot batch rules with each being in the format 'target_size:size_1,size_2,...';"
        "default: empty",
    )

    default_input_file = "/afs/desy.de/user/r/riegerma/public/testfile.root"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # verify the input data
        self.input_file = self.default_input_file
        known_input_data = {"random", "incremental", "zeros", "ones"}
        if self.input_data not in known_input_data:
            self.input_file = expand_path(self.input_data, abs=True)
            if not os.path.exists(self.input_file):
                raise ValueError(
                    f"invalid input data '{self.input_data}', must be a file or any of {','.join(known_input_data)}",
                )
            self.input_data = "file"

    def store_parts(self):
        parts = super().store_parts()

        # build a combined string that represents the significant parameters
        input_str = f"file{law.util.create_hash(self.input_file)}" if self.input_data == "file" else self.input_data
        params = [
            f"input_{input_str}",
            f"nevents_{self.n_events}",
            f"ncalls_{self.n_calls}",
        ]

        # optional parts
        if self.tfaot_batch_rules:
            params.append(f"tfaotrules_{self.tfaot_batch_rules}")

        parts.insert_before("version", "runtime_params", "__".join(params))

        return parts


class ModelParameters(BaseTask):
    """
    General parameters for the model definition and the runtime measurement.
    """

    model_file = luigi.Parameter(
        default="$MLP_BASE/examples/simple_dnn/model_tf.yaml",
        description="json or yaml file containing information of model to be tested; "
        "default: $MLP_BASE/examples/simple_dnn/model_tf.yaml",
    )
    model_name = luigi.Parameter(
        default=law.NO_STR,
        description="when set, use this name for storing outputs instead of a hashed version of --model-file; "
        "default: empty",
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
            params["model_file"] = expand_path(params["model_file"], abs=True)

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
        names = [model.full_name for model in self.models]
        if len(names) >= 5:
            names = [f"{len(names)}x{law.util.create_hash(names)}"]
        parts.insert_before("version", "model_params", f"models__{'__'.join(names)}")

        return parts


class BatchSizesParameters(BaseTask):
    """
    Parameters to control batch sizes.
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
    Parameters plotting customizations.
    """

    x_log = luigi.BoolParameter(
        default=True,
        significant=False,
        description="plot the x-axis logarithmically; default: True",
    )
    y_log = luigi.BoolParameter(
        default=False,
        significant=False,
        description="plot the y-axis logarithmically; default: False",
    )
    y_min = luigi.FloatParameter(
        default=law.NO_FLOAT,
        significant=False,
        description="minimum y-axis value; default: empty",
    )
    y_max = luigi.FloatParameter(
        default=law.NO_FLOAT,
        significant=False,
        description="maximum y-axis value; default: empty",
    )
    bs_normalized = luigi.BoolParameter(
        default=True,
        significant=False,
        description="normalize the measured values with the batch size; default: True",
    )
    error_style = luigi.ChoiceParameter(
        choices=["bars", "band"],
        default="band",
        significant=False,
        description="style of errors / uncerainties of due to averaging; choices: bars,band; default: band",
    )
    top_right_label = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="stick a label over the top right corner of the plot",
    )

    @property
    def custom_plot_params(self):
        return {
            "x_log": self.x_log,
            "y_log": self.y_log,
            "y_min": self.y_min if self.y_min != law.NO_FLOAT else None,
            "y_max": self.y_max if self.y_max != law.NO_FLOAT else None,
            "bs_normalized": self.bs_normalized,
            "error_style": self.error_style,
            "top_right_label": None if self.top_right_label == law.NO_STR else self.top_right_label,
        }
