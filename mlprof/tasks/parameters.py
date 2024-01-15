# coding: utf-8

"""
Collection of the recurrent luigi parameters for different tasks.
"""

import os

import luigi
import law

from mlprof.tasks.base import BaseTask


class CMSSWParameters(BaseTask):
    """
    Parameters related to the CMSSW environment
    """

    cmssw_version = luigi.Parameter(
        default="CMSSW_13_3_1",
        description="CMSSW version; default: CMSSW_13_3_1",
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

    model_file = luigi.Parameter(
        default="$MLP_BASE/examples/simple_dnn/model.json",
        description="json file containing information of model to be tested; "
        "default: $MLP_BASE/examples/simple_dnn/model.json",
    )
    model_name = luigi.Parameter(
        default=law.NO_STR,
        description="when set, use this name for storing outputs instead of a hashed version of "
        "--model-file; default: empty",
    )
    input_type = luigi.Parameter(
        default="random",
        description="either 'random', 'incremental', 'zeros', or a path to a root file; default: random",
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
                    f"input type '{self.input_type}' is neither 'random' nor 'incremental' nor 'zeros' nor a path to an existing "
                    f"root file",
                )

        # cached model content
        self._model_data = None

    @property
    def model_data(self):
        if self._model_data is None:
            self._model_data = law.LocalFileTarget(self.model_file).load(formatter="json")
        return self._model_data

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
            f"input_{law.util.create_hash(self.input_file) if self.input_file else self.input_type}",
            f"nevents_{self.n_events}",
            f"ncalls_{self.n_calls}",
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
        return {"log_y": self.log_y, "bs_normalized": self.bs_normalized, "filling": self.filling,
                "top_right_label": self.top_right_label,
                }
