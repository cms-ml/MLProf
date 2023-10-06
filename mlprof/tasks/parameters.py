# coding: utf-8

"""
Collection of the recurrent luigi parameters for different tasks
"""

import os

import luigi
import law

from mlprof.tasks.base import BaseTask


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
        description="comma-separated list of absolute paths of input files for the CMSSW analyzer; "
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
        description="number of evaluations to be performed before starting the actual measurement; default: 10",
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
    TODO: docstring
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

class PlotCustomParameters(BaseTask):
    """
    TODO: docstring
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

    @property
    def custom_plot_params(self):
        return {"log_y":self.log_y, "bs_normalized":self.bs_normalized, "filling":self.filling}
