# coding: utf-8

"""
Collection of the sandbox tasks for the different tasks
"""

from mlprof.tasks.parameters import CMSSWParameters


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
