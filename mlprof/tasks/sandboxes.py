# coding: utf-8

"""
Collection of the sandbox tasks for the different tasks
"""

import os

import law  # type: ignore[import-untyped]

from mlprof.tasks.parameters import CMSSWParameters, ModelParameters


class CMSSWSandboxTask(CMSSWParameters, ModelParameters):
    """
    Base class for tasks in cmssw sandboxes.
    """

    @property
    def cmssw_setup_script(self):
        if self.model.data["inference_engine"] == "tfaot":
            return "install_sandbox_tfaot.sh"
        return "install_sandbox.sh"

    @property
    def cmssw_setup_args(self):
        if self.model.data["inference_engine"] == "tfaot":
            return self.model_file
        return ""

    @property
    def cmssw_install_dir(self):
        # engine representation
        engine = self.model.data["inference_engine"]
        engine_str = "tfaot" if engine == "tfaot" else "tf_onnx"

        # hash of all signficant values
        hash_parts = [
            self.cmssw_version,
            self.scram_arch,
            self.cmssw_setup_script,
            self.cmssw_setup_args,
        ]
        if engine == "tfaot":
            comp_data = self.model._all_data["compilation"]
            hash_parts.append((
                sorted(comp_data["batch_sizes"]),
                sorted(comp_data.get("tf_xla_flags") or []),
                sorted(comp_data.get("xla_flags") or []),
            ))

        return f"{self.cmssw_version}_{engine_str}_{law.util.create_hash(hash_parts)}"

    @property
    def sandbox(self):
        # preparations
        args = self.cmssw_setup_args
        install_dir = os.path.join("$MLP_CMSSW_BASE", self.cmssw_install_dir)

        # sandbox parts
        parts = [
            "cmssw",
            f"{self.cmssw_version}",
            f"arch={self.scram_arch}",
            f"setup=$MLP_BASE/cmssw/{self.cmssw_setup_script}",
            f"dir={install_dir}",
        ]
        if args:
            parts.append(f"args={args}")

        return "::".join(parts)

    def run(self):
        raise NotImplementedError
