# coding: utf-8

from __future__ import annotations

__all__: list[str] = []

import os

import law  # type: ignore[import-untyped]


def expand_path(path: str, abs: bool = False, dir: bool = False) -> str:
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if abs:
        path = os.path.abspath(path)
    if dir:
        path = os.path.dirname(path)
    return path


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
