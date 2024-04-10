# coding: utf-8

__all__ = []

import os


def expand_path(path, abs=False, dir=False):
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if abs:
        path = os.path.abspath(path)
    if dir:
        path = os.path.dirname(path)
    return path
