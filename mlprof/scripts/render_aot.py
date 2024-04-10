# coding: utf-8

"""
Renders variables in a tfaot inference plugin.
"""

from __future__ import annotations

import os
import re
import collections


Input = collections.namedtuple("Input", ["array_type", "shape1"])
Output = collections.namedtuple("Output", ["array_type", "shape1"])

# mapping from header type to array type
type_to_array = {
    "bool": "Bool",
    "int32_t": "Int32",
    "int32": "Int32",
    "int": "Int32",
    "int64_t": "Int64",
    "int64": "Int64",
    "long": "Int64",
    "float": "Float",
    "double": "Double",
}


def render_aot(plugin_file: str, header_file: str, model_name: str) -> None:
    # prepare paths
    plugin_file = os.path.expandvars(os.path.expanduser(plugin_file))
    header_file = os.path.expandvars(os.path.expanduser(header_file))
    if not os.path.exists(plugin_file):
        raise FileNotFoundError(f"plugin file '{plugin_file}' does not exist")
    if not os.path.exists(header_file):
        raise FileNotFoundError(f"header file '{header_file}' does not exist")

    # parse the header to extract input and output signatures
    input_signatures, output_signatures = parse_signatures(header_file)

    # prepare content to fill
    content = {}
    input_arrays = [f"tfaot::{inp.array_type}Arrays" for inp in input_signatures]
    input_names = [f"input{i}" for i in range(len(input_signatures))]
    output_arrays = [f"tfaot::{outp.array_type}Arrays" for outp in output_signatures]
    output_names = [f"output{i}" for i in range(len(input_signatures))]

    # model member
    content["model"] = [
        f"tfaot::Model<tfaot_model::{model_name}> model_;",
    ]
    # inputs
    content["inputs"] = [
        f"{arr} {name} = create{inp.array_type}Input({inp.shape1});"
        for inp, arr, name in zip(input_signatures, input_arrays, input_names)
    ]
    # outputs
    content["outputs"] = [
        f"{arr} {name};"
        for arr, name in zip(output_arrays, output_names)
    ]
    # untied inference
    content["untied_inference"] = [
        f"model_.run<{', '.join(output_arrays)}>(batchSize_, {', '.join(input_names)});"
    ]
    # tied inference
    content["tied_inference"] = [
        f"std::tie({', '.join(output_names)}) = {content['untied_inference'][0]};"
    ]

    # read the plugin file content
    with open(plugin_file, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]

    # write the lines back with the content filled in
    with open(plugin_file, "w") as f:
        for line in lines:
            m = re.match(r"^(.*)//\s+INSERT=([^,\s]+).*$", line)
            if m:
                line = m.group(1).join([""] + content[m.group(2)])
            f.write(f"{line}\n")


def parse_signatures(header_file: str) -> tuple[list[Input], list[Output]]:
    from cms_tfaot import parse_header

    # extract header header
    header_data = parse_header(header_file)

    # build input and output signatures
    input_signatures = [
        Input(type_to_array[t], c)
        for t, c in zip(header_data.arg_types, header_data.arg_counts_no_batch)
    ]
    output_signatures = [
        Output(type_to_array[t], c)
        for t, c in zip(header_data.res_types, header_data.res_counts_no_batch)
    ]

    return input_signatures, output_signatures


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin_file", help="path to the plugin file to render in-place")
    parser.add_argument("header_file", help="path to the header file to read variables from")
    parser.add_argument("model_name", help="name of the model class")
    args = parser.parse_args()

    render_aot(args.plugin_file, args.header_file, args.model_name)

    return 0


if __name__ == "__main__":
    main()
