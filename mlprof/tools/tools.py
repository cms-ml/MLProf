# coding: utf-8


def all_elements_list_as_one_string(some_string_list, delimiter):
    '''
    Change all the elements of a list of strings to a single string separating the elements of the
    list with the delimiter. The first element of the string is the delimiter. The obtained string is returned.
    '''
    string = ""
    for element in some_string_list:
        string = string + delimiter + str(element)
    return string


def create_corrected_cfg(template_cfg_path, output_cfg_path, params_dict):
    '''
    Use the template_cfg and the values from the params_dict to create the cfg to be used in cmsRun
    '''
    with open(template_cfg_path, 'r') as f_in, open(output_cfg_path, 'w') as f_out:
        for line in f_in:
            for key, value in params_dict.items():
                line = line.replace(key, str(value))
            f_out.write(line)


def create_name_and_size_vectors(inputs):
    dict_inputs = {}
    for i, input_value in enumerate(list(inputs)):
        splitted_input = input_value.split(":")
        key_listed = splitted_input[:-1]
        key = "".join(key_listed)
        # not really needed with "join" as long as there is only one string for the name [0] would work too
        splitted_shape = splitted_input[-1].split("-")
        int_splitted_shape = [int(value) for value in splitted_shape]
        dict_inputs[key] = int_splitted_shape
    inputs = dict_inputs

    input_tensor_names = list(inputs.keys())
    input_sizes = []
    inputs_dimensions = []
    for input_class in inputs.values():
        inputs_dimensions.append(len(input_class))
        input_sizes = input_sizes + input_class

    if len(input_tensor_names) != len(inputs_dimensions):
        print("WARNING: The number of input names for the input tensor does not match the "
              "number of classes for which an input size was given")

    return inputs, input_tensor_names, input_sizes, inputs_dimensions


def merge_csv_files(input_paths, output_path):
    import pandas as pd

    dataset_list = []
    for input_path in input_paths:
        pd_dataset = pd.read_csv(input_path, delimiter=",", names=["batch_size", "mean", "std"])
        dataset_list = dataset_list + [pd_dataset]
    all_datasets = pd.concat(dataset_list)
    all_datasets.to_csv(output_path, header=False, index=False)
