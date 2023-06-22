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


def merge_csv_files(input_paths, output_path):
    import pandas as pd

    dataset_list = []
    for input_path in input_paths:
        pd_dataset = pd.read_csv(input_path, delimiter=",", names=["batch_size", "mean", "std"])
        dataset_list = dataset_list + [pd_dataset]
    all_datasets = pd.concat(dataset_list)
    all_datasets.to_csv(output_path, header=False, index=False)
