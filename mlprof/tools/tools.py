def all_elements_list_as_one_string(some_string_list):
    '''
    Change all the elements of a list of strings to a single string separating the elements of the
    list with whitespaces. The obtained string is returned.
    '''
    string = ""
    for element in some_string_list:
        string = string + " " + element
    return string
