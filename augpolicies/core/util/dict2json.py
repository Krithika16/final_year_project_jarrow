import copy


def serializable_objects_in_dict(input_dict):
    dict_copy = {}
    for key in input_dict:
        if key != "strategy":
            dict_copy[key] = input_dict[key]
    dict_copy = copy.deepcopy(dict_copy)
    return recurrsive_to_serializable(dict_copy)


def recurrsive_to_serializable(element):
    if type(element) is dict:
        for sub_element_key in element:
            element[sub_element_key] = recurrsive_to_serializable(element[sub_element_key])
        return element
    elif type(element) is list:
        for idx, sub_element in enumerate(element):
            element[idx] = recurrsive_to_serializable(sub_element)
        return element
    elif type(element) in [int, float, str, bool]:
        return element
    else:
        return element.__name__
