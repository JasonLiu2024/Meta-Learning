import numpy as np
import pickle
import data as dat

def get_data(datafile_name):
    pass

def setup(data_names_list):
    data_dictionary = {}
    for data_name in data_names_list:
        print(f"loading {data_name}__________________________")
        data_dictionary[data_name] = {}
        data_dictionary[data_name]['data'], data_dictionary[data_name]['label']  = load_data(
            dat.data_file_dictionary[data_name]['where data'])
        data_dictionary[data_name]['train indices'], data_dictionary[data_name]['valid indices'], data_dictionary[data_name]['test indices']= load_indices(
            dat.data_file_dictionary[data_name]['where indices'])
    return data_dictionary

def load_data(datafile):
    """ DOES: loads data from pickle file
        GIVE: data, as numpy array
        datafile:       filename
        type_of_input:  printout info"""
    with open(datafile, 'rb') as file:
        raw_input, raw_output = pickle.load(file)
    print(f"\tinput shape (number, dimension): {raw_input.shape}")
    print(f"\tlabel shape (number, dimension): {raw_output.shape}")
    return raw_input, raw_output

def load_indices(indicesfile):
    """ DOES: loads indices from pickle file
        GIVE: indices, as list
        indicesfile:    filename
        type_of_input:  printout info"""
    with open(indicesfile, 'rb') as file:
        train_indices, valid_indices, test_indices = pickle.load(file)
    print(f"\tthere are {len(train_indices)} folds")
    print(f"\t{len(train_indices[0])} for training, {len(valid_indices[0])} for validating, {len(test_indices)} for testing")
    return train_indices, valid_indices, test_indices

def get_task_indices(indices, ratio, size):
    """ DOES: get indices for meta-learning tasks
        GIVE: indices, as numpy array
        indices:    list of indices  
        ratio:      support-query split
        size:       total number of examples per task"""
    indices = np.asarray(indices) # convert to array to index with list
    count = indices.shape[0]
    # list comprehension; ok to overshoot indices (list ignores the overshoot)
    groups = [indices[i:i + size] for i in range(0, count, size)]
    tasks_indices = []
    for group in groups:
        group_count = group.shape[0] # need to count, bc integer division
        # I use unpacking operator * here; gets list out of range()
        support_indices = group[[*range(0, int(group_count * ratio))]]
        query_indices = group[[*range(int(group_count * ratio), group_count)]]
        task_indices = (support_indices, query_indices)
        tasks_indices.append(task_indices)
    return tasks_indices