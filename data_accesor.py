def get_data(datafile_name):
    pass

import pickle
def get_raw_data(datafile, type_of_input):
    with open(datafile, 'rb') as file:
        raw_input, raw_output = pickle.load(file)
    print(f"type of data: {type_of_input}")
    print(f"\tnumber of examples: {raw_input.shape[0]}")
    print(f"\traw input shape: {raw_input.shape}, dimension {raw_input.shape[1]}")
    print(f"\traw output shape: {raw_output.shape}, dimension {raw_output.shape[1]}")
    return raw_input, raw_output

def get_cross_validation_indices(indicesfile, type_of_input):
    print(f"type of data: {type_of_input}")
    with open(indicesfile, 'rb') as file:
        train_indices, valid_indices, test_indices = pickle.load(file)
    print(f"\tthere are {len(train_indices)} folds")
    print(f"\t{len(train_indices[0])} for training, {len(valid_indices[0])} for validating, {len(test_indices)} for testing")
    return train_indices, valid_indices, test_indices