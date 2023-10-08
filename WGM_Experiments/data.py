import numpy as np
data_folder_dictionary = {
    'temperature_230509_discrete': "5.9.23/",
    'pressure_230516_discrete': "5.16.23_Pressure/"
}

data_file_dictionary = {
    'temperature_230509_discrete': {
        'folder for raw data'   :"5.9.23/",
        'where data'       :"data_temp230509.pickle",
        'where indices'    :"indices_temp230509.pickle"
    },
    'pressure_230516_discrete': {
        'folder for raw data'   :"5.16.23_Pressure/",
        'where data'       :"data_pres230516.pickle",
        'where indices'    :"indices_pres230516.pickle" 
    }
}

def alternate_rows_loop(datas):
    """ datas: LIST of SHUFFLEd data
        \ntheir shape[0] = count; shape[1 & up] = dimension"""
    sizes = [data.shape[0] for data in datas]
    max_size = np.max(sizes)
    print(f"longest dataset has {max_size} items")
    alternate_rows = []
    for index in range(max_size):
        for data_index, data in enumerate(datas):
            if index < sizes[data_index]:
                alternate_rows.append(data[index])
    return alternate_rows
import itertools as it
def alternate_rows_itertools(datas):
    alternate_rows = []
    for group in it.zip_longest(*datas):
        # alternate_rows += [item for item in group if item is not None] # <- list comp
        # def condition(item): # this function returns BOOL!
        #     return item is not None
        # alternate_rows += list(filter(condition, group)) # use filter function (DNE itertools.ifilter in Python 3!)
        alternate_rows += list(filter(lambda q:q is not None, group)) # most concise, lambda = function
    return np.asarray(alternate_rows)

class data_location:
    def __init__(self):
        pass
    # def __init__(self,
    #              input_kind,
    #              label_kind,
    #              folder,
    #              file_dictionary,
    #              ) -> None:
    #     pass
    def yo(self):
        print("yo")