data_folder_dictionary = {
    'temperature_230509_discrete': "5.9.23/",
    'pressure_230516_discrete': "5.16.23_Pressure/"
}

data_file_dictionary = {
    'temperature_230509_discrete'   : "data_pres230516.pickle",
    'pressure_230516_discrete'      : "data_temp230509.pickle",
    'temperature': {
        'folder for raw data'   :"5.9.23/",
        'folder for data'       :"data_temp230509.pickle",
        'folder for indices'    :"indices_pres230516.pickle" 
    },
    'pressure': {
        'folder for raw data'   :"5.16.23_Pressure/",
        'folder for data'       :"data_pres230516.pickle",
        'folder for indices'    :"indices_temp230509.pickle" 
    }
}

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