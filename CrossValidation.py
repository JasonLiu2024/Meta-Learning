import torch
from torch.utils.data import DataLoader
from tools import SaveBestCrossValidationModel
from Style import TextColor
import numpy as np
"""does cross-validation"""
class CrossValidator:
    """ number_of_cross_validation_rounds
        \nnumber_of_epochs
        \nsaver (actual object)
        \ndataset (the function of the object)
        \ndatas: (list of text) data names
        \ndata_dictionary  
        """
    def __init__(self, number_of_cross_validation_rounds, number_of_epochs, 
                 saver, dataset, datas, data_dictionary, manager, settings,
                 device):
        self.number_of_cross_validation_rounds = number_of_cross_validation_rounds
        self.number_of_epochs = number_of_epochs
        self.saver = saver
        self.cross_validation_loss = []
        self.dataset = dataset
        self.datas = datas
        self.data_dictionary = data_dictionary
        self.settings = settings
        self.manager = manager
        self.device = device
        print(f"{TextColor.Bold}{TextColor.BrightGreen_text}__________CROSS VALIDATION__________{TextColor.End}")
        print(f"Cross-validation rounds: {self.number_of_cross_validation_rounds}")
        print(f"Epochs: {self.number_of_epochs}")
        print(f"Datas to learn: ")
        for index, data in enumerate(self.datas):
            print(f"\t{index}: {data}")
    def complete_notify(self):
        # unicode https://www.geeksforgeeks.org/python-program-to-print-emojis/
        print(f"\U0001f607 {TextColor.Bold}{TextColor.BrightGreen_text}TRAINing COMPLETE____________________________{TextColor.End}")
    def single_task_train(self, data_index): # magenta
        print(f"{TextColor.Bold}{TextColor.Magenta_text}SINGLE TASK____________________________{TextColor.End}")
        print(f"we're learning: {self.datas[data_index]}")
        self.cross_validation_loss = []
        for round_index in range(self.number_of_cross_validation_rounds):
            print(f">round {round_index}")
            network_object = self.manager(self.number_of_epochs, round_index, self.settings, self.device)
            valid_loss = network_object.train( # DON'T do so in separate function
                DataLoader(self.dataset(
                self.data_dictionary[self.datas[data_index]]['data'],
                self.data_dictionary[self.datas[data_index]]['label'],
                self.data_dictionary[self.datas[data_index]]['train indices'][round_index],
                device=self.device,), shuffle=False, batch_size=self.settings['batch size']),
                DataLoader(self.dataset(
                self.data_dictionary[self.datas[data_index]]['data'],
                self.data_dictionary[self.datas[data_index]]['label'],
                self.data_dictionary[self.datas[data_index]]['valid indices'][round_index],
                device=self.device,), shuffle=False, batch_size=self.settings['batch size']))
            self.saver(current_loss=valid_loss, round=round_index)
            self.cross_validation_loss.append(valid_loss)
        print(f"{TextColor.Bold}{TextColor.BrightGreen_text}BEST{TextColor.End} model: {self.saver.best_model_name} with {self.saver.current_best_loss}")
        print(f"trained on {self.datas[data_index]}")
        print(f"Aggregate performance: Valid loss mean {np.mean(self.cross_validation_loss)}, std {np.std(self.cross_validation_loss)}")
    def multi_task_train_sequential(self): # blue
        """ learn ONE data at a time, need to reset model in between"""
        print(f"{TextColor.Bold}{TextColor.Blue_text}MULTI TASK, Sequential____________________________{TextColor.End}")
        print(f"we're learning: multiple tasks")
        print(f"given [1, 2, 3], [a, b, c]: learn [1, 2, 3], reset model, learn [a, b, c]")
        self.cross_validation_loss = [[] for data in self.datas]
        for cross_validation_round in range(self.number_of_cross_validation_rounds):
            print(f"CV round {cross_validation_round}_________________________________")
            for index, data in enumerate(self.datas):
                network_object = self.manager(self.number_of_epochs, cross_validation_round, self.settings, self.device)
                print(f"using: {index} {data}")
                valid_loss = network_object.train(
                    DataLoader(self.dataset(
                    self.data_dictionary[data]['data'],
                    self.data_dictionary[data]['label'],
                    self.data_dictionary[data]['train indices'][cross_validation_round],
                    device=self.device,), shuffle=False, batch_size=self.settings['batch size']),
                    DataLoader(self.dataset(
                    self.data_dictionary[data]['data'],
                    self.data_dictionary[data]['label'],
                    self.data_dictionary[data]['valid indices'][cross_validation_round],
                    device=self.device,), shuffle=False, batch_size=self.settings['batch size']))
                network_object.reset_for_sequential()
                self.cross_validation_loss[index].append(valid_loss)
            self.saver(current_loss=valid_loss, round=cross_validation_round)
        print(f"{TextColor.Bold}{TextColor.BrightGreen_text}BEST{TextColor.End} model: {self.saver.best_model_name} with {self.saver.current_best_loss}")
        print(f"trained datas sequentially")
        print(f"Aggregate performance: yo")
        for index, cv_loss in enumerate(self.cross_validation_loss):
            print(f"{self.datas[index]}: Valid loss mean {np.mean(cv_loss)}, std {np.std(cv_loss)}")  
        
    def multi_task_train_weave(self, weave): # blue
        """ learn altogether, using 'super dataset' woven from datasets
            weave: function for weaving"""
        print(f"\U0001f9f5{TextColor.Bold}{TextColor.Magenta_text}MULTI TASK, Interweave____________________________{TextColor.End}")
        print(f"we're learning: multiple tasks")
        print(f"given [1, 2, 3], [a, b, c]: learn [1, a, 2, b, 3, c], simple handling of different counts")
        # for number, data in enumerate(self.datas):
        #     print(f"\t{number}: {data}")
        self.cross_validation_loss = []
        for round_index in range(self.number_of_cross_validation_rounds):
            print(f">round {round_index}")
            network_object = self.manager(self.number_of_epochs, round_index, self.settings, self.device)
            # t = weave([data_dictionary[data]['data'][data_dictionary[data]['train indices'][round_index]] for data in datas])
            # print(f"shape of woven: {t.shape}")
            # i = sum([len(data_dictionary[data]['train indices'][0]) for data in datas])
            # print(f"length of woven: {i}")
            valid_loss = network_object.train( # DON'T do so in separate function
                DataLoader(self.dataset(
                weave([self.data_dictionary[data]['data'][self.data_dictionary[data]['train indices'][round_index]] for data in self.datas]),
                weave([self.data_dictionary[data]['label'][self.data_dictionary[data]['train indices'][round_index]] for data in self.datas]),
                range(sum([len(self.data_dictionary[data]['train indices'][0]) for data in self.datas])),
                device=self.device,), shuffle=False, batch_size=self.settings['batch size']),
                DataLoader(self.dataset(
                weave([self.data_dictionary[data]['data'][self.data_dictionary[data]['valid indices'][round_index]] for data in self.datas]),
                weave([self.data_dictionary[data]['label'][self.data_dictionary[data]['valid indices'][round_index]] for data in self.datas]),                    
                range(sum([len(self.data_dictionary[data]['valid indices'][0]) for data in self.datas])),
                device=self.device,), shuffle=False, batch_size=self.settings['batch size']))    
            self.saver(current_loss=valid_loss, round=round_index)
            self.cross_validation_loss.append(valid_loss)
        print(f"{TextColor.Bold}{TextColor.BrightGreen_text}BEST{TextColor.End} model: {self.saver.best_model_name} with {self.saver.current_best_loss}")
        print(f"trained datas by weaving them")
        print(f"Aggregate performance: Valid loss mean {np.mean(self.cross_validation_loss)}, std {np.std(self.cross_validation_loss)}")
    def test_all(self):
        """ test data one by one"""
        print(f"{TextColor.Bold}{TextColor.Blue_text}TEST____________________________{TextColor.End}")
        retained_loss = {}
        for data in self.datas:
            network_object = self.manager(None, None, self.settings, self.device)
            network_object._network.load_state_dict(torch.load(self.settings['best model folder'] + self.saver.best_model_name))
            print(f"Testing {data}, loss: ", end=" ")
            # print(f"shape: {np.asarray(self.data_dictionary[data]['label'])[0:4]}")
            test_loss = network_object.test(
                DataLoader(self.dataset(
                self.data_dictionary[data]['data'],
                self.data_dictionary[data]['label'],
                self.data_dictionary[data]['test indices'],
                device=self.device,), shuffle=False, batch_size=self.settings['batch size']))
            # record results
            retained_loss[data] = test_loss
            print(f"{test_loss}")
            # print(f"{data} Test loss: {test_loss}")