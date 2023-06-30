import torch
import data_accessor as acc
import random

# add scheduler
# https://arxiv.org/abs/1711.05101
# Adam can substantially benefit from a scheduled learning rate multiplier. 
# The fact that Adam is an adaptive gradient algorithm and as such adapts 
# the learning rate for each parameter does not rule out the possibility to 
# substantially improve its performance by using a global learning rate 
# multiplier, scheduled, e.g., by cosine annealing.
class Scheduler():
    def __init__(self, optimizer, patience, minimum_learning_rate, factor):
        # wait 'patience' number of epochs to change learning rate
        # learning rates' lower bound: 'minimum_learning_rate'
        # update learning rate by 'factor'ArithmeticError
        self.optimizer = optimizer
        self.patience = patience
        self.minimum_learning_rate = minimum_learning_rate
        self.factor = factor
        # use 'min' mode because:
        # we are monitoring loss
        # we do stuff when loss stops DEcreasing
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min', patience=self.patience,
            factor=self.factor, min_lr=self.minimum_learning_rate,
        )
        # print(f"SCHEDULER: {self.scheduler}:")
        # print(f"\tpatience = {self.patience}, factor = {self.factor}" + 
        #       f"minimum_learning_rate = {minimum_learning_rate}")
    def __call__(self, validation_loss):
        self.scheduler.step(validation_loss)

# early stopping (patience step)
# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping
# -with-pytorch/
class EarlyStopping():
    def __init__(self, patience, min_delta):
        # if no improvement after 'patience' epochs, stop training
        # to count as improvement, need to change by 'min_delta' amount
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss >= self.min_delta:
            # improved enough
            self.best_loss = loss
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            # did NOT improve enough :C
            self.counter += 1
            if self.counter >= self.patience:
                # it's stopping time! :C
                # no need reset early_stop, because we only use it once
                self.early_stop = True 
class TaskDataset(torch.utils.data.TensorDataset):
    """ input: input data
        label: label
        indices: indices used e.g. training indices
        """
    def __init__(self, input, label, indices, ratio, size, device):
        # X is already normalized
        self.input = torch.from_numpy(input).to(device)
        self.label = torch.from_numpy(label).to(device)
        self.task_indices = acc.get_task_indices(indices=indices, ratio=ratio, size=size)
        self.indices = range(len(self.task_indices))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        support_indices, query_indices = self.task_indices[index]
        support_input = self.input[support_indices]
        support_label = self.label[support_indices]
        query_input =   self.input[query_indices]
        query_label =   self.label[query_indices] 
        return support_input, support_label, query_input, query_label
class SiameseDataset(torch.utils.data.TensorDataset):
    """ input: input data
        label: label
        indices: indices used e.g. training indices
        """
    def __init__(self, input, label, indices, device):
        # X is already normalized
        self.input = torch.from_numpy(input).to(device)
        self.label = torch.from_numpy(label).to(device)
        self.access_indices = indices
        self.indices = range(len(self.access_indices))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index): # assume index lies within subset_X_IDs
        index = self.access_indices[index]
        other_index = random.choice(self.access_indices)
        input_1 = self.input[index]
        input_2 = self.input[other_index]
        label_1 = self.label[index]
        label_2 = self.label[other_index]
        return input_1, label_1, input_2, label_2