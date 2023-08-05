import torch
import data_accessor as acc
import random
import matplotlib.pyplot as plt

class SaveBestModel():
    def __init__(self, folder_address, current_best_loss = float('inf')):
        self.current_best_loss = current_best_loss
        self.save = False
        self.folder = folder_address
    def __call__(self, current_loss, model, round):
        # no save optimizer, since we are using model for inference
        if current_loss < self.current_best_loss:
            self.current_best_loss = current_loss
            self.save = True
        if self.save == True:
            torch.save(model.state_dict(), self.folder + f'CV={round}' + '.pth')
    def reset(self):
        self.current_best_loss = float('inf')

class SaveBestCrossValidationModel():
    def __init__(self, folder_address, current_best_loss = float('inf')):
        self.current_best_loss = current_best_loss
        self.save = False
        self.folder = folder_address
        self.best_model_name = None
    def __call__(self, current_loss, round):
        # no save optimizer, since we are using model for inference
        if current_loss < self.current_best_loss:
            # print(f"current loss {current_loss} < {self.current_best_loss}")
            self.current_best_loss = current_loss
            self.best_model_name = f'CV={round}' + '.pth'
    def reset(self):
        self.current_best_loss = float('inf')

class Scheduler():
    def __init__(self, optimizer, patience, minimum_learning_rate, factor):
        # wait 'patience' number of epochs to change learning rate
        # learning rates' lower bound: 'minimum_learning_rate'
        # update learning rate by 'factor'
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

class PatienceEarlyStopping():
    def __init__(self, patience, min_delta):
        """ if no improvement after 'patience' epochs, stop training
        to count as improvement, need to change by 'min_delta' amount"""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss >= self.min_delta: # improved enough
            self.best_loss = loss
            self.counter = 0
        elif self.best_loss - loss < self.min_delta: # did NOT improve enough :C
            self.counter += 1
            if self.counter >= self.patience: # it's stopping time! :C
                # no need reset early_stop, because we only use it once
                self.early_stop = True 
    def reset(self):
        self.best_loss = float('inf')
        self.early_stop = False

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
        self.input = torch.Tensor(input).to(device)
        self.label = torch.Tensor(label).to(device)
        self.access_indices = indices
        self.indices = range(len(self.access_indices))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index): 
        index = self.access_indices[index]
        other_index = random.choice(self.access_indices)
        input_1 = self.input[index]
        input_2 = self.input[other_index]
        label_1 = self.label[index]
        label_2 = self.label[other_index]
        return input_1.float(), label_1.float(), input_2.float(), label_2.float()
    
import matplotlib.pyplot as plt
def plot_loss(train_loss, valid_loss):
    """ OUTDATED
        plot TRAIN alongside VALID loss"""
    plt.figure(figsize=(4.8, 4.8))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot()
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.legend(["train loss", "valid loss"], loc ="upper right")
    plt.show()

def plot_losses(train_loss, valid_loss, CV_round):
        """UPDATED"""
        f, ax = plt.subplots(figsize=(6, 4.5)) # NOT subplot()
        train_loss_color = 'blue'
        valid_loss_color = 'green'
        plt.plot(train_loss, '-', color=train_loss_color)
        plt.plot(valid_loss, '-', color=valid_loss_color)
        FONT = {'fontname':'Ubuntu'}
        ax.text(0.99, 0.90, 'train loss',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color=train_loss_color, fontsize=16, **FONT)
        ax.text(0.99, 0.80, 'valid loss',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color=valid_loss_color, fontsize=16, **FONT)
        ax.text(0.01, 0.90, f'CV round: {CV_round}',
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                color='black', fontsize=16, **FONT)
        plt.show()

class DefaultDataset(torch.utils.data.TensorDataset):
    """ input: input data
        label: label
        indices: indices used e.g. training indices
        """
    def __init__(self, input, label, indices, device):
        self.input = torch.Tensor(input).to(device)
        self.label = torch.Tensor(label).to(device)
        self.access_indices = indices
        self.indices = range(len(self.access_indices))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index): 
        index = self.access_indices[index]
        input = self.input[index]
        label = self.label[index]
        return input.float(), label.float()
  