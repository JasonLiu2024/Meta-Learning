import torch
import torch.nn as nn
from TCN import *
"""Triplet network + Auxiliary network"""
class TripletAux_and_TCNNetwork(torch.nn.Module):
    """ Input: pos, neg, anchor, anchor_label
        Output: pos_prediction, neg_prediction"""
    def __init__(self, device, input_dimension, input_channels, feature_dimension, output_dimension,
                 convolution_levels_ct=3):
        super().__init__()
        self.input_dimension = input_dimension
        self.feature_dimension = feature_dimension
        self.output_dimension = output_dimension
        self.input_channels = input_channels
        self.convolution_levels_ct = convolution_levels_ct
        self.device = device
        # conv1d from TemporalConvNet expects input shape=(batch size, channel ct, length of signal)
        # here, (channel ct, length of signal) is the shape of our input
        # print(f"[ct for ct in range(self.convolution_levels_ct)]: {type([self.input_channels for ct in range(self.convolution_levels_ct)])}")
        self.tcn = TemporalConvNet(input_channels=self.input_channels, 
                                   output_channels_list=[self.input_channels for ct in range(self.convolution_levels_ct)], 
                                   kernel_size=3, dropout=0)
        # self.feature_sequential = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_dimension, 3000),
        #     nn.ReLU(),
        #     torch.nn.Linear(3000, 600),
        #     nn.ReLU(),
        #     torch.nn.Linear(600, 600),
        #     nn.ReLU(),
        #     torch.nn.Linear(600, 300),
        #     nn.ReLU(),
        #     torch.nn.Linear(300, self.feature_dimension)
        # )
        self.feature_sequential = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension, 3000),
            nn.ReLU(),
            torch.nn.Linear(3000, 900),
            nn.ReLU(),
            torch.nn.Linear(900, 300),
            nn.ReLU(),
            torch.nn.Linear(300, self.feature_dimension),
            # nn.ReLU(),
        )
        self.auxiliary_sequential = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dimension, 100),
            nn.ReLU(),
            torch.nn.Linear(100, 100),
            nn.ReLU(),
            torch.nn.Linear(100, self.output_dimension)
        )
        self.to(device)
        self.float()
    def forward(self, pos, neg, anchor, anchor_label):
        convolved_pos = self.tcn(pos)
        convolved_neg = self.tcn(neg)
        convolved_anchor = self.tcn(anchor)
        feature_pos = self.feature_sequential(convolved_pos)
        feature_neg = self.feature_sequential(convolved_neg)
        feature_anchor = self.feature_sequential(convolved_anchor)
        feature_space_difference_pos_anchor = feature_pos - feature_anchor
        feature_space_difference_neg_anchor = feature_neg - feature_anchor
        label_space_difference_pos_anchor = self.auxiliary_sequential(feature_space_difference_pos_anchor)
        label_space_difference_neg_anchor = self.auxiliary_sequential(feature_space_difference_neg_anchor)
        prediction_pos = anchor_label + label_space_difference_pos_anchor
        prediction_neg = anchor_label + label_space_difference_neg_anchor
        return prediction_pos, prediction_neg
    
import numpy as np
from tools import SaveBestModel, PatienceEarlyStopping, Scheduler, plot_losses
class TripletAux_and_TCNManager:
    """ DOES: train & evaluate a Triplet_and_TCN network
        """
    def __init__(self, epoch, cross_validation_round, setting, device):
        # self._network = SingleTaskNetwork(device, s['input dimension'], s['feature dimension'], s['output dimension'])
        self._network = TripletAux_and_TCNNetwork(device, setting['input dimension'], setting['input channels'], 
            setting['feature dimension'], setting['output dimension'], setting['number of convolution levels'])
        self._network.apply(self.initializer)
        self._learning_rate = setting['learning rate']
        self._optimizer = torch.optim.Adam(
            params=self._network.parameters(), lr=self._learning_rate,
            weight_decay=3e-3)
        self._energy = nn.MSELoss()
        self._train_loss = []
        self._valid_loss = []
        self._test_loss = []
        self._epoch = epoch
        self._stopper = PatienceEarlyStopping(patience=5, min_delta=1e-7)
        self._cross_validation_round = cross_validation_round
        self._saver = SaveBestModel(setting['best model folder'])
        self._scheduler = Scheduler(optimizer=self._optimizer, 
            minimum_learning_rate=1e-6, patience=5, factor=0.5)
    def initializer(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight) # normal version
    def _step(self, job):
        pos, pos_label, neg, neg_label, anchor, anchor_label = job
        pos_prediction, neg_prediction = self._network(pos, neg, anchor, anchor_label)
        pos_loss = self._energy(pos_prediction, pos_label)
        neg_loss = self._energy(neg_prediction, neg_label)
        loss = (pos_loss + neg_loss) / 2.0
        return loss     
    def train(self, train_dataloader, valid_dataloader):
        """ DOES: calculate loss from tasks
            NOTE: we have a BATCH of tasks here """
        for e in range(self._epoch):
            # print(f"train() epoch {e}")
            batch_train_loss = []
            for _, batch in enumerate(train_dataloader): 
                self._optimizer.zero_grad()
                loss = self._step(batch)
                loss.backward()
                self._optimizer.step()
                batch_train_loss.append(loss.item())
            self._train_loss.append(np.mean(batch_train_loss))
            batch_valid_loss = []
            with torch.no_grad():
                for _, batch in enumerate(valid_dataloader): 
                    loss = self._step(batch)
                    batch_valid_loss.append(loss.item())
            self._valid_loss.append(np.mean(batch_valid_loss))
            # saving, early stopping, scheduler for EACH epoch!
            self._saver(current_loss=np.mean(batch_valid_loss), 
                  model=self._network, 
                  round=self._cross_validation_round
                  )
            self._scheduler(np.mean(batch_valid_loss))
            self._stopper(np.mean(batch_valid_loss))
            if self._stopper.early_stop == True:
                print(f"EARLY STOPPING @ epoch {e}")
                break
        # summary printout, after we're done with epochs
        print(f"min train loss: {np.min(self._train_loss)}")
        print(f"min valid loss: {np.min(self._valid_loss)}")
        plot_losses(self._train_loss, self._valid_loss, self._cross_validation_round)
        return np.min(self._valid_loss)
    def test(self, test_dataloader):
        with torch.no_grad():
            batch_test_loss = []
            for _, batch in enumerate(test_dataloader): 
                loss = self._step(batch)
                batch_test_loss.append(loss.item())
            self._test_loss.append(np.mean(batch_test_loss)) 
        return np.min(self._test_loss)
    def reset_for_sequential(self):
        self._saver.reset()
        self._stopper.reset()
        self._train_loss = []
        self._valid_loss = []
        # reset auxiliary network
        self._network.auxiliary_sequential.apply(self.initializer)

import random
# 'utils' having red wavy line in 'torch.utils.data.TensorDataset' is BUG
class TripletAux_and_TCNDataset(torch.utils.data.TensorDataset):
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
        anchor_index = random.choice(self.access_indices)
        neg_index = random.choice(self.access_indices)
        pos = self.input[index]
        pos_label = self.label[index]
        anchor = self.input[anchor_index]
        anchor_label = self.label[anchor_index]
        neg = self.input[neg_index]
        neg_label = self.label[neg_index]
        # adding dummy dimension for 'channel'
        return pos[None, :], pos_label[None, :], neg[None, :], neg_label[None, :], anchor[None, :], anchor_label[None, :]