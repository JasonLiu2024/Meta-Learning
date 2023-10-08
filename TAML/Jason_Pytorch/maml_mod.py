"""Implementation of model-agnostic meta-learning for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
from numpy.typing import NDArray
from torch.utils.data import dataset, sampler, dataloader

# import omniglot
import util

NUM_INPUT_CHANNELS = 3
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600

class Network_with_second_back_propagation():
    """
    Builds and Calculates network with second back-propagation
    -
    Functions:
        1 ```build_network()``` : initialize parameters\n
            the parameter is maintained (updated) OUTSIDE this object!
        2 ```forward_propagation()``` : calculates incoming data with parameters\n"""
    def __init__(self, device : str ='cuda' if torch.cuda.is_available() else 'cpu'):
        self._device = device
    def build_network(self,) -> dict[str, torch.Tensor]:
        """
        Makes dictionary of parameters for manual calculation
            This dictionary is maintained OUTSIDE this object
        Returns: 
            1 ```parameter``` (dict[str, torch.Tensor]): the network's parameter"""
        raise NotImplementedError()
    def forward_propagation(self, incoming : torch.Tensor, parameter : dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inputs:
            1 ```incoming``` (torch.Tensor): input to the network 
                (naming collision with Python's 'input').
                shape (Batch, input size)
            2 ```parameter``` (dict[str, torch.Tensor]): parameter to calculate ```incoming```
                follows SAME form as build_network()'s return
        Return:
            1 ```result``` (torch.Tensor) calculated by the network 
                shape (Batch, result size)"""
        raise NotImplementedError()
    
class Network_convolution_then_linear(Network_with_second_back_propagation):
    def __init__(self, output_dimension : int, number_of_convolution_layer : int,
            number_of_input_channel : int, number_of_hidden_channel : int, convolution_kernel_size : int,
            ):
        super().__init__()
        self._output_dimension = output_dimension
        self._number_of_convolution_layer = number_of_convolution_layer
        self._number_of_input_channel = number_of_input_channel
        self._number_of_hidden_channel = number_of_hidden_channel
        self._convolution_kernel_size = convolution_kernel_size
    def build_network(self,) -> dict[str, torch.Tensor]:
        parameter : dict[str, torch.Tensor] = {}
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.
        """
        # build convolutional layer (feature extractor)
        in_channels = self._number_of_input_channel
        for layer in range(self._number_of_convolution_layer):
            parameter[f'convolution_weight{layer}'] = nn.init.xavier_uniform_(
                torch.empty(
                    self._number_of_hidden_channel,
                    in_channels,
                    self._convolution_kernel_size,
                    self._convolution_kernel_size,
                    requires_grad=True,
                    device=self._device))
            parameter[f'convolution_bias{layer}'] = nn.init.zeros_(
                torch.empty(
                    self._number_of_hidden_channel,
                    requires_grad=True,
                    device=self._device))
            in_channels = self._number_of_hidden_channel
        # build linear layer (linear head)
        parameter[f'linear_weight'] = nn.init.xavier_uniform_(
            torch.empty(
                self._output_dimension,
                self._number_of_hidden_channel,
                requires_grad=True,
                device=self._device))
        parameter[f'linear_bias'] = nn.init.zeros_(
            torch.empty(
                self._output_dimension,
                requires_grad=True,
                device=self._device))
        return parameter
    def forward_propagation(self, incoming, parameter):
        x = incoming
        for layer in range(self._number_of_convolution_layer):
            x = F.conv2d(
                input=x,
                weight=parameter[f'convolution_weight{layer}'],
                bias=parameter[f'convolution_bias{layer}'],
                stride=1,
                padding='same'
            )
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        x = torch.mean(x, dim=[2, 3])
        x = F.linear(
            input=x,
            weight=parameter[f'linear_weight'],
            bias=parameter[f'linear_bias'])
        return x

class MAML:
    """Model-Agnostic Meta-Learning network object
    -
    Input:
        1 ```number of inner steps``` (int): steps in inner loop\n
        2 ```inner learning rate``` (float): lr for inner loop\n
        3 ```learn inner learning rate``` (bool): learn inner lr\n
        4 ```outer learning rate``` (float): lr for outer loop\n
        5 ```network``` (Network_with_second_back_propagation): network to build model 
            and forward-propagate
    """
    def __init__(
            self,  
            number_of_inner_steps : int, 
            inner_learning_rate : float, learn_inner_learning_rate : bool, 
            outer_learning_rate : float, 
            network : Network_with_second_back_propagation
    ):
        self._network : Network_with_second_back_propagation = network
        self._meta_parameters = self._network.build_network()
        self._number_of_inner_steps = number_of_inner_steps
        self._inner_learning_rate : dict[str, torch.Tensor]= {
            k: torch.tensor(inner_learning_rate, requires_grad=learn_inner_learning_rate)
            for k in self._meta_parameters.keys()}
        self._outer_learning_rate = outer_learning_rate
        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_learning_rate.values()),
            lr=self._outer_learning_rate)
        self._start_train_step = 0

    def _inner_loop(self, support_input : torch.Tensor, support_label : torch.Tensor, we_are_training : bool) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Adapts network parameters to ONE task
        -
        Inputs:
            1 ```support_input``` (Tensor): task support set inputs
                shape (number of images, channels, height, width)
            2 ```support_label``` (Tensor): task support set labels
                shape (number of images,)
            3 ```we_are_training``` (bool): whether we are training or evaluating
                received from ```_outer_step()```
        Returns:
            1 ```parameter``` (dict[str, Tensor]): adapted network parameters.\n
            2 ```accuracy_list``` (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
        """
        accuracy_list = []
        cloned_parameter = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        # This method computes the inner loop (adaptation) procedure
        # over the course of _num_inner_steps steps for one
        # task. It also scores the model along the way.
        # Use F.cross_entropy to compute classification losses.
        # Use util.score to compute accuracies.
        for _ in range(self._number_of_inner_steps):
            # Forward propagation to obtain y_support
            y_support = self._network.forward_propagation(incoming=support_input, parameter=cloned_parameter)
            # get loss
            inner_loss = F.cross_entropy(y_support, support_label)
            # get accuracy
            accuracy = util.score(y_support, support_label)
            accuracy_list.append(accuracy)
            # get gradients (for EACH layer)
            grads_list = autograd.grad(inner_loss, cloned_parameter.values(), create_graph=we_are_training)
            for (layer, layer_name) in enumerate(cloned_parameter.keys()):
                cloned_parameter[layer_name] = cloned_parameter[layer_name] - self._inner_learning_rate[layer_name] * grads_list[layer]
        # post-adaptation accuracy <- note that we're NOT concerned with the loss!
        y_support = self._network.forward_propagation(incoming=support_input, parameter=cloned_parameter)
        post_adapt_accuracy = util.score(y_support, support_label)
        accuracy_list.append(post_adapt_accuracy)
        return cloned_parameter, accuracy_list

    def _outer_step(self, task_batch : tuple[tuple[torch.Tensor, ...], ...], we_are_training : bool):
        """Get Loss from BATCH of Tasks.
        -
        Inputs:
            1 ```task batch``` (tuple): batch of tasks from an Omniglot DataLoader
                each task is an (images support, labels support, images query, labels query)!
            2 ```train``` (bool): whether we are training or evaluating
        Returns:
            1 ```outer loss``` (Tensor): mean MAML loss over the batch, 
                Scalar.
            2 ```accuracies support``` (ndarray): support set accuracy over inner loop steps, 
                averaged over the task batch dimension.
                shape (number of inner steps + 1,)
            3 ```accuracy query``` (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch_list = []
        accuracies_support_batch_list = []
        accuracy_query_batch_list = []
        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(self._network._device)
            labels_support = labels_support.to(self._network._device)
            images_query = images_query.to(self._network._device)
            labels_query = labels_query.to(self._network._device)
            # For a given task, use the _inner_loop method to adapt for
            # _num_inner_steps steps, then compute the MAML loss and other metrics.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            adapted_parameters, accuracy_support_over_inner_steps = self._inner_loop(support_input=images_support, support_label=labels_support, we_are_training=we_are_training)
            predicted_labels_query = self._network.forward_propagation(incoming=images_query, parameter=adapted_parameters)
            # get loss
            outer_loss_batch = F.cross_entropy(predicted_labels_query, labels_query)
            outer_loss_batch_list.append(outer_loss_batch) # NOT .item() because we need to back propagate!
            # get accuracy
            accuracies_support_batch_list.append(accuracy_support_over_inner_steps)
            accuracy_query_batch = util.score(predicted_labels_query, labels_query)
            accuracy_query_batch_list.append(accuracy_query_batch)
        outer_loss = torch.mean(torch.stack(outer_loss_batch_list))
        accuracies_support = np.mean(accuracies_support_batch_list, axis=0) # don't need back-propagate on accuracy
        accuracy_query = np.mean(accuracy_query_batch_list)
        return outer_loss, accuracies_support, accuracy_query

    def train(self, dataloader_train : dataloader.DataLoader, dataloader_val : dataloader.DataLoader):
        """Train the MAML.
        -
        1. Optimizes MAML parameter with ```dataloader train``` 
        2. Periodically Validate on ```dataloader val```, logging metrics, and
        saving checkpoints.

        Inputs:
            1 ```dataloader train``` (DataLoader): loader for train tasks\n
            2 ```dataloader val``` (DataLoader): loader for validation tasks"""
        print(f'Starting training at iteration {self._start_train_step}.')
        loss_list = []
        accuracy_list = []
        for i_step, task_batch in enumerate(dataloader_train, start=self._start_train_step):
            self._optimizer.zero_grad()
            # 1. Optimize MAML parameter with dataloader_train
            outer_loss, accuracies_support, accuracy_query = self._outer_step(task_batch, we_are_training=True)
            outer_loss.backward()
            self._optimizer.step()
            # write down loss and accuracy
            loss_list.append(outer_loss.item())
            accuracy_list.append(accuracy_query)
            # 2. Periodically validate on dataloader_val
            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_query:.3f}'
                )

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch in dataloader_val:
                    outer_loss, accuracies_support, accuracy_query = self._outer_step(val_task_batch, we_are_training=False)
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(accuracies_pre_adapt_support)
                accuracy_post_adapt_support = np.mean(accuracies_post_adapt_support)
                accuracy_post_adapt_query = np.mean(accuracies_post_adapt_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracy_pre_adapt_support:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}')

    def test(self, dataloader_test : dataloader.DataLoader):
        """Evaluate the MAML on test tasks.
        -
        Inputs:
            1 ```dataloader test``` (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = self._outer_step(task_batch, we_are_training=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}')