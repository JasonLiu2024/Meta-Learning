"""Implementation of Bayesian Task-Adaptive Meta-Learning algorithm (Bayesian TAML).
TAML is an augmentation of MAML"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data.dataloader import DataLoader
from encoder import InferenceNetwork

NUM_INPUT_CHANNELS = 3
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_INTERVAL = 8
NUM_TEST_TASKS = 600 # not sure what this is doing!

class TAML:
    """TAML trainer object.
    
    output_dimension:       here's it's 'way,' number of classes per task
    max_shots:              maximum number of shots (there can be variable # of shots!)
    inner_loop_step_ct:     number of steps in _inner_loop
    inner_learning_rate:    learning rate for _inner_loop
    learn_inner_learning_rate:  learn the above parameter
    outer_learning_rate:    learning rate for _outer_step
    batch_size:             number of Tasks per batch (NOT examples!)
    """

    def __init__(self, output_dimension : int, inner_loop_step_ct : int, 
        inner_learning_rate : float, learn_inner_learning_rate : bool, 
        outer_learning_rate : float, batch_size : int, encoder : InferenceNetwork,
        ):
        # number of tasks per batch
        self.batch_size = batch_size
        # encoder network
        self.encoder : InferenceNetwork = encoder.to(DEVICE)
        # encoder network's balancing variables
        # parameters (initialization)
        meta_parameters : dict[str, torch.Tensor] = {}
        """model architecture:
        1 feature extractor:    4 convolutional layers
        2 linear layer:         1 linear layer
        """
        # construct feature extractor
        in_channels : int = NUM_INPUT_CHANNELS
        for i in range(NUM_CONV_LAYERS):
            meta_parameters[f'convolution_weight_{i}'] = nn.init.xavier_uniform_(
                torch.empty(
                    NUM_HIDDEN_CHANNELS,
                    in_channels,
                    KERNEL_SIZE,
                    KERNEL_SIZE,
                    requires_grad=True,
                    device=DEVICE
                )
            )
            meta_parameters[f'convolution_bias_{i}'] = nn.init.zeros_(
                torch.empty(
                    NUM_HIDDEN_CHANNELS,
                    requires_grad=True,
                    device=DEVICE
                )
            )
            in_channels = NUM_HIDDEN_CHANNELS
        # construct linear head layer
        meta_parameters[f'linear_weight_{NUM_CONV_LAYERS}'] = nn.init.xavier_uniform_(
            torch.empty(
                output_dimension,
                NUM_HIDDEN_CHANNELS,
                requires_grad=True,
                device=DEVICE
            )
        )
        meta_parameters[f'linear_bias_{NUM_CONV_LAYERS}'] = nn.init.zeros_(
            torch.empty(
                output_dimension,
                requires_grad=True,
                device=DEVICE
            )
        )
        self._meta_parameters = meta_parameters
        self._num_inner_steps = inner_loop_step_ct
        self._inner_lrs = {
            k: torch.tensor(inner_learning_rate, requires_grad=learn_inner_learning_rate)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_learning_rate
        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._start_train_step = 0 # useful to picking-up on training
    def _view_model(self, parameters : dict[str, torch.Tensor]) -> None:
        """Displays the shape of model parameters"""
        print(f"model architecture:")
        for layer_name, layer_tensor in parameters.items():
            print(f"layer: {layer_name}")
            print(f"\tshape: {layer_tensor.shape}")
    def _forward(self, images : torch.Tensor, parameters : dict[str, torch.Tensor]) -> torch.Tensor:
        """Gets prediction of classes (they're logits)

        images:     batch of images
        parameters: parameters to compute the prediction
        RETURN:     
            batch of logits (one for ea image!)

        Note: "Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive." MAML author
        """
        # print(parameters.keys())
        x = images
        for i in range(NUM_CONV_LAYERS):
            x = F.conv2d(
                input=x,
                weight=parameters[f'convolution_weight_{i}'],
                bias=parameters[f'convolution_bias_{i}'],
                stride=1,
                padding='same'
            )
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        x = torch.mean(x, dim=[2, 3]) # average out the height and width dimensions!
        return F.linear(
            input=x,
            weight=parameters[f'linear_weight_{NUM_CONV_LAYERS}'],
            bias=parameters[f'linear_bias_{NUM_CONV_LAYERS}']
        )
    def _omega_modify_inner_loss(self, 
        omega: torch.Tensor, inner_loop_loss : torch.Tensor) -> torch.Tensor:
        """change task-specific loss with omega (class-level balancing varaible)"""
        return torch.sum(inner_loop_loss * F.softmax(omega, -1))
    def _zeta_update_initialization(self, zeta : dict[str, torch.Tensor], theta : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """update theta with zeta (task-level balancing variable)"""
        for layer_key in zeta.keys():
            # Specifically, we use z_train = 1 + z _test convolutional network weights 
            # ... z_train = z _test for the biases
            # which modifies the network parameters θ like: (let θ_0 be NEW initializtion)
            if '_weight' in layer_key:
                # θ_0 = θ + zeta
                theta[layer_key] = theta[layer_key] * (1. + zeta[layer_key])
            else: # '_b' in layer_key:
                # θ_0 = θ o zeta (o is custom operator, see below)
                theta[layer_key] = theta[layer_key] + zeta[layer_key]
        return theta
    def _gamma_modify_inner_step_learning_rate_updater(self, gamma_tensor : torch.Tensor, updater : torch.Tensor) -> torch.Tensor:
        """change learning rate with gamma (task-level balancing variable)"""
        # TAML author's name for 'updater' is 'delta'
        return updater * torch.exp(gamma_tensor)
    def _inner_loop(self, images_suppt : torch.Tensor, labels : torch.Tensor, train : bool, 
        # omega, zeta, gamma here are TAML's extra!
        omega : torch.Tensor, zeta : dict[str, torch.Tensor], gamma : dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor], list[float]]:
        """Adapt parameters to a Task(this is the MAML's inner loop)-ONLY for support set!

        images: support set inputs
        labels: support set outputs
        train:  (from _outer_step()) Yes=we're training, No=we're evaluating
        RETURN: 
            adapted parameters
            support set accuracies (shape=(inner loop steps + 1))
        """
        accuracy_suppt : list[float] = []
        parameters : dict[str, torch.Tensor] = { # clone parameters to update them
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        """TAML: update initialization with zeta"""
        if self.encoder.use_z == True:
            parameters = self._zeta_update_initialization(zeta, parameters)
        for _ in range(self._num_inner_steps):
            predictions = self._forward(images_suppt, parameters)
            accuracy = score(predictions, labels)
            accuracy_suppt.append(accuracy)
            loss = F.cross_entropy(predictions, labels.to(torch.int64)) # just one number!
            """TAML: update loss with omega; doens't ruin .backward!"""
            if self.encoder.use_o == True:
                loss = self._omega_modify_inner_loss(omega, loss)
            # get grads, the gradient of the loss
            # we're Always doing a backpropagation here, since we need it to adapt to the current task
            # BUT:
            #   training -> create graph because we're reusing it when loss.backward outside!
            #   evaluating -> NO create graph, we're not doing backward
            # Do NOT use loss.backward(retain_graph=True), it's NEVER correct (and wastes memory)
            # Use work-around, such as autograd here below!
            # we want to get our hands on the gradients, so we can update it
            # this is why we use .grad (.backward() just populates tensors' .grad field)
            # https://discuss.pytorch.org/t/whats-the-difference-between-torch-autograd-grad-and-backward/94638
            grads = torch.autograd.grad( # type-check complains, but this works
                loss, parameters.values(), create_graph=train)
            # update parameters (manually)
            for (layer_name, tensor), grad in zip(parameters.items(), grads):
                "TAML: update inner loop learning rate with gamma"
                if self.encoder.use_g == True:
                    parameters[layer_name] = tensor - self._gamma_modify_inner_step_learning_rate_updater(
                        gamma[layer_name], updater=self._inner_lrs[layer_name] * grad)
                # gamma modifies the 'updater,' which is self._inner_lrs[layer_name] * grad (parameters subtract that thing!)
                else:  
                    parameters[layer_name] = tensor - self._inner_lrs[layer_name] * grad # original no-gamma version!
        return parameters, accuracy_suppt

    def _outer_step(self, task_batch : tuple[torch.Tensor], train : bool) -> tuple[
        torch.Tensor, NDArray, float
    ]:  # pylint: disable=unused-argument
        """Get Loss (and metrics) on Batch of Tasks (ea Task dealt by _inner_loop())

        task_batch: batch of Tasks (from dataloader)
        train:      Yes=we're training, No=we're evaluating   
        RETURN:
            outer_loss:           (ONE value) MAML loss over batch (scalar value)
            accuracy_suppt_batch: (LOTs of values) support set accuracy (shape=(inner loop steps + 1))
            accuracy_query_batch: (ONE value) query set accuracy (average over entire batch of tasks)
        """
        loss_batch_list : list[torch.Tensor]= []
        accuracy_suppt_batch_list : list[list[float]]= []
        accuracy_query_batch_list : list[float]= []
        for task in task_batch:
            images_suppt, labels_suppt, images_query, labels_query = task
            # print(images_suppt.shape)
            # print(labels_suppt.shape)
            # print(images_query.shape)
            # print(labels_query.shape)
            """encoder extracts balancing varaibles for each task"""
            # getting these by themselves do NOT affect .backward()
            omega, gamma, zeta, KL = self.encoder(images_suppt, labels_suppt, do_sample=train)
            Cardinality_train = images_suppt.shape[0]
            Cardinality_test = images_query.shape[0]
            # scale KL term w.r.t train and test set sizes
            KL /= (Cardinality_train + Cardinality_test)
            # adapt parameters from support set
            """inner loop start"""
            parameters, accuracy_suppt = self._inner_loop(
                images_suppt, labels_suppt, train, omega, zeta, gamma)
            # make predictions, using those updated parameters
            predictions = self._forward(images_query, parameters)
            # the loss is just one number (scalar value)
            loss = F.cross_entropy(predictions, labels_query.to(torch.int64))
            accuracy_query = score(predictions, labels_query)
            loss_batch_list.append(loss)
            accuracy_suppt_batch_list.append(accuracy_suppt)
            accuracy_query_batch_list.append(accuracy_query)
        loss_batch : torch.Tensor = torch.mean(torch.stack(loss_batch_list)) 
        accuracy_suppt_batch : NDArray = np.mean(accuracy_suppt_batch_list, axis=0) # shape=(batch_size, inner loop steps + 1)
        accuracy_query_batch : float = np.mean(accuracy_query_batch_list)
        return loss_batch, accuracy_suppt_batch, accuracy_query_batch
    def train(self, dataloader_train : DataLoader, dataloader_valid : DataLoader, valid_interval : int) -> tuple[
        list[float], list[float], list[float], list[float],]:
        """Optimize parameters; validate once in a while
        
        RETURN:
            train loss              list
            train accuracy (query)  list
            valid loss              list
            valid accuracy (query)  list"""
        print(f"It's Training time! (start from iteration {self._start_train_step})")
        all_train_loss_list : list[float] = []
        all_train_accuracy_list : list[float] = []
        all_valid_loss_list : list[float] = []
        all_valid_accuracy_list : list[float] = []
        for iteration, train_task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step # in case we're picking up from somewhere
        ):
            self._optimizer.zero_grad()
            train_loss_batch, train_accuracy_suppt, train_accuracy_query = (
                self._outer_step(train_task_batch, train=True) # Yes training
            )
            train_loss_batch.backward()
            self._optimizer.step()
            # record
            all_train_loss_list.append(train_loss_batch.item())
            all_train_accuracy_list.append(train_accuracy_query)
            # validate once in a while!
            if iteration % valid_interval == 0:
                print(f"\tIt's Validating time! iteration {iteration}")
                valid_loss_list = []
                valid_accuracy_suppt_thn_list = [] # accuracy before adaptation (first inner loop step)
                valid_accuracy_suppt_now_list = [] # accuracy after adaptation (last inner loop step)
                # we deal with query set AFTER working on support set, 
                # so there's NO pre-adapt support!
                valid_accuracy_query_list = []
                for valid_task_batch in dataloader_valid:
                    valid_loss_batch, valid_accuracy_suppt, valid_accuracy_query = (
                        self._outer_step(valid_task_batch, train=False) # NO training
                    )
                    valid_loss_list.append(valid_loss_batch.item())
                    valid_accuracy_suppt_thn_list.append(valid_accuracy_suppt[0])
                    valid_accuracy_suppt_now_list.append(valid_accuracy_suppt[-1])
                    valid_accuracy_query_list.append(valid_accuracy_query)
                    # record:
                    # print(f"all valid loss, appended: {valid_loss_batch.item()}")
                    all_valid_loss_list.append(valid_loss_batch.item())
                    all_valid_accuracy_list.append(valid_accuracy_query)
                # average out losses over the many Batches
                valid_loss = np.mean(valid_loss_list)
                valid_accuracy_suppt_thn = np.mean(valid_accuracy_suppt_thn_list)
                valid_accuracy_suppt_now = np.mean(valid_accuracy_suppt_now_list)
                valid_accuracy_query = np.mean(valid_accuracy_query_list)
                # print(
                #     f'\t validation loss:                          '
                #     f'{valid_loss:.3f}'
                #     f'\n\t average support accuracy before adapting: '
                #     f'{valid_accuracy_suppt_thn:.3f}'
                #     f'\n\t average support accuracy after adapting:  '
                #     f'{valid_accuracy_suppt_now:.3f}'
                #     f'\n\t average query accuracy:                   '
                #     f'{valid_accuracy_query:.3f}'
                # )
        print(f"best train loss:     {np.min(all_train_loss_list)}")
        print(f"best train accuracy: {np.max(all_train_accuracy_list)}")
        print(f"best valid loss:     {np.min(all_valid_loss_list)}")
        print(f"best valid accuracy: {np.max(all_valid_accuracy_list)}")
        return all_train_loss_list, all_train_accuracy_list, all_valid_loss_list, all_valid_accuracy_list
    def test(self, dataloader_test : DataLoader) -> tuple[
        float, float,]:
        """Test
        
        RETURN:
            test loss               minimum
            test accuracy (query)   minimum"""
        print(f"It's Testing time!")
        all_test_loss_list : list[float] = []
        all_test_accuracy_list : list[float] = []
        accuracies = []
        for task_batch in dataloader_test:
            test_loss_batch, test_accuracy_suppt, test_accuracy_query = self._outer_step(
                task_batch, train=False) # NO training!
            accuracies.append(test_accuracy_query)
            # record:
            all_test_loss_list.append(test_loss_batch.item())
            all_test_accuracy_list.append(test_accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print(f"best test loss:     {np.min(all_test_loss_list)}")
        print(f"best test accuracy: {np.max(all_test_accuracy_list)}")
        return np.min(all_test_loss_list), np.min(all_test_accuracy_list)

def score(logits : torch.Tensor, labels : torch.Tensor):
    """mean accuracy of a model's predictions on a set of examples.
        
        logits: model prediction (logits form), shape=(# examples, # classes)
        labels: classification labels from 0 to num_classes - 1, shape=(# examples,)
    """
    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    # print(f"score() logits: {logits.shape}")
    # print(f"score() labels: {labels.shape} \n\t{labels}")
    # print(f"score() predicts: {torch.argmax(logits, dim=-1).shape} \n\t{torch.argmax(logits, dim=-1)}")
    return torch.mean(y).item()