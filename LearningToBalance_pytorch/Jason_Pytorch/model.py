import torch
import numpy as np
from layers import *
from encoder import InferenceNetwork
from torch import autograd

class LearningToBalance():
  def __init__(self, data_name : str, number_of_inner_gradient_steps : int,
               ways : int, shots : int, metabatch, inner_learning_rate : float):
    if data_name == 'cifar':
      self.xdim = 32
      self.input_channel = 3
      self.number_of_channels = 32
    elif data_name == 'mimgnet':
      self.xdim = 84
      self.input_channel = 3 
      self.number_of_channels = 32
    else:
      raise ValueError("Invalid dataset name")
    
    self.number_of_inner_gradient_steps = number_of_inner_gradient_steps
    self.number_of_classes = ways
    self.metabatch = metabatch
    self.inner_lr = inner_learning_rate

    self.encoder = InferenceNetwork(ways, shots, data_name, True, True, True)
    self.use_inner_step_size_vector = True # learnable, from Meta-SGD

    self.use_zeta = True
    self.use_gamma = True
    self.use_omega = True
    self.use_alpha = True

  def get_parameter(self, name : str) -> dict[str, torch.Tensor]:
    """name: name of parameter in {theta, alpha}"""
    if name == 'theta':
      init_convolution  = lambda x : torch.nn.init.trunc_normal_(x, std=0.02)
      init_bias         = lambda x : torch.nn.init.zeros_(x)
      init_dense        = lambda x : torch.nn.init.normal_(x, std=0.02)
    else: # name == 'alpha'
      init_convolution  = lambda x : torch.nn.init.constant_(x, 0.01)
      init_bias         = lambda x : torch.nn.init.constant_(x, 0.01)
      init_dense        = lambda x : torch.nn.init.constant_(x, 0.01)
    parameter_dictionary : dict[str, torch.Tensor]= {}
    for l in [1, 2, 3, 4]:
      channels = self.input_channel if l == 1 else self.number_of_channels
      weight = torch.Tensor(size=[3, 3, channels, self.number_of_channels])
      parameter_dictionary[f'convolution_{l}_weight'] = init_convolution(weight)
      bias = torch.Tensor(size=[self.number_of_channels])
      parameter_dictionary[f'convolution_{l}_bias'] = init_bias(bias)

    remaining = 2 * 2 if self.xdim == 32 else 5 * 5
    if name == 'theta':
      parameter_dictionary[f'dense_weight'] = torch.zeros(size=[remaining * self.number_of_channels, self.number_of_classes])
      parameter_dictionary[f'dense_bias'] = torch.zeros(size=[self.number_of_classes])
    else:
      single_weight = init_dense(torch.Tensor(size=[remaining * self.number_of_channels, 1]))
      single_bias = init_bias(torch.Tensor(size=[1]))
      parameter_dictionary[f'dense_weight'] = torch.tile(single_weight, [1, self.number_of_classes])
      parameter_dictionary[f'dense_bias'] = torch.tile(single_bias, [self.number_of_classes])
    return parameter_dictionary

  def forward_theta(self, x : torch.Tensor, theta : dict[str, torch.Tensor]):
    x = torch.reshape(x, [-1, self.input_channel, self.xdim, self.xdim])
    for l in [1, 2, 3, 4]:
      weight = theta[f'convolution_{l}_weight']
      bias = theta[f'convolution_{l}_bias']
      x = ConvolutionBlock_F(x, weight, bias)
    weight = theta[f'dense_weight']
    bias = theta[f'dense_bias']
    x = DenseBlock_F(x, weight, bias)
    return x
  def forward_outer_objective(self, input : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    do_sample=True):
    """outer objective w.r.t. the task
    \nthe meta-learning goal is:
    \n\tminimize sum of loss of task over all tasks
    \n\twhere loss of task is """
    theta = self.get_parameter('theta')
    if self.use_alpha: 
      alpha = self.get_parameter('alpha')
    x_train, y_train, x_test, y_test = input
    omega, gamma, zeta, KL = self.encoder.forward(x_train, y_train, True)
    Cardinality_train = x_train.shape[0]
    Cardinality_test = x_test.shape[0]
    # scale KL term w.r.t train and test set sizes
    KL /= (Cardinality_train + Cardinality_test)

    """zeta: modulate MAML initialization"""
    if self.use_zeta:
      theta_update_by_zeta = {}
      for layer_key in zeta.keys():
        # Specifically, we use z_train = 1 + z _test convolutional network weights 
        # ... z_train = z _test for the biases
        # which modifies the network parameters θ like: (let θ_0 be NEW initializtion)
        if '_w' in layer_key:
          # θ_0 = θ + zeta
          theta_update_by_zeta[layer_key] = theta[layer_key] * (1. + zeta[layer_key])
        elif '_b' in layer_key:
          # θ_0 = θ o zeta (o is custom operator, see below)
          theta_update_by_zeta[layer_key] = theta[layer_key] + zeta[layer_key]
        else:
          assert("check ur dictionary!")
      theta.update(theta_update_by_zeta) # literally updating the dictionary!
    """inner gradient steps"""
    for _ in range(self.number_of_inner_gradient_steps):
      inner_logits = self.forward_theta(x_train, theta)
      cross_entropy_per_clas = CrossEntropy_Class(inner_logits, y_train)
      """omega: modulate class-specific loss"""
      if self.use_omega:
        inner_loss = torch.sum(cross_entropy_per_clas * F.softmax(omega, -1))
      else:
        inner_loss = torch.sum(cross_entropy_per_clas)
      # make computation graph, so 2nd-order derivativa can be calculated
      # when we call .backward() on OUTER loss
      grads = autograd.grad(outputs=inner_loss, inputs=list(theta.values()), create_graph=True)
      gradient_dictionary : dict[str, torch.Tensor] = dict(zip(theta.keys(), grads))
      # inner gradient step
      theta_new = {}
      for layer_key in theta.keys():
        if self.use_alpha: # manually checked that alpha is bound
          delta = alpha[layer_key] * gradient_dictionary[layer_key]
        else:
          delta = self.inner_lr * gradient_dictionary[layer_key]
        """use gamma: modulate task-specific learning rate"""
        if self.use_gamma:
          theta_new[layer_key] = theta[layer_key] - delta * torch.exp(gamma[layer_key])
        else:
          theta_new[layer_key] = theta[layer_key] - delta
      theta.update(theta_new)

    """outer-loss & test_accuracy"""
    logits_test = self.forward_theta(x_test, theta)
    cross_entropy = CrossEntropy(logits_test, y_test)
    accuracy = Accuracy(logits_test, y_test)
    prediction = F.softmax(logits_test, -1)
    return cross_entropy, accuracy, KL, prediction
