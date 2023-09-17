import torch
import numpy as np
from layers import *
from encoder import InferenceNetwork
from torch import autograd
NUMBER_OF_HIDDEN_CHANNELS = 64 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class LearningToBalance():
  def __init__(self, data_name : str, number_of_inner_gradient_steps : int,
               ways : int, shots : int, inner_learning_rate : float,
               outer_learning_rate : float, batch_size : int):
    if data_name == 'cifar':
      self.xdim = 32
      self.number_of_channels = 32
    elif data_name == 'mimgnet':
      self.xdim = 84
      self.number_of_channels = 32
    else:
      raise ValueError("Invalid dataset name")
    # training parameters
    self.number_of_inner_gradient_steps = number_of_inner_gradient_steps
    self.number_of_classes = ways
    # self.metabatch = metabatch
    self.inner_lr = inner_learning_rate
    self.outer_lr = outer_learning_rate
    self.batch_size = batch_size # number of TASKs in a batch!
    # balancing variables from InferenceNetwork
    self.encoder = InferenceNetwork(ways, shots, data_name, True, True, True).to(DEVICE)
    self.use_inner_step_size_vector = True # learnable, from Meta-SGD
    self.use_zeta = True
    self.use_gamma = True
    self.use_omega = True
    self.use_alpha = True # learn alpha, instead of using set value
    """model definition: 4 convolutional layers, followed, by 1 linear layer"""
    self.number_of_convolutional_layer = 4
    self.input_channel = 3
    self.kernel_size = 3
    self.number_of_hidden_channels = NUMBER_OF_HIDDEN_CHANNELS
    self.parameter = self.get_parameter_maml('theta') # 'this is the 'theta'
    self.alpha = self.get_parameter_maml('alpha')
    """training"""
    self.optimizer = torch.optim.Adam(params=list(self.parameter.values()) + list(self.alpha.values()), lr=self.outer_lr)
    self.validation_interval = 50
  def get_parameter_maml(self, name : str) -> dict[str, torch.Tensor]:
    """initializes self.parameters"""
    # initializers
    if name == 'theta':
      init_convolution  = lambda x : torch.nn.init.trunc_normal_(x, std=0.02)
      init_bias         = lambda x : torch.nn.init.zeros_(x)
      init_dense        = lambda x : torch.nn.init.normal_(x, std=0.02)
    else: # name == 'alpha'
      init_convolution  = lambda x : torch.nn.init.constant_(x, 0.01)
      init_bias         = lambda x : torch.nn.init.constant_(x, 0.01)
      init_dense        = lambda x : torch.nn.init.constant_(x, 0.01)
    parameters = {}
    input_channel = self.input_channel
    # convolutional layers
    print(f"model, get_parameter, generating params {name}")
    for l in range(1, self.number_of_convolutional_layer + 1):
      parameters[f'convolution_{l}_weight'] = init_convolution(torch.empty(
        size=[self.number_of_hidden_channels, input_channel, self.kernel_size, self.kernel_size],
        requires_grad=True,
        device=DEVICE
      ))
      parameters[f'convolution_{l}_bias'] = init_bias(torch.empty(
        # do NOT assign int to 'size' field! (type problem)
        # for weight dimension [64, 3, 3, 3], we expect bias of [64]<-one dimension only!
        self.number_of_hidden_channels,
        requires_grad=True,
        device=DEVICE
      ))
      input_channel = self.number_of_hidden_channels
      # print(f"\tconv layer {l}, weight shape {parameters[f'convolution_{l}_weight'].shape}")
      # print(f"\tconv layer {l}, bias shape   {parameters[f'convolution_{l}_bias'].shape}")
    # linear layer
    parameters[f'dense_weight'] = init_dense(torch.empty(
      size=[self.number_of_classes, self.number_of_hidden_channels],
      requires_grad=True,
      device=DEVICE
    ))
    parameters[f'dense_bias'] = init_bias(torch.empty(
      size=[self.number_of_classes],
      requires_grad=True,
      device=DEVICE
    ))
    # print(f"\tlinear layer, weight shape {parameters[f'dense_{self.number_of_convolutional_layer}_weight'].shape}")
    # print(f"\tlinear layer, bias shape   {parameters[f'dense_{self.number_of_convolutional_layer}_bias'].shape}")

    return parameters
  # omitted, replaced by original MAML code's handling
  # def get_parameter(self, name : str) -> dict[str, torch.Tensor]:
  #   """name: name of parameter in {theta, alpha}"""
  #   if name == 'theta':
  #     init_convolution  = lambda x : torch.nn.init.trunc_normal_(x, std=0.02)
  #     init_bias         = lambda x : torch.nn.init.zeros_(x)
  #     init_dense        = lambda x : torch.nn.init.normal_(x, std=0.02)
  #   else: # name == 'alpha'
  #     init_convolution  = lambda x : torch.nn.init.constant_(x, 0.01)
  #     init_bias         = lambda x : torch.nn.init.constant_(x, 0.01)
  #     init_dense        = lambda x : torch.nn.init.constant_(x, 0.01)
  #   parameter_dictionary : dict[str, torch.Tensor]= {}
  #   for l in [1, 2, 3, 4]:
  #     channels = self.input_channel if l == 1 else self.number_of_channels
  #     weight = torch.Tensor(size=[3, 3, channels, self.number_of_channels])
  #     parameter_dictionary[f'convolution_{l}_weight'] = init_convolution(weight)
  #     bias = torch.Tensor(size=[self.number_of_channels])
  #     parameter_dictionary[f'convolution_{l}_bias'] = init_bias(bias)

  #   remaining = 2 * 2 if self.xdim == 32 else 5 * 5
  #   if name == 'theta':
  #     parameter_dictionary[f'dense_weight'] = torch.zeros(size=[remaining * self.number_of_channels, self.number_of_classes])
  #     parameter_dictionary[f'dense_bias'] = torch.zeros(size=[self.number_of_classes])
  #   else: # name == 'alpha'
  #     single_weight = init_dense(torch.Tensor(size=[remaining * self.number_of_channels, 1]))
  #     single_bias = init_bias(torch.Tensor(size=[1]))
  #     parameter_dictionary[f'dense_weight'] = torch.tile(single_weight, [1, self.number_of_classes])
  #     parameter_dictionary[f'dense_bias'] = torch.tile(single_bias, [self.number_of_classes])
  #   return parameter_dictionary

  def forward_theta(self, x : torch.Tensor, theta : dict[str, torch.Tensor]):
    x = torch.reshape(x, [-1, self.input_channel, self.xdim, self.xdim])
    # print(f"model forward_theta:")
    for l in range(1, self.number_of_convolutional_layer + 1):
      weight = theta[f'convolution_{l}_weight']
      bias = theta[f'convolution_{l}_bias']
      # more steps inside ConvolutionBlock_F, compared to original MAML code!
      # print(f"\tconv layer {l}, weight shape {theta[f'convolution_{l}_weight'].shape}")
      # print(f"\tconv layer {l}, bias shape   {theta[f'convolution_{l}_bias'].shape}")
      # print(f"\tinput dimension: {x.shape}")
      x = ConvolutionBlock_F(x, weight, bias) # just executes the calculation
    weight = theta[f'dense_weight']
    bias = theta[f'dense_bias']
    x = torch.mean(x, dim=[2, 3])
    # print(f"\tlinear layer, weight shape {weight.shape}")
    # print(f"\tlinear layer, bias shape   {weight.shape}")
    # print(f"\tinput dimension: {x.shape}")
    x = DenseBlock_F(x, weight, bias) # just executes the calculation
    # print(f"result of a forward_theta shape: {x.shape}")
    return x
  def zeta_update_initialization(self, 
    zeta : dict[str, torch.Tensor], theta : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """update theta with zeta"""
    # print(f"model: zeta_update_initialization")
    theta_update_by_zeta = {}
    if self.use_zeta:
      for layer_key in zeta.keys():
        # Specifically, we use z_train = 1 + z _test convolutional network weights 
        # ... z_train = z _test for the biases
        # which modifies the network parameters θ like: (let θ_0 be NEW initializtion)
        if '_w' in layer_key:
          # θ_0 = θ + zeta
          theta_update_by_zeta[layer_key] = theta[layer_key] * (1. + zeta[layer_key])
          # print(f"\tW theta shape {(theta[layer_key].shape)}")
          # print(f"\tW zeta shape  {(zeta[layer_key].shape)}")
          # print(f"\tWupdated zeta shape  {(theta_update_by_zeta[layer_key].shape)}")
        elif '_b' in layer_key:
          # θ_0 = θ o zeta (o is custom operator, see below)
          theta_update_by_zeta[layer_key] = theta[layer_key] + zeta[layer_key]
          # print(f"\tB theta shape {(theta[layer_key].shape)}")
          # print(f"\tB zeta shape  {(zeta[layer_key].shape)}")
          # print(f"\tB updated zeta shape  {(theta_update_by_zeta[layer_key].shape)}")
        else:
          assert("LearningToBalance::theta_update_initialization(): \n\tcheck ur dictionary keys!")
    return theta_update_by_zeta
  def omega_modify_loss_of_class(self, 
    omega: torch.Tensor, cross_entropy_per_class : torch.Tensor) -> torch.Tensor:
    # """change task-specific loss with omega"""
    # print(f"model omega_modify_loss_of_class")
    if self.use_omega:
      inner_loss = torch.sum(cross_entropy_per_class * F.softmax(omega, -1))
    else:
      inner_loss = torch.sum(cross_entropy_per_class)
    return inner_loss
  def gamma_modeify_inner_step_learning_rate(self, 
    theta_tensor : torch.Tensor, gamma_tensor : torch.Tensor, delta : torch.Tensor) -> torch.Tensor:
    """change learning rate with gamma"""
    # print(f"model gamma_modeify_inner_step_learning_rate")
    if self.use_gamma:
      return theta_tensor - delta * torch.exp(gamma_tensor)
    else:
      return theta_tensor - delta
  def _inner_loop(self, x_train : torch.Tensor, y_train : torch.Tensor, train : bool,
    theta : dict[str, torch.Tensor], omega : torch.Tensor, gamma : dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    theta_update_by_inner_loop : dict[str, torch.Tensor] = {}
    """return: updated parameters theta, list of inner accuracy (length is number_of_inner_gradient_steps + 1)"""
    # print(f"model _inner_loop")
    inner_loss : torch.Tensor = torch.empty(0) # only for type hint
    accuracy_list = []
    # print(f"model _inner_loop:")
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
    # cross-entropy expects: logits_dim=[batch, class], target=[batch]
    y_train = y_train.to(torch.int64)
    for _ in range(self.number_of_inner_gradient_steps):
      inner_logits = self.forward_theta(x_train, theta)
      # print(f"\tinner_logits shape: {inner_logits.shape}, dtype: {inner_logits.dtype}")
      # print(f"\ty_train shape:      {y_train.shape}, dtype: {y_train.dtype}")
      cross_entropy_per_class = CrossEntropy_Class(inner_logits, y_train)
      inner_accuracy = Accuracy(inner_logits, y_train)
      """omega: modulate class-specific loss"""
      inner_loss = self.omega_modify_loss_of_class(omega, cross_entropy_per_class)
      # make computation graph, so 2nd-order derivativa can be calculated
      # when we call .backward() on OUTER loss
      # when train, DO make graph, to allow back propagation; when validate, NO make graph!
      grads = autograd.grad(outputs=inner_loss, inputs=list(theta.values()), create_graph=train)
      gradient_dictionary : dict[str, torch.Tensor] = dict(zip(theta.keys(), grads))
      accuracy_list.append(inner_accuracy)
      """inner gradient step"""
      for layer_key in theta.keys():
        if self.use_alpha: # manually checked that alpha is bound
          delta = self.alpha[layer_key] * gradient_dictionary[layer_key]
        else:
          delta = self.inner_lr * gradient_dictionary[layer_key]
        """use gamma: modulate task-specific learning rate"""
        theta_update_by_inner_loop[layer_key] = self.gamma_modeify_inner_step_learning_rate(
          theta_tensor=theta[layer_key], gamma_tensor=gamma[layer_key], delta=delta
        )
      # no gradient update for this last logit and accuracy
      last_inner_logits = self.forward_theta(x_train, theta_update_by_inner_loop)
      last_inner_accuracy = Accuracy(last_inner_logits, y_train)
      accuracy_list.append(last_inner_accuracy.item())
    return theta_update_by_inner_loop, accuracy_list
  def _outer_step_single_task(self, input_task : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    train : bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """return: cross entropy, outer accuracy, KL, prediction, inner accuracy
    \nouter objective w.r.t. one task"""
    # print(f"model _outer_step_single_task")
    # theta = self.get_parameter('theta')
    # alpha : dict[str, torch.Tensor] = {}
    # if self.use_alpha: 
    #   alpha = self.get_parameter('alpha')
    # we have 'batch of batch'
    # so dimension becomes [batch of task, batch, channels, width, height]
    # print(f"input task shape: {input_task.shape}")
    x_train, y_train, x_test, y_test = input_task
    # x_train = x_train[0]
    # y_train = y_train[0]
    # x_test = x_test[0]
    # y_test = y_test[0]
    # print(f"x_train shape {x_train.shape}")
    # print(f"y_train shape {y_train.shape}")
    # print(f"x_test shape  {x_test.shape}")
    # print(f"y_test shape  {y_test.shape}")
    # .forward is called by the __call__() function, DON'T call it directly!
    # training time = DO sample; no training time = NO sample!
    omega, gamma, zeta, KL = self.encoder(x_train, y_train, do_sample=train)
    Cardinality_train = x_train.shape[0]
    Cardinality_test = x_test.shape[0]
    # scale KL term w.r.t train and test set sizes
    KL /= (Cardinality_train + Cardinality_test)
    """zeta: modulate MAML initialization"""
    theta_update_by_zeta = self.zeta_update_initialization(zeta, self.parameter)
    self.parameter.update(theta_update_by_zeta)
    """inner gradient steps; omega & gamma for task_specific modulation"""
    theta_update_by_inner_loop, inner_accuracy = self._inner_loop(x_train, y_train, train, self.parameter, omega, gamma)
    self.parameter.update(theta_update_by_inner_loop)
    """outer-loss & test_accuracy"""
    logits_test = self.forward_theta(x_test, self.parameter)
    cross_entropy = CrossEntropy(logits_test, y_test.to(torch.int64))
    outer_accuracy = Accuracy(logits_test, y_test)
    prediction = F.softmax(logits_test, -1)
    return cross_entropy, outer_accuracy, KL, prediction, inner_accuracy
  def _outer_step_(self, input_task_batch : list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], train : bool
    ):
    """return: loss, result dictionary
    \nmade of _outer_step_single_task"""
    # print(f"model _outer_step")
    cross_entropy_list = []
    outer_accuracy_list : list[torch.Tensor]= []
    KL_list = []
    prediction_list = []
    inner_accuracy_list : list[list[torch.Tensor]]= []
    # print(f"input_task_batch type: {type(input_task_batch)}")
    for t in range(self.batch_size):
    # """possible improvement: use torch.vmap instead of calculating one by one"""
      # print(f"length of input_task_batch: {input_task[0].shape}")
      input_task = (input_task_batch[0][t], input_task_batch[1][t], 
                    input_task_batch[2][t], input_task_batch[3][t])
      cross_entropy, outer_accuracy, KL, prediction, inner_accuracy = self._outer_step_single_task(input_task, train)
      # print(f"cross_entropy {cross_entropy}")
      cross_entropy_list.append(cross_entropy)
      outer_accuracy_list.append(outer_accuracy)
      inner_accuracy_list.append(inner_accuracy)
      prediction_list.append(prediction)
      KL_list.append(KL)
    # print(f"cross_entropy_list length: {len(cross_entropy_list)}")
    loss = torch.mean(torch.stack(cross_entropy_list))
    KL = torch.mean(torch.stack(KL_list))
    return loss, torch.mean(torch.stack(outer_accuracy_list)), KL, torch.mean(torch.Tensor(inner_accuracy_list), dim=0)
  def train(self, train_dataloader, valid_dataloader, ):
    """return: train loss, valid loss, train accuracy dictionary, valid accuracy dicitonary"""
    print(f"T-R-A-I-N")
    print(f"train_dataloader length? {len(train_dataloader)}")
    print(f"valid_dataloader length? {len(valid_dataloader)}")
    # print(f"length of dataloader {len(train_dataloader)}")
    train_loss = []
    train_outer_accuracy = [] # outer means query; inner means support
    train_inner_accuracy_fresh = [] # pre-adapting
    train_inner_accuracy_adapt = [] # post-adapting
    """TODO: show valid loss s.t. it reflects when number of validation != number of training, or throw Error"""
    valid_loss = []
    valid_outer_accuracy = [] # outer means query; inner means support
    valid_inner_accuracy_fresh = [] # pre-adapting
    valid_inner_accuracy_adapt = [] # post-adapting
    # print(f"yo")
    for step, task_batch in enumerate(train_dataloader):
      """task_batch is a BATCH (list) of tasks, batch = batch_size,
      shape: [task batch, number of classes, width, height, number of channels]
      e.g. [5, 70 = way * (shot + query), 32, 32, 3 bc RGB]"""
      # print(f"train step {step}")
      # enumerated dataloader is a list[tuple[stuffs]]
      # print(f"type of dataloader: {type(train_dataloader)}")
      # print(f"type of task batch: {type(task_batch)}")
      self.optimizer.zero_grad()
      outer_loss, outer_accuracy, KL, inner_accuracy = self._outer_step_(task_batch, train=True)
      # print(f"ran _outer_step()")
      outer_loss.backward(retain_graph=True)
      self.optimizer.step()
      train_loss.append(outer_loss.item())
      train_outer_accuracy.append(outer_accuracy)
      train_inner_accuracy_fresh.append(inner_accuracy[0])
      train_inner_accuracy_adapt.append(inner_accuracy[-1])
      if step % self.validation_interval == 0: # it's validation time!
        print(f"V-A-L-I-D")
        for v_step, task_batch in enumerate(valid_dataloader):
          # print(f"valid step {v_step}")
          outer_loss, outer_accuracy, KL, inner_accuracy = self._outer_step_(task_batch, train=False)
          valid_loss.append(outer_loss.item())
          valid_outer_accuracy.append(outer_accuracy)
          valid_inner_accuracy_fresh.append(inner_accuracy[0])
          valid_inner_accuracy_adapt.append(inner_accuracy[-1])
    train_accuracy = {
      'outer':            train_outer_accuracy,
      'inner pre-adapt':  train_inner_accuracy_fresh,
      'inner post_adapt': train_inner_accuracy_adapt
    }
    valid_accuracy = {
      'outer':            valid_outer_accuracy,
      'inner pre-adapt':  valid_inner_accuracy_fresh,
      'inner post_adapt': valid_inner_accuracy_adapt
    }
    return train_loss, valid_loss, train_accuracy, valid_accuracy
  def test(self, test_dataloader):
    """return: test loss, test accuracy dictionary"""
    print("T-E-S-T")
    print(f"test_dataloader length? {len(test_dataloader)}")
    test_loss = []
    test_outer_accuracy = [] # outer means query; inner means support
    test_inner_accuracy_fresh = [] # pre-adapting
    test_inner_accuracy_adapt = [] # post-adapting
    for step, task_batch in enumerate(test_dataloader):
      # print(f"test step {step}")
      outer_loss, outer_accuracy, KL, inner_accuracy = self._outer_step_(task_batch, train=False)
      test_loss.append(outer_loss.item())
      test_outer_accuracy.append(outer_accuracy)
      test_inner_accuracy_fresh.append(inner_accuracy[0])
      test_inner_accuracy_adapt.append(inner_accuracy[-1])
    test_accuracy = {
      'outer':            test_outer_accuracy,
      'inner pre-adapt':  test_inner_accuracy_fresh,
      'inner post_adapt': test_inner_accuracy_adapt
    }
    return test_loss, test_accuracy
      

