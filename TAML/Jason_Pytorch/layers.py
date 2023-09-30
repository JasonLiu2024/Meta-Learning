import numpy as np
import torch
import torch.nn.functional as F
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""functions (NOT used)"""
log = lambda x : torch.log(x + 1e-20)
softmax = torch.nn.Softmax()
relu = torch.nn.ReLU()
# Softplus is a smooth approximation to Softmax function, 
# output always positive
softplus = torch.nn.Softplus() 
sigmoid = torch.nn.Sigmoid()
exp = torch.exp

def KL_Diagonal_StandardNormal(q : torch.distributions.Normal) -> torch.Tensor:
    """Kullback-Leibler divergence KL(p || q) 
    original name: kl_diagnormal_stdnormal()"""
    q_shape : torch.Size = q.mean.size()
    p = torch.distributions.Normal(loc=torch.zeros(size=q_shape).to(DEVICE), 
                                   scale=torch.ones(size=q_shape).to(DEVICE))
    """we're looking for KL[q(φ |train data; ψ) || φ]
    where q(φ |train data; ψ) is our approximate posterior"""
    return torch.distributions.kl_divergence(q, p)

def DENSE(x : torch.Tensor, output_dimension : int,) -> torch.nn.Module:
    """Linear layer"""
    layer = torch.nn.Linear(x.size(dim=1), output_dimension)
    torch.nn.init.normal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer
def CONV(x : torch.Tensor, out_channels : int, 
         weight : torch.Tensor, bias : torch.Tensor, 
         kernel_size=3, strides=1, groups=1) -> torch.nn.Module:
  """Convolution 2D, 
  \nweight shape: (out_channel, in_channel/groups, kernel_size[0], kernel_size[1])
  \nbias shape: out_channel"""
  # weight/filter is matrix used to do convolution
  # shape: (out_channel, in_channel/groups, kernel_size[0], kernel_size[1])
  # where groups = number of groups to split input channels into
  # (default convolution applies filter to all input channels)
  # source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
  layer = torch.nn.Conv2d(in_channels=x.size()[1], out_channels=out_channels, kernel_size=3, stride=3)
  layer.weight = torch.nn.Parameter(weight)
  layer.bias = torch.nn.Parameter(bias)
  torch.nn.init.trunc_normal_(layer.weight, std=0.02)
  torch.nn.init.zeros_(layer.bias) 
  return layer
# NOT used, only for reference
def POOL() -> torch.nn.Module:
  layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
  return layer
  # tf.layers.max_pooling2d(x, 2, 2, padding='valid', **kwargs)
# NOT used, only for reference
def Same_Padding(img_size, kernel_size, stride=1) -> int:
  """find padding size to make output size equal to input size, for conv2d
  \nassume SQUARE img"""
  return ((stride - 1) * img_size - stride + kernel_size)//2
  # formula: ((s - 1) * i - s + f) / 2, s=stride, i=img_size, f=kernel_size
  # source: https://saturncloud.io/blog/is-there-really-no-paddingsame-option-for-pytorchs-conv2d/#:~:text=Many%20data%20scientists%20who%20are,to%20achieve%20the%20same%20functionality.

def ConvolutionBlock_F(x : torch.Tensor, 
  weight : torch.Tensor, bias : torch.Tensor) -> torch.Tensor:
  # print(f"ConvolutionBlock_F: input shape {x.shape}") # checks out
  x = F.conv2d(x, weight, bias, stride=1, padding='same')
  x = F.batch_norm(x, None, None, training=True) 
  # batch norm BEFORE relu!
  # NO max pooling
  x = F.relu(x)
  # print(f"ConvolutionBlock_F: x shape: {x.shape}")
  # running_mu = torch.zeros(x.shape[1]).to(DEVICE)
  # running_var = torch.ones(x.shape[1]).to(DEVICE)
  # print(f"ConvolutionBlock_F: x infinite?: {torch.isfinite(x)}")
  # x = F.max_pool2d(input=x, kernel_size=(1, 1), stride=(1, 1))
  return x

# class ConvolutionBlock(torch.nn.Module):
#   def __init__(self, in_channel : int, out_channel : int, weight : torch.Tensor, bias : torch.Tensor):
#     super().__init__()
#     self.in_channel = in_channel
#     self.out_channel = out_channel
#     # padding='same' pads the input so the output has the shape as the input. However, this mode doesn’t support any stride values other than 1.
#     # source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
#     self.network = torch.nn.Sequential(
#       torch.nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, padding='same'),
#       torch.nn.ReLU(), # in tensorflow, activation function is part of batch normalization
#       torch.nn.BatchNorm2d(self.out_channel),
#       torch.nn.MaxPool2d(kernel_size=(1,2,2,1), stride=(1,2,2,1))
#     )
#   def forward(self, x : torch.Tensor):
#     return self.network(x)

def DenseBlock_F(x : torch.Tensor,
  weight : torch.Tensor, bias : torch.Tensor) -> torch.Tensor:
  # x = torch.nn.Flatten()(x)
  # return torch.matmul(input=x, other=weight) + bias
  return F.linear(input=x, weight=weight, bias=bias)

# class DenseBlock(torch.nn.Module):
#   def __init__(self, weights : torch.Tensor, bias : torch.Tensor):
#     super().__init__()
#     self.weights = weights
#     self.bias = bias
#     self.network = torch.nn.Flatten()
#   def forward(self, x : torch.Tensor):
#     return torch.matmul(input=self.network(x), other=self.weights) + self.bias

def CrossEntropy(logits : torch.Tensor, labels : torch.Tensor):
  """cross entropy loss for logits, 
  \nsoftmax is part of cross_entropy()"""
  # source: https://stackoverflow.com/questions/49377483/about-tf-nn-softmax-cross-entropy-with-logits-v2
  logits = F.softmax(logits, dim=1)
  loss = F.cross_entropy(input=logits, target=labels)
  return torch.mean(loss)

def CrossEntropy_Class(logits : torch.Tensor, labels : torch.Tensor):
  # print(f"CrossEntropy_Class")
  # print(f"\tlogits shape: {logits.shape}")
  # print(f"\tlabels shape: {labels.shape}")
  logits = F.softmax(logits, dim=1)
  loss = F.cross_entropy(logits, labels, reduction='none')
  # print(f"loss: {loss}")
  # print(f"\tloss shape {loss.shape}")
  # tf.expand_dims() == torch.unsqueeze()
  # source: https://discuss.pytorch.org/t/equivalent-to-torch-unsqueeze-in-tensorflow/26379
  # print(f"CrossEntropy_Class, labels: \n\t{labels}")
  right = torch.unsqueeze(loss, 1)
  # print(f"\tright shape: {right.shape}, dtype {right.dtype}")
  left = torch.transpose(torch.unsqueeze(labels, 1), 0, 1)
  # print(f"\tleft shapeL {left.shape}, dtype {left.dtype}")
  perclass = torch.matmul(left.to(torch.float32), right)
  # print(f"\tperclass shape: {perclass.shape}")
  # source: https://stackoverflow.com/questions/65092587/tensorflow-to-pytorch
  # source: https://stackoverflow.com/questions/59132647/tf-cast-equivalent-in-pytorch
  N = float(torch.unsqueeze(labels, 1).shape[1]) # number of 
  # print(f"\nN: {N}")
  way = float(torch.unsqueeze(labels, 1).shape[0]) # number of classes ?
  # print(f"\nway: {way}")
  # squeeze removes a dimension if it's size is 1
  # source: https://pytorch.org/docs/stable/generated/torch.squeeze.html
  return torch.squeeze(perclass) * way / N

def Accuracy(logits, labels, axis=-1):
  # use .eq(), NOT .equal() need array, not single value
  # source: https://pytorch.org/docs/stable/generated/torch.eq.html
  correct = torch.eq(torch.argmax(logits, -1), torch.argmax(labels, -1))
  # 'correct' is boolean tensor, need .float() to turn it into number!
  # source: https://stackoverflow.com/questions/62150659/how-to-convert-a-tensor-of-booleans-to-ints-in-pytorch
  # equivalent to: self.to(torch.float32)
  # source: https://pytorch.org/docs/stable/generated/torch.Tensor.float.html
  return torch.mean(correct.float(), axis)