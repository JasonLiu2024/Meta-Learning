"""implementation of Temporal Convolutional Network (TCN)
    github: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    citation block:
    @article{BaiTCN2018,
	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
	journal   = {arXiv:1803.01271},
	year      = {2018},
    }
    paper: https://arxiv.org/pdf/1803.01271.pdf"""
import torch
import torch.nn as nn
# weight_norm is a function INSIDE weight_norm!
from torch.nn.utils.weight_norm import weight_norm

class ChopLastDimension(nn.Module):
    """inherits from nn.Module
        \nturns .shape=(x, y, z) tensor into .shape=(x, y, z-chomp_size)
        e.g. with chomp_size=3, .shape=(x, y, z) turns into .shape=z-3
        \ninput tensor need 3 dimensions! (.shape is (x, y, z))
        \nin the last dimension, we take the [first-to-chomp_size-from-last) section
        \nTLDR: we take out the last chomp_size entries
        with chomp_size=3, last dimension [1, 2, 3, 4] turns into [1] because we take out the last 3 entries
        \nThis is the Chomp1d class from the original implementation"""
    def __init__(self, chomp_size):
        # super(Chomp1d, self).__init__() # call constructor of the base class
        super().__init__() # better syntax in python 3.0 <- no need to name this class
        # see https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        self.chomp_size = chomp_size

    def forward(self, x):
        # .contiguous() manipulates the memory for the tensor t so it's contiguous
        # for tensor [[1, 2, 3], [4, 5, 6]], some ways to store it in memory:
        # 1. 1, 2, 3, 4, 5, 6 (contiguous)
        # 2. 1, 2, 3
        #    4, 5, 6 (NO contiguous)
        # PyTorch can complain that a tensor isn't contiguous, so we call .contiguous()
        # see https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        return x[:, :, :-self.chomp_size].contiguous()
    
class ResidualBlock(nn.Module):
    """inherits from nn.Module
        \nthe Resicual Block(see Figure 1 from paper)
        \ndilation factor d: btwn every two neighboring entry in kernel, put d-1 spacing
        \nelement in the spacing area is IGNORED in the convolution calculation
        \nfor d=2, with kernel_size-3, let ignored position be = o:
        \nin the 1D case
        kernel covers [x, x, x] -> covers [x, o, x, o, x]! (spacing of 2 - 1 = 1)
        \nin the 2D case, we have: 
        \n                    x, o, x, o, x
        \nx, x, x             o, o, o, o, o
        \nx, c, x     to      x, o, x, o, x   
        \nx, x, x             o, o, o, o, o
        \n                    x, o, x, o, x
        \nThis is the Temporal Block in the original implementation
        """
    def __init__(self, input_channels, output_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.nonlinearity = nn.ReLU()
        self.convolution_1 = weight_norm(nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation))
        self.convolution_2 = weight_norm(nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation))
        self.network = nn.Sequential(
            self.convolution_1,
            ChopLastDimension(padding),
            self.nonlinearity,
            nn.Dropout(dropout),
            self.convolution_2,
            ChopLastDimension(padding),
            self.nonlinearity,
            nn.Dropout(dropout))
        # makes sure input and output have same number of channels
        #    "a 1x1 convolution is added when residual input and output have different dimensions"
        self.downsample = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=1) if input_channels != output_channels else None
        self.init_weights()

    def init_weights(self):
        self.convolution_1.weight.data.normal_(0, 0.01)
        self.convolution_2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
    #   "A residual block (He et al., 2016) contains a branch leading out to a series of transformations F, 
    #   whose outputs are added to the input x of the block:
    #   o = Activation(x + F(x))"
    # Here, F is the network!
    # downsampling is here to make sure dimensions are same, so the addition actually works!
    #   "This effectively allows layers to learn modifications to the identity mapping rather 
    #   than the entire transformation, which has repeatedly been shown to benefit very deep net- works.""
        out = self.network(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.nonlinearity(out + res)

class TemporalConvNet(nn.Module):
    """channel_ct_list: number of channels for each LEVEL of the network"""
    def __init__(self, input_channels, output_channels_list, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(output_channels_list)
        # print(f"TemporalConvNet::ResidualBlocks")
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else output_channels_list[i-1]
            out_channels = output_channels_list[i]
            layers += [ResidualBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            # print(f"in: {in_channels}, out: {out_channels}, kernel: {kernel_size}")
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)