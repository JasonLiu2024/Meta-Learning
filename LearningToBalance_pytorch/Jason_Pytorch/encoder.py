import torch

class InferenceNetwork:
    """
    ways: Int number of ways
    shots: Int number of shots
    data_name : String name of data {cifar, mimgnet}
    z: Bool, whether to use z
    g: Bool, whether to use g
    o: Bool, wehther to use o"""
    def __init__(self, ways, shots, data_name, z, g, o):
        if data_name == 'cifar':
            self.xdim = 32 
            self.input_channel = 3
            self.number_of_channels = 32
        elif data_name == 'mimgnet':
            self.xdim = 84 
            self.input_channel = 3
            self.number_of_channels = 32
        else:
            raise ValueError("InferenceNetwork::wrong data name")
        self.number_of_classes = ways
        self.number_of_shots = shots
        self.need_z = z
        self.neeg_g = g
        self.neeg_o = o
    def statistics_pooling(self, x, cardinality):
        """get element-wise sample mean, variance, cardinality"""
        variance, mean = torch.var_mean(input=x, dim=0)
        cardinality = torch.tile(input=torch.reshape(input=cardinality, shape=[-1]), dims=mean.shape.as_list())
        return torch.stack(tensors=[mean, variance, cardinality], dim=1)
    def get_posterior(self, x, y):
        x = 
    
