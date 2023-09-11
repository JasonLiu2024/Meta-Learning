import torch
import numpy as np
from layers import KL_Diagonal_StandardNormal
from collections import OrderedDict

class InferenceNetwork(torch.nn.Module):
    """
    this is a dataset encoder. It acquires balancing variables z, g, o (see explanation in get_posterior_distribution())
    ways: Int number of ways
    shots: Int number of shots
    data_name : String name of data {cifar, mimgnet}
    boolean flags for whether to use z, g, o, and s
    z (z in paper): Bool, whether to use z, for class imbalance
    g (γ in paper): Bool, whether to use g, for task imbalance
    o (ω in paper): Bool, wehther to use o, for task imbalance"""
    def __init__(self, ways : int, shots : int, data_name : str, 
                 need_z : bool, need_g : bool, need_o : bool, need_s = True):
        super().__init__()
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
        self.use_z = need_z
        self.z_dimension = 2 * self.number_of_channels * 4
        self.use_g = need_g
        self.g_dimension = 5
        self.use_o = need_o
        self.o_dimension = 1
        self.use_s = need_s

        self.network_class_encoder_1 = torch.nn.Sequential(
            # tensorflow's conv2d omits the in_channel field
            torch.nn.Conv2d(self.input_channel, 10, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(10, 10, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            # use LazyLinear to infer dimension!
            torch.nn.LazyLinear(64)
        )
        self.network_transition_1 = torch.nn.Sequential(
            torch.nn.LazyLinear(4),
            torch.nn.ReLU()
        )
        self.network_task_encoder_2 = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32)
        )
        self.network_transition_2 = torch.nn.Sequential(
            torch.nn.LazyLinear(4),
            torch.nn.ReLU()
        )
        self.network_omega = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.ReLU()
        )
        self.network_omega_mean = torch.nn.Sequential(
            torch.nn.LazyLinear(self.o_dimension),
            torch.nn.ReLU()
        )
        self.network_omega_std = torch.nn.Sequential(
            torch.nn.LazyLinear(self.o_dimension),
            torch.nn.ReLU()
        )
        self.network_gamma = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.ReLU()
        )
        self.network_gamma_mean = torch.nn.Sequential(
            torch.nn.LazyLinear(self.g_dimension),
            torch.nn.ReLU()
        )
        self.network_gamma_std = torch.nn.Sequential(
            torch.nn.LazyLinear(self.g_dimension),
            torch.nn.ReLU()
        )
        self.network_zeta = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.ReLU()
        )
        self.network_zeta_mean = torch.nn.Sequential(
            torch.nn.LazyLinear(self.z_dimension),
            torch.nn.ReLU()
        )
        self.network_zeta_std = torch.nn.Sequential(
            torch.nn.LazyLinear(self.z_dimension),
            torch.nn.ReLU()
        )
    def get_MeanVarianceCardinality(self, x : torch.Tensor, cardinality) -> torch.Tensor:
        """get element-wise sample mean, variance, cardinality"""
        # print(f"get_MeanVarianceCardinality::input shape {x.size()}")
        variance, mean = torch.var_mean(input=x, dim=0)
        # print(f"get_MeanVarianceCardinality::mean shape {mean.size()}")
        cardinality = torch.tile(input=torch.reshape(input=cardinality, shape=[-1]), dims=list(mean.size()))
        # print(f"get_MeanVarianceCardinality::cardinality {cardinality.size()}")
        class_summary = torch.stack(tensors=[mean, variance, cardinality], dim=1)
        # print(f"get_MeanVarianceCardinality::output shape: {class_summary.size()}")
        return class_summary
    def statistics_pooling_1(self, x : torch.Tensor, y : torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """get: class summary vector class_summary (s in paper)–for EACH class, compiled altogether
        \nalso returns list of cardinality"""
        # encode entire class
        x = x.reshape([-1, self.input_channel, self.xdim, self.xdim,])
        # print(f"statistics_pooling_1::x reshaped: {x.shape}")
        x = self.network_class_encoder_1(x)
        # print(f"statistics_pooling_1::x shape after encode: {x.shape}")
        # statistics pooling 1
        y_sum = torch.sum(y, 1)
        # print(f"statistics_pooling_1::y_sum shape {y_sum.size()}")
        y_largest_index = torch.argmax(y, 1)
        MeanVarianceCardinality_list = []
        Cardinality_list : list[torch.Tensor] = []
        for cls in range(self.number_of_classes):
            cls_mask = torch.eq(y_largest_index, cls) & torch.greater(y_sum, 0)
            cls_mask = cls_mask.int().float()
            # use cls_index as boolean mask
            x_cls = x * cls_mask
            # print(f"x_cls shape: {x_cls.size()}")
            y_cls = y * cls_mask
            # normalize the size of the set
            Cardinality_cls = (torch.sum(y_cls) - 1.) / (self.number_of_shots - 1.)
            # print(f"Cardinality_cls: {Cardinality_cls}")
            # print(f"Cardinality_cls shape: {Cardinality_cls.size()}")
            Cardinality_list.append(Cardinality_cls)
            MeanVarianceCardinality = self.get_MeanVarianceCardinality(x_cls, Cardinality_cls)
            MeanVarianceCardinality_list.append(MeanVarianceCardinality)
        # print(f"statistics_pooling_1()::MeanVarianceCardinality_list {MeanVarianceCardinality_list[0].size()}")
        class_summary : torch.Tensor = torch.stack(MeanVarianceCardinality_list, 0)
        # print(f"statistics_pooling_1()::class_summary first shape: {class_summary.size()}")
        class_summary = self.network_transition_1(class_summary)
        class_summary = class_summary.reshape([self.number_of_classes, -1])
        # print(f"statistics_pooling_1()::class_summary shape: {class_summary.size()}")
        return class_summary, Cardinality_list
    def statistics_pooling_2(self, class_summary : torch.Tensor, Cardinality_list) -> torch.Tensor:
        """get: summary of entire task aka all classes from statistics_pooling_1()"""
        # encode entire task
        # print(f"statistics_pooling_2()::task_summary shape: {class_summary.size()}")
        task_summary : torch.Tensor = self.network_task_encoder_2(class_summary)
        # statistics pooling 2
        task_summary = self.get_MeanVarianceCardinality(task_summary, torch.sum(torch.Tensor(Cardinality_list)))
        task_summary = torch.unsqueeze(task_summary, 0)
        task_summary = self.network_transition_2(task_summary)
        task_summary = task_summary.reshape([1, -1])
        return task_summary
    def get_omega(self, class_summary : torch.Tensor) -> torch.distributions.Normal:
        """omega: balancing variable for class imbalance
        \nits use: multiply to class-specific gradient
        \nthere's an omega for each class!
        \npicks up from statistics_pooling_1()"""
        class_summary_embedding : torch.Tensor = self.network_omega(class_summary)
        summary_o = class_summary if self.use_s else class_summary_embedding
        mu = torch.squeeze(self.network_omega_mean(summary_o))
        sigma = torch.squeeze(self.network_omega_std(summary_o))
        # torch.nn.Softplus works on number input only
        # BUT, torch.nn.functional.softplus() works on entire tensor, element-wise
        q = torch.distributions.Normal(mu, torch.nn.functional.softplus(sigma))
        return q
    def get_gamma(self, task_summary : torch.Tensor) -> torch.distributions.Normal:
        """gamma: balancing variable for task imbalance
        \nits use: multiply to class-specific learning rate
        \nthere's one gamma for each task!
        \nuse result from statistics_pooling_2()"""
        task_summary_embedding = self.network_gamma(task_summary)
        mu = torch.squeeze(self.network_gamma_mean(task_summary_embedding))
        sigma = torch.squeeze(self.network_gamma_std(task_summary_embedding))
        q = torch.distributions.Normal(mu, torch.nn.functional.softplus(sigma))
        return q
    def get_zeta(self, task_summary : torch.Tensor) -> torch.distributions.Normal:
        """zeta: balancing variable for task imbalance
        \nits use: change the initial parameters for learning a task
        \nthere's one zeta for each task!
        \nuse result from statistics_pooling_2()"""
        task_summary_embedding = self.network_zeta(task_summary)
        mu = torch.squeeze(self.network_zeta_mean(task_summary_embedding))
        sigma = torch.squeeze(self.network_zeta_std(task_summary_embedding))
        q = torch.distributions.Normal(mu, torch.nn.functional.softplus(sigma))
        return q
    def get_posterior_distribution(self, x : torch.Tensor, y : torch.Tensor) -> tuple[
        torch.distributions.Normal, torch.distributions.Normal, torch.distributions.Normal
    ]:
        """get approximation of posterior distribution, parametrized by ψ Psi
        \nmeta-learning goal: maximize conditional log likelihood, given by
        \nlog(p(train label, test label | train input, test input; theta θ))
        \nfor this, we need TRUE posterior p(φ|train data, test data)
        \nfor task Tau τ, Psi φ is made of ω, γ, and z 
        \nwe get approximation of this posterior, p(φ|train data, test data; parametrized by Psi ψ)
        \nwe can drop (dependency of) the test data term, so the pipeline for generating φ and for inferring φ is the same"""
        class_summary, Cardinality_list = self.statistics_pooling_1(x, y)
        task_summary = self.statistics_pooling_2(class_summary, Cardinality_list)
        omega_distribution = self.get_omega(class_summary)
        gamma_distribution = self.get_gamma(task_summary)
        zeta_distribution = self.get_zeta(task_summary)
        return omega_distribution, gamma_distribution, zeta_distribution
    def forward(self, x : torch.Tensor, y : torch.Tensor, do_sample : bool) -> tuple[
        torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        omega_distribution, gamma_distribution, zeta_distribution = self.get_posterior_distribution(x, y)
        # get kl divergence
        omega_KL = torch.sum(KL_Diagonal_StandardNormal(omega_distribution))
        gamma_KL = torch.sum(KL_Diagonal_StandardNormal(gamma_distribution))
        zeta_KL = torch.sum(KL_Diagonal_StandardNormal(zeta_distribution))
        # sample variable from posterior
        omega = torch.empty() # only for type-hinting
        KL = torch.empty() # only for type-hinting
        if self.use_o == True:
            KL += omega_KL
            # .mean is a Tensor property of the distribution class. 
            # DON'T do .mean() <- get 'Tensor not callable'
            # source: https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.mean
            omega = omega_distribution.sample() if do_sample else omega_distribution.mean
        gamma = {}
        if self.use_g == True:
            KL += gamma_KL
            g = gamma_distribution.sample() if do_sample else gamma_distribution.mean
            # five 1's because gamma dimension is 5! see __init__()
            g = torch.split(g, [1, 1, 1, 1, 1])
            for l in [1, 2, 3, 4]:
                gamma[f'convolution_{l}_weight'] = g[l - 1]
                gamma[f'convolution_{l}_bias'] = g[l - 1]
            gamma[f'dense_weight'] = g[4]
            gamma[f'dense_bias'] = g[4]
        zeta = {}
        if self.use_z == True:
            KL += zeta_KL
            z = zeta_distribution.sample() if do_sample else zeta_distribution.mean
            # five 1's because gamma dimension is 5! see __init__()
            z_weight = torch.split(z[:self.number_of_channels * 4], [self.number_of_channels] * 4)
            z_bias = torch.split(z[self.number_of_channels * 4:], [self.number_of_channels] * 4)
            for l in [1, 2, 3, 4]:
                zeta[f'convolution_{l}_weight'] = z_weight[l - 1]
                zeta[f'convolution_{l}_bias'] = z_bias[l - 1]
        return omega, gamma, zeta, KL


