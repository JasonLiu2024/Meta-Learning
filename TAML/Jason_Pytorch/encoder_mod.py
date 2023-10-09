"""Inference network
- 
Content:
    1 ```Inference Network```"""
import torch
import torch.nn as nn

def statistics_pooling(incoming : torch.Tensor, cardinality : torch.Tensor):
    variance, mean = torch.var_mean(input=incoming, dim=0)
    cardinality = torch.tile(input=cardinality, dims=mean.shape)
    return torch.stack([mean, variance, cardinality], 1)

def initialize_weights_encoder(layer : torch.nn.Module) -> None:
    """Initialize weights for encoder networks NN1, NN2"""
    if isinstance(layer, nn.Conv2d):
        torch.nn.init.normal_(tensor=layer.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(tensor=layer.bias) # we're using bias
    elif isinstance(layer, nn.Linear):  
        torch.nn.init.trunc_normal_(tensor=layer.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(tensor=layer.bias)

class InferenceNetwork(nn.Module):
    """Encodes ONE Class (weight shared for all classes)
    -
    Inputs:
        1 ```number of ways``` (int)\n
        2 ```number of shots``` (int)\n
        3 ```use zeta``` (bool)\n
        4 ```use gamma``` (bool)\n
        5 ```use omega``` (bool)
    Forward() arguments
        1 ```incoming``` (torch.Tensor): input data, shape(N, C, H, W)
        2 ```labels``` (torch.Tensor): shape (N, 1)
        3 ```we_are_training``` (bool): if training, we sample from distributions """
    def __init__(self, number_of_ways : int, max_shots : int, input_channels : int, 
        hidden_channels : int, use_zeta : bool, use_gamma : bool, use_omega : bool, 
        main_network_number_of_convolution_layers : int=4,
        main_network_number_of_hidden_channels : int=32,
        device : str='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.use_zeta = use_zeta
        self.use_gamma = use_gamma
        self.use_omega = use_omega
        self.number_of_ways = number_of_ways
        self.max_shots = max_shots
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        # information about main network
        self.main_network_number_of_convolution_layers = main_network_number_of_convolution_layers
        self.main_network_number_of_hidden_channels =main_network_number_of_hidden_channels
        # summarize all classes
        self.NN1_part_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=640, out_features=64))
        self.NN1_part_1.apply(initialize_weights_encoder)
        # summarize the dataset
        self.NN1_part_2 = nn.Sequential(
            nn.Linear(in_features=3, out_features=4),
            nn.ReLU())
        self.NN1_part_2.apply(initialize_weights_encoder)
        self.NN2_part_1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32))
        self.NN2_part_1.apply(initialize_weights_encoder)
        self.NN2_part_2 = nn.Sequential(
            nn.Linear(in_features=3, out_features=4),
            nn.ReLU())
        self.NN2_part_2.apply(initialize_weights_encoder)
        # omega
        self.NN_omega_optional = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU())
        self.NN_omega_optional.apply(initialize_weights_encoder)
        self.NN_omega_mu = nn.Sequential(
            nn.Linear(in_features=64 if self.use_omega else 256, out_features=1),
            nn.ReLU())
        self.NN_omega_mu.apply(initialize_weights_encoder)
        self.NN_omega_sigma = nn.Sequential(
            nn.Linear(in_features=64 if self.use_omega else 256, out_features=1),
            nn.ReLU())
        # gamma
        self.NN_gamma_optional = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU())
        self.NN_gamma_optional.apply(initialize_weights_encoder)
        self.NN_gamma_mu = nn.Sequential(
            nn.Linear(in_features=64, out_features=5),
            nn.ReLU())
        self.NN_gamma_mu.apply(initialize_weights_encoder)
        self.NN_gamma_sigma = nn.Sequential(
            nn.Linear(in_features=64, out_features=5),
            nn.ReLU())
        self.NN_gamma_sigma.apply(initialize_weights_encoder)
        # zeta
        self.NN_zeta_optional = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU())
        self.NN_zeta_optional.apply(initialize_weights_encoder)
        self.NN_zeta_mu = nn.Sequential(
            nn.Linear(in_features=64, out_features=2*32*4),
            nn.ReLU())
        self.NN_zeta_mu.apply(initialize_weights_encoder)
        self.NN_zeta_sigma = nn.Sequential(
            nn.Linear(in_features=64, out_features=2*32*4),
            nn.ReLU())
        self.NN_zeta_sigma.apply(initialize_weights_encoder)

    def get_omega_posterior(self, all_classes_summary : torch.Tensor) -> torch.distributions.Distribution:
        all_class_summary_encoded = self.NN_omega_optional(all_classes_summary) if self.use_omega else all_classes_summary
        # print("all class summary encoded ", all_class_summary_encoded.shape)
        mu = torch.squeeze(self.NN_omega_mu(all_class_summary_encoded))
        sigma = torch.squeeze(self.NN_omega_sigma(all_class_summary_encoded))
        # print("omega: ", mu.shape, sigma.shape)
        posterior_omega = torch.distributions.Normal(loc=mu, scale=torch.nn.functional.softplus(sigma))
        return posterior_omega
    def get_gamma_posterior(self, dataset_summary : torch.Tensor) -> torch.distributions.Distribution:
        dataset_summary_encoded = self.NN_gamma_optional(dataset_summary)
        # print("all class summary encoded ", all_class_summary_encoded.shape)
        mu = torch.squeeze(self.NN_gamma_mu(dataset_summary_encoded))
        sigma = torch.squeeze(self.NN_gamma_sigma(dataset_summary_encoded))
        # print("gamma: ", mu.shape, sigma.shape)
        posterior_gamma = torch.distributions.Normal(loc=mu, scale=torch.nn.functional.softplus(sigma))
        return posterior_gamma
    def get_zeta_posterior(self, dataset_summary : torch.Tensor) -> torch.distributions.Distribution:
        dataset_summary_encoded = self.NN_zeta_optional(dataset_summary)
        # print("all class summary encoded ", all_class_summary_encoded.shape)
        mu = torch.squeeze(self.NN_zeta_mu(dataset_summary_encoded))
        sigma = torch.squeeze(self.NN_zeta_sigma(dataset_summary_encoded))
        # print("zeta: ", mu.shape, sigma.shape)
        posterior_zeta = torch.distributions.Normal(loc=mu, scale=torch.nn.functional.softplus(sigma))
        return posterior_zeta
    def get_kl_divergence(self, posterior_distribution : torch.distributions.Distribution) -> torch.Tensor:
        posterior_shape = posterior_distribution.mean.shape
        standard_normal_distribution = torch.distributions.Normal(torch.zeros(size=posterior_shape).to(self.device), torch.ones(posterior_shape).to(self.device))
        # find KL(q || p)
        return torch.distributions.kl_divergence(q=posterior_distribution, p=standard_normal_distribution)
    def forward(self, incoming : torch.Tensor, labels : torch.Tensor, we_are_training : bool) -> tuple[
        None | torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], float | torch.Tensor]:
        assert incoming.dim() == 4 # shape=(N, C, H, W)
        assert labels.dim() == 2 # shape=(N, 1)
        """summarize all classes"""
        examples_encoded = self.NN1_part_1(incoming)
        # y_sum = torch.sum(labels, 0) # not needed?
        # y_ = torch.argmax(labels, 1) # the paper uses one-hot encoding, so class 5 is [0, 0, 0, 0, 1]
        labels_flattened = torch.flatten(labels) # but we're using numbers, so class 5 is just 4 (indexed from 0)
        set_size_normalized_list : list[torch.Tensor] = []
        class_summary_list = []
        for this_class in range(self.number_of_ways):
            # torch.equal gives ONE boolean value
            # index_for_this_class = torch.logical_and(torch.eq(labels_flattened, c), torch.greater(y_sum, 0)) # not needed
            index_for_this_class = torch.eq(labels_flattened, this_class)
            examples_encoded_matching_this_class = examples_encoded[index_for_this_class] # x's corresponding to class c
            # y_matching_this_class = labels[index_for_this_class] # not used!
            set_size = torch.sum(index_for_this_class[index_for_this_class == True]) # so good
            # print(f"\tset size: {set_size}, dtype: {set_size.dtype}")
            # N_c = (torch.sum(y_matching_this_class) - 1.0) / (self.max_shots - 1.0) # normalized set size
            set_size_normalized = set_size / (self.max_shots - 1.0)
            set_size_normalized_list.append(set_size)
            summary_for_this_class = statistics_pooling(incoming=examples_encoded_matching_this_class, cardinality=set_size_normalized)
            class_summary_list.append(summary_for_this_class)
        all_classes_summary = torch.stack(tensors=class_summary_list, dim=0)
        # print("\tstacked dimension: ", all_classes_summary.shape)
        all_classes_summary : torch.Tensor = self.NN1_part_2(all_classes_summary)
        all_classes_summary = torch.reshape(input=all_classes_summary, shape=(self.number_of_ways, -1))
        # print("\tstage 1 result: ", all_classes_summary.shape)
        """summarize entire dataset"""
        dataset_encoded = self.NN2_part_1(all_classes_summary)
        # print("\tdataset encoding shape: ", dataset_encoded.shape)
        # pytorch NO can calculate mean of integers (int64 or long); so I sum them and take average instead!
        average_class_set_size = torch.sum(input=torch.stack(set_size_normalized_list, dim=0)) / self.number_of_ways
        # print("\taverage class set size: ", average_class_set_size)
        dataset_summary = statistics_pooling(incoming=dataset_encoded, cardinality=average_class_set_size)
        # print("\tdataset summary (post stats pooling): ", dataset_summary.shape)
        dataset_summary = torch.unsqueeze(input=dataset_summary, dim=0)
        # print("\tdataset summary reshaped: ", dataset_summary.shape)
        dataset_summary = self.NN2_part_2(dataset_summary)
        dataset_summary = torch.reshape(dataset_summary, shape=(1, -1))
        # print("\tdataset summary reshaped: ", dataset_summary.shape)
        """get balancing variables"""
        omega_posterior = self.get_omega_posterior(all_classes_summary=all_classes_summary)
        gamma_posterior = self.get_gamma_posterior(dataset_summary=dataset_summary)
        zeta_posterior = self.get_zeta_posterior(dataset_summary=dataset_summary)
        omega_KL = self.get_kl_divergence(omega_posterior)
        gamma_KL = self.get_kl_divergence(gamma_posterior)
        zeta_KL = self.get_kl_divergence(zeta_posterior)
        KL = 0.0
        omega = None
        if self.use_omega:
            KL += torch.sum(omega_KL)
            omega : torch.Tensor = omega_posterior.sample() if we_are_training else omega_posterior.mean
        gamma = None
        if self.use_gamma:
            KL += torch.sum(gamma_KL)
            gamma_raw = gamma_posterior.sample() if we_are_training else gamma_posterior.mean
            gamma_split : list[torch.Tensor] = torch.split(gamma_raw, split_size_or_sections=[1] * 5)
            gamma : dict[str, torch.Tensor] = {}
            for layer in range(self.main_network_number_of_convolution_layers):
                gamma[f'convolution_weight{layer}'] = gamma_split[layer]
                gamma[f'convolution_bias{layer}'] = gamma_split[layer]
            gamma[f'linear_weight'] = gamma_split[-1]
            gamma[f'linear_bias'] = gamma_split[-1]
        zeta = None
        if self.use_zeta:
            KL += torch.sum(zeta_KL)
            zeta_raw = zeta_posterior.sample() if we_are_training else zeta_posterior.mean
            zeta_weight_split = torch.split(zeta_raw[:self.main_network_number_of_hidden_channels * 4],
                [self.main_network_number_of_hidden_channels] * self.main_network_number_of_convolution_layers)
            zeta_bias_split = torch.split(zeta_raw[self.main_network_number_of_hidden_channels * 4:],
                [self.main_network_number_of_hidden_channels] * self.main_network_number_of_convolution_layers)
            zeta : dict[str, torch.Tensor] = {}
            for layer in range(self.main_network_number_of_convolution_layers):
                zeta[f'convolution_weight{layer}'] = zeta_weight_split[layer]
                zeta[f'convolution_bias{layer}'] = zeta_bias_split[layer]
        # print(f"KL: {KL}")
        return omega, gamma, zeta, KL
