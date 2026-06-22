import torch
from torch.distributions import Gamma, Pareto, Cauchy, Normal
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


class WeightInitializer(ABC):
    @abstractmethod
    def initialize(self, input_size: int, output_size: int) -> torch.Tensor:
        pass


class RandnWeight(WeightInitializer):
    def __init__(self, scale: float = 0.5, sparsity: float = 0.0, mu: float = 0, sigma: float = 1.0):
        self.scale = scale
        self.sparsity = sparsity
        self.mu = mu
        self.sigma = sigma

    def initialize(self, input_size: int, output_size: int) -> torch.Tensor:
        weights = self.scale * Normal(loc=self.mu, scale=self.sigma).sample(
            torch.Size([input_size, output_size])
        )
        if self.sparsity > 0:
            mask = torch.rand(input_size, output_size) < self.sparsity
            weights *= mask.float()

        return weights


class GammaWeightRandMask(WeightInitializer):
    def __init__(self, alpha: float = 2.0, beta: float = 1.0, scale: float = 0.5, sparsity: float = 0.0):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.sparsity = sparsity

    def initialize(self, input_size: int, output_size: int) -> torch.Tensor:
        weights = self.scale * Gamma(concentration=self.alpha, rate=self.beta).sample(
            torch.Size([input_size, output_size])
        )

        negative_mask = torch.rand(input_size, output_size) < 0.5
        weights[negative_mask] = -weights[negative_mask]

        if self.sparsity > 0:
            mask = torch.rand(input_size, output_size) < self.sparsity
            weights *= mask.float()

        return weights


class GammaWeightRandnMask(WeightInitializer):
    def __init__(self, alpha: float = 2.0, beta: float = 1.0, scale: float = 0.5, sparsity: float = 0.0):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.sparsity = sparsity

    def initialize(self, input_size: int, output_size: int) -> torch.Tensor:
        weights = self.scale * Gamma(concentration=self.alpha, rate=self.beta).sample(
            torch.Size([input_size, output_size])
        )

        negative_mask = torch.randn(input_size, output_size) < 0.5
        weights[negative_mask] = -weights[negative_mask]

        if self.sparsity > 0:
            mask = torch.randn(input_size, output_size) < self.sparsity
            weights *= mask.float()

        return weights



class ParetoWeightRandnMask(WeightInitializer):
    def __init__(self, alpha: float = 0.2, xm: float = 0.5,
                 scale_factor: float = 0.5, sparsity: float = 0.0):
        self.alpha = alpha
        self.xm = xm
        self.scale_factor = scale_factor
        self.sparsity = sparsity

    def initialize(self, input_size: int, output_size: int) -> torch.Tensor:
        pareto_dist = Pareto(scale=self.xm, alpha=self.alpha)
        weights = pareto_dist.sample(torch.Size([input_size, output_size]))

        weights = self.scale_factor * weights

        negative_mask = torch.randn(input_size, output_size) < 0.5
        weights[negative_mask] = -weights[negative_mask]

        if self.sparsity > 0:
            mask = torch.rand(input_size, output_size) < self.sparsity
            weights *= mask.float()

        return weights


class CauchyWeightRandnMask(WeightInitializer):
    def __init__(self, loc: float = 0, scale: float = 0.1,
                 scale_factor: float = 0.5, sparsity: float = 0.0):
        self.loc = loc
        self.scale = scale
        self.scale_factor = scale_factor
        self.sparsity = sparsity

    def initialize(self, input_size: int, output_size: int) -> torch.Tensor:
        cauchy_dist = Cauchy(loc=self.loc, scale=self.scale)
        weights = cauchy_dist.sample(torch.Size([input_size, output_size]))

        weights = self.scale_factor * weights

        sign_mask = torch.rand(input_size, output_size) < 0.5
        weights[sign_mask] = -weights[sign_mask]

        # 应用稀疏化
        if self.sparsity > 0:
            sparsity_mask = torch.rand(input_size, output_size) < self.sparsity
            weights *= sparsity_mask.float()

        return weights


class LiquidStateNetwork:
    def __init__(self, dt: float = 1.0):
        self.network = Network(dt=dt)
        self.layers: Dict[str, Any] = {}
        self.attention_modules = {}

    def add_input_layer(self, size: int, shape: tuple, name: str):
        """添加输入层"""
        input_layer = Input(size, shape=shape)
        self.network.add_layer(input_layer, name=name)
        self.layers[name] = input_layer
        return input_layer

    def create_liquid_layer(self,
                            num_neurons: int,
                            name: str,
                            thresh_mean: float = -52,
                            thresh_std: float = 1.0) -> LIFNodes:

        output_layer = LIFNodes(
            num_neurons,
            thresh=thresh_mean + thresh_std * np.random.randn(num_neurons).astype(float)
            # thresh=thresh_mean
        )
        self.network.add_layer(output_layer, name=name)
        self.layers[name] = output_layer
        return output_layer


    def connect_layers(self,
                       source_name: str,
                       target_name: str,
                       weight_initializer: WeightInitializer,
                       connection_type: str = "feedforward") -> None:

        source = self.layers[source_name]
        target = self.layers[target_name]

        if connection_type == "feedforward":
            weights = weight_initializer.initialize(source.n, target.n)
        elif connection_type == "recurrent":
            weights = weight_initializer.initialize(source.n, target.n)
            weights = torch.triu(weights, diagonal=1) + torch.tril(weights, diagonal=-1)
        else:
            raise ValueError(f"Unsupported connection type: {connection_type}")
        connection = Connection(
            source=source,
            target=target,
            w=weights
        )
        self.network.add_connection(connection, source=source_name, target=target_name)

    def get_network(self) -> Network:
        return self.network
