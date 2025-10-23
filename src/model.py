"""src/model.py
Neural-network architectures used in the experiments.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn
from omegaconf import DictConfig

################################################################################
#                               architectures                                  #
################################################################################

class TwoLayerMLP(nn.Module):
    """Simple two-layer perceptron (~1.2 M parameters for 256 units)."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        output_classes: int,
        activation: str = "relu",
        weight_init: str = "kaiming_uniform",
    ) -> None:
        super().__init__()
        act_cls: Any = getattr(nn, activation.capitalize(), nn.ReLU)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_units),
            act_cls(),
            nn.Linear(hidden_units, output_classes),
        )
        self.apply(lambda m: self._init(m, weight_init))

    @staticmethod
    def _init(m: nn.Module, scheme: str):
        if isinstance(m, nn.Linear):
            if scheme == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif scheme == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            else:
                raise ValueError(f"Weight-init scheme '{scheme}' not recognised.")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):  # noqa: D401 – trivial forward
        return self.net(x)

################################################################################
#                               factory                                        #
################################################################################

def build_model(cfg: DictConfig):
    name = cfg.name.lower()
    if name == "two-layer-mlp-1.2m" or name == "two_layer_mlp_1.2m":
        return TwoLayerMLP(
            input_dim=cfg.input_dim,
            hidden_units=cfg.hidden_units,
            output_classes=cfg.output_classes,
            activation=cfg.activation,
            weight_init=cfg.weight_init,
        )
    raise NotImplementedError(f"Unknown model: {cfg.name}")
