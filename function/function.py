import torch
from torch import nn


class Function(nn.Module):

    def forward(self) -> torch.Tensor:
        return self.calculate(self.x)

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError