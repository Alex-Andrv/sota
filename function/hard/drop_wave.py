import numpy as np
import torch

from function.function import Function


class DropWave(Function):
    def __init__(self, start_point: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum(x ** 2, dim=0)
        numerator = 1 + torch.cos(12 * torch.sqrt(norm_squared))
        denominator = 0.5 * (norm_squared) + 2
        result = -numerator / denominator
        return result
