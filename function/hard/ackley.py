import numpy as np
import torch

from function.function import Function


class Ackley(Function):
    def __init__(self, start_point: torch.Tensor, a: int = 20, b: int = 0.2, c: int = 2 * np.pi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)
        self.a = a
        self.b = b
        self.c = c
        self.dim = len(self.x)

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        sum_sq_term = -self.a * torch.exp(-self.b * torch.sqrt(torch.mean(x ** 2, dim=0)))
        cos_term = -torch.exp(torch.mean(torch.cos(self.c * x), dim=0))

        result = self.a + torch.exp(torch.ones(1)) + sum_sq_term + cos_term

        return result