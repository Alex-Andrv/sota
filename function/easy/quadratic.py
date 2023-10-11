import torch

from function.function import Function


class Quadratic(Function):
    def __init__(self, start_point: torch.Tensor, diagonal_values: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)
        self.diagonal = diagonal_values

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        return x[0] * self.diagonal[0] * x[0] + x[1] * self.diagonal[1] * x[1]