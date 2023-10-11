import torch

from function.function import Function


class Quadratic(Function):
    def __init__(self, start_point: torch.Tensor, diagonal_values: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)
        self.diagonal = torch.diag(diagonal_values)

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.diagonal).matmul(x)