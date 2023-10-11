import torch

from function.function import Function


class Bohachevsky(Function):

    def __init__(self, start_point: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        term1 = x[0] ** 2
        term2 = 2 * x[1] ** 2
        term3 = -0.3 * torch.cos(3 * torch.tensor(2 * torch.pi) * x[0])
        term4 = -0.4 * torch.cos(4 * torch.tensor(2 * torch.pi) * x[1])
        return term1 + term2 + term3 + term4 + 0.7