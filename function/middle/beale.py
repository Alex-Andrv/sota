import torch

from function.function import Function


class Beale(Function):

    def __init__(self, start_point: torch.Tensor, a: float = 1.5, b: float = 2.25, c: float = 2.625, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)
        self.a = a
        self.b = b
        self.c = c

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        term1 = (self.a - x[0] + x[0] * x[1]) ** 2
        term2 = (self.b - x[0] + x[0] * x[1] ** 2) ** 2
        term3 = (self.c - x[0] + x[0] * x[1] ** 3) ** 2
        return term1 + term2 + term3