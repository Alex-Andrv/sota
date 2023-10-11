import torch

from function.function import Function


class Matyas(Function):
    def __init__(self, start_point: torch.Tensor, a: float = 0.26, b: float = 0.48, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = torch.nn.Parameter(start_point)
        self.a = a
        self.b = b

    def calculate(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.a * (x[0] ** 2 + x[1] ** 2)
        term2 = self.b * x[0] * x[1]
        return term1 - term2